use std::ops::Bound;

use rkyv::rancor::Error;
use sqlparser::ast::{Expr, SelectItem};

use crate::{
    Column, ColumnType, DatabaseError, DbTransaction, DbValue, Key, Row, Schema,
};

use expr::{eval_expr_bool, eval_value_expr};
use scan::{Scanner, find_best_scanner};

pub mod expr;
mod scan;

#[derive(thiserror::Error, Debug)]
pub enum SqlError {
    #[error("Database error: {0}")]
    Database(#[from] crate::DatabaseError),
    #[error("column not found: {0}")]
    ColumnNotFound(String),
    #[error("unsupported expression")]
    UnsupportedExpr,
    #[error("parse error: {0}")]
    Parse(String),
    #[error("unsupported statement")]
    UnsupportedStatement,
}

fn project_row(schema: &Schema, row: &Row, projections: &[SelectItem]) -> Result<Row, SqlError> {
    let mut values = Vec::new();
    for item in projections {
        match item {
            SelectItem::Wildcard(_) => {
                for (i, _col) in schema.columns.iter().enumerate() {
                    if schema.implicit_pk && i == schema.primary_key {
                        continue;
                    }
                    values.push(row.values[i].clone());
                }
            }
            SelectItem::UnnamedExpr(Expr::Identifier(ident))
            | SelectItem::ExprWithAlias {
                expr: Expr::Identifier(ident),
                ..
            } => {
                let col_idx = schema
                    .columns
                    .iter()
                    .position(|c| c.name.eq_ignore_ascii_case(&ident.value))
                    .ok_or_else(|| SqlError::ColumnNotFound(ident.value.clone()))?;
                values.push(row.values[col_idx].clone());
            }
            _ => return Err(SqlError::UnsupportedExpr),
        }
    }
    Ok(Row { values })
}

pub fn execute<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    sql: &str,
) -> Result<Vec<Row>, SqlError> {
    use sqlparser::ast::Statement as SqlStatement;
    use sqlparser::dialect::GenericDialect;
    use sqlparser::parser::Parser;

    let stmts =
        Parser::parse_sql(&GenericDialect {}, sql).map_err(|e| SqlError::Parse(e.to_string()))?;

    let mut last = Vec::new();
    for stmt in stmts {
        last = match stmt {
            SqlStatement::Query(query) => execute_query(tx, *query)?,
            SqlStatement::CreateTable(ct) => execute_create_table(tx, ct)?,
            SqlStatement::Insert(insert) => execute_insert(tx, insert)?,
            SqlStatement::Update(update) => execute_update(tx, update)?,
            SqlStatement::Delete(delete) => execute_delete(tx, delete)?,
            SqlStatement::CreateIndex(ci) => execute_create_index(tx, ci)?,
            _ => return Err(SqlError::UnsupportedStatement),
        };
    }
    Ok(last)
}

fn execute_query<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    query: sqlparser::ast::Query,
) -> Result<Vec<Row>, SqlError> {
    use sqlparser::ast::{SetExpr, TableFactor};
    let SetExpr::Select(select) = *query.body else {
        return Err(SqlError::UnsupportedStatement);
    };

    if select.from.len() != 1 || !select.from[0].joins.is_empty() {
        return Err(SqlError::UnsupportedStatement);
    }
    let TableFactor::Table { name, .. } = &select.from[0].relation else {
        return Err(SqlError::UnsupportedStatement);
    };
    let table_name = name
        .0
        .last()
        .and_then(|p| p.as_ident())
        .map(|i| i.value.clone())
        .ok_or(SqlError::UnsupportedStatement)?;

    let projections = select.projection;
    let filter = select.selection;

    tx.with_lock(|mut locked_tx| {
        let schema = locked_tx.get_schema(&table_name)?;
        let indexed_columns = locked_tx.get_indexed_columns(&table_name)?;

        let scanner = filter
            .as_ref()
            .and_then(|f| find_best_scanner(&table_name, &schema, &indexed_columns, f))
            .unwrap_or_else(|| Scanner::PkRange {
                table: table_name.clone(),
                range: (Bound::Unbounded, Bound::Unbounded),
            });

        let mut rows = Vec::new();
        scanner.scan::<_, SqlError, N>(locked_tx, |_tx, _key, archived| {
            let row: Row = rkyv::deserialize::<Row, Error>(archived)
                .map_err(DatabaseError::Internal)?;
            if let Some(expr) = &filter {
                if !eval_expr_bool(expr, &schema, &row)? {
                    return Ok(false);
                }
            }
            rows.push(project_row(&schema, &row, &projections)?);
            Ok(false)
        })?;
        Ok(rows)
    })
}

fn execute_create_table<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    ct: sqlparser::ast::CreateTable,
) -> Result<Vec<Row>, SqlError> {
    use sqlparser::ast::{ColumnOption, DataType};

    let table_name = ct
        .name
        .0
        .last()
        .and_then(|p| p.as_ident())
        .map(|i| i.value.clone())
        .ok_or(SqlError::UnsupportedStatement)?;

    let mut columns = Vec::new();
    let mut pk_index: Option<usize> = None;

    for col_def in &ct.columns {
        let column_type = match &col_def.data_type {
            DataType::TinyInt(_)
            | DataType::SmallInt(_)
            | DataType::Int(_)
            | DataType::Integer(_)
            | DataType::BigInt(_) => ColumnType::Integer,
            DataType::Float(_)
            | DataType::Real
            | DataType::Double(_)
            | DataType::DoublePrecision => ColumnType::Float,
            DataType::Bool | DataType::Boolean => ColumnType::Bool,
            DataType::Text | DataType::Varchar(_) | DataType::Char(_) => ColumnType::Text,
            _ => return Err(SqlError::UnsupportedExpr),
        };

        let mut nullable = true;
        for opt in &col_def.options {
            match &opt.option {
                ColumnOption::NotNull => nullable = false,
                ColumnOption::PrimaryKey(_) => {
                    nullable = false;
                    pk_index = Some(columns.len());
                }
                _ => {}
            }
        }

        columns.push(Column {
            name: col_def.name.value.clone(),
            column_type,
            nullable,
        });
    }

    // Also check table-level PRIMARY KEY constraint.
    if pk_index.is_none() {
        use sqlparser::ast::{Expr, TableConstraint};
        for constraint in &ct.constraints {
            if let TableConstraint::PrimaryKey(pk) = constraint {
                if let Some(idx_col) = pk.columns.first() {
                    if let Expr::Identifier(ident) = &idx_col.column.expr {
                        pk_index = columns
                            .iter()
                            .position(|c| c.name.eq_ignore_ascii_case(&ident.value));
                        if let Some(idx) = pk_index {
                            columns[idx].nullable = false;
                        }
                    }
                }
                break;
            }
        }
    }

    let schema = Schema {
        columns,
        primary_key: 0, // overwritten by create_table
        implicit_pk: false,
    };

    tx.with_lock(|mut locked_tx| {
        locked_tx.create_table(&table_name, schema, pk_index)?;
        Ok(vec![])
    })
}

fn execute_insert<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    insert: sqlparser::ast::Insert,
) -> Result<Vec<Row>, SqlError> {
    use sqlparser::ast::{SetExpr, TableObject};

    let TableObject::TableName(obj_name) = insert.table else {
        return Err(SqlError::UnsupportedStatement);
    };
    let table_name = obj_name
        .0
        .last()
        .and_then(|p| p.as_ident())
        .map(|i| i.value.clone())
        .ok_or(SqlError::UnsupportedStatement)?;

    // Named columns from INSERT INTO t (col1, col2) — may be empty.
    let col_names: Vec<String> = insert.columns.iter().map(|i| i.value.clone()).collect();

    let SetExpr::Values(values_list) = *insert.source.ok_or(SqlError::UnsupportedStatement)?.body
    else {
        return Err(SqlError::UnsupportedStatement);
    };

    tx.with_lock(|mut locked_tx| {
        let schema = locked_tx.get_schema(&table_name)?;

        for value_row in &values_list.rows {
            // Build a full row initialised to Null.
            let mut values = vec![DbValue::Null; schema.columns.len()];

            if col_names.is_empty() {
                // No column list: values map positionally, skipping implicit PK.
                let start = if schema.implicit_pk { 1 } else { 0 };
                for (i, expr) in value_row.iter().enumerate() {
                    let col_idx = start + i;
                    if col_idx >= schema.columns.len() {
                        return Err(SqlError::UnsupportedExpr);
                    }
                    values[col_idx] = eval_value_expr(expr)?;
                }
            } else {
                for (col_name, expr) in col_names.iter().zip(value_row.iter()) {
                    let col_idx = schema
                        .columns
                        .iter()
                        .position(|c| c.name.eq_ignore_ascii_case(col_name))
                        .ok_or_else(|| SqlError::ColumnNotFound(col_name.clone()))?;
                    values[col_idx] = eval_value_expr(expr)?;
                }
            }

            locked_tx.insert(&table_name, &Row { values })?;
        }

        Ok(vec![])
    })
}

fn execute_update<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    update: sqlparser::ast::Update,
) -> Result<Vec<Row>, SqlError> {
    use sqlparser::ast::{AssignmentTarget, TableFactor};

    if !update.table.joins.is_empty() {
        return Err(SqlError::UnsupportedStatement);
    }
    let TableFactor::Table { name, .. } = &update.table.relation else {
        return Err(SqlError::UnsupportedStatement);
    };
    let table_name = name
        .0
        .last()
        .and_then(|p| p.as_ident())
        .map(|i| i.value.clone())
        .ok_or(SqlError::UnsupportedStatement)?;

    let filter = update.selection;

    tx.with_lock(|mut locked_tx| {
        let schema = locked_tx.get_schema(&table_name)?;

        // Resolve assignments to (column_index, expr) pairs.
        let mut assignments = Vec::new();
        for a in &update.assignments {
            let col_name = match &a.target {
                AssignmentTarget::ColumnName(obj) => obj
                    .0
                    .last()
                    .and_then(|p| p.as_ident())
                    .map(|i| &i.value)
                    .ok_or(SqlError::UnsupportedExpr)?,
                AssignmentTarget::Tuple(_) => return Err(SqlError::UnsupportedExpr),
            };
            let col_idx = schema
                .columns
                .iter()
                .position(|c| c.name.eq_ignore_ascii_case(col_name))
                .ok_or_else(|| SqlError::ColumnNotFound(col_name.clone()))?;
            assignments.push((col_idx, &a.value));
        }

        let mut to_update: Vec<(Key, Row)> = Vec::new();
        locked_tx.scan::<_, SqlError>(
            &table_name,
            (Bound::<Key>::Unbounded, Bound::<Key>::Unbounded),
            |_tx, key, archived| {
                let mut row: Row = rkyv::deserialize::<Row, Error>(archived)
                    .map_err(DatabaseError::Internal)?;
                if let Some(expr) = &filter {
                    if !eval_expr_bool(expr, &schema, &row)? {
                        return Ok(false);
                    }
                }
                for &(col_idx, ref expr) in &assignments {
                    row.values[col_idx] = eval_value_expr(expr)?;
                }
                to_update.push((key, row));
                Ok(false)
            },
        )?;
        for (key, row) in &to_update {
            locked_tx.update(&table_name, *key, row)?;
        }
        Ok(vec![])
    })
}

fn execute_create_index<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    ci: sqlparser::ast::CreateIndex,
) -> Result<Vec<Row>, SqlError> {
    let table_name = ci
        .table_name
        .0
        .last()
        .and_then(|p| p.as_ident())
        .map(|i| i.value.clone())
        .ok_or(SqlError::UnsupportedStatement)?;

    if ci.columns.len() != 1 {
        return Err(SqlError::UnsupportedStatement);
    }
    let col_expr = &ci.columns[0].column.expr;
    let Expr::Identifier(ident) = col_expr else {
        return Err(SqlError::UnsupportedExpr);
    };
    let col_name = &ident.value;

    tx.with_lock(|mut locked_tx| {
        locked_tx.create_index(&table_name, col_name)?;
        Ok(vec![])
    })
}

fn execute_delete<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    delete: sqlparser::ast::Delete,
) -> Result<Vec<Row>, SqlError> {
    use sqlparser::ast::{FromTable, TableFactor};

    let tables = match &delete.from {
        FromTable::WithFromKeyword(t) | FromTable::WithoutKeyword(t) => t,
    };
    if tables.len() != 1 || !tables[0].joins.is_empty() {
        return Err(SqlError::UnsupportedStatement);
    }
    let TableFactor::Table { name, .. } = &tables[0].relation else {
        return Err(SqlError::UnsupportedStatement);
    };
    let table_name = name
        .0
        .last()
        .and_then(|p| p.as_ident())
        .map(|i| i.value.clone())
        .ok_or(SqlError::UnsupportedStatement)?;

    let filter = delete.selection;

    tx.with_lock(|mut locked_tx| {
        let schema = locked_tx.get_schema(&table_name)?;

        let mut to_delete: Vec<Key> = Vec::new();
        locked_tx.scan::<_, SqlError>(
            &table_name,
            (Bound::<Key>::Unbounded, Bound::<Key>::Unbounded),
            |_tx, key, archived| {
                if let Some(expr) = &filter {
                    let row: Row = rkyv::deserialize::<Row, Error>(archived)
                        .map_err(DatabaseError::Internal)?;
                    if !eval_expr_bool(expr, &schema, &row)? {
                        return Ok(false);
                    }
                }
                to_delete.push(key);
                Ok(false)
            },
        )?;
        for key in &to_delete {
            locked_tx.delete(&table_name, *key)?;
        }
        Ok(vec![])
    })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::NamedTempFile;

    use crate::{Database, DbValue, Pager};

    use super::execute;

    fn open_db() -> (Database<4096>, NamedTempFile) {
        let temp = NamedTempFile::new().unwrap();
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(temp.path())
            .unwrap();
        let pager = Pager::<4096>::new(file);
        let db = Database::create(pager).unwrap();
        (db, temp)
    }

    #[test]
    fn create_insert_select() {
        let (db, _temp) = open_db();
        let mut tx = db.begin_transaction();

        execute(
            &mut tx,
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL)",
        )
        .unwrap();

        execute(
            &mut tx,
            "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)",
        )
        .unwrap();
        execute(
            &mut tx,
            "INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)",
        )
        .unwrap();
        tx.commit().unwrap();

        let mut tx = db.begin_transaction();

        let rows = execute(&mut tx, "SELECT name, age FROM users").unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0].values,
            vec![DbValue::Text("Alice".into()), DbValue::Integer(30)]
        );
        assert_eq!(
            rows[1].values,
            vec![DbValue::Text("Bob".into()), DbValue::Integer(25)]
        );

        tx.commit().unwrap();
    }

    #[test]
    fn update_rows() {
        let (db, _temp) = open_db();
        let mut tx = db.begin_transaction();

        execute(
            &mut tx,
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL)",
        )
        .unwrap();

        execute(
            &mut tx,
            "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)",
        )
        .unwrap();
        execute(
            &mut tx,
            "INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)",
        )
        .unwrap();

        // Update with WHERE clause
        execute(&mut tx, "UPDATE users SET age = 31 WHERE id = 1").unwrap();

        let rows = execute(&mut tx, "SELECT name, age FROM users").unwrap();
        assert_eq!(
            rows[0].values,
            vec![DbValue::Text("Alice".into()), DbValue::Integer(31)]
        );
        assert_eq!(
            rows[1].values,
            vec![DbValue::Text("Bob".into()), DbValue::Integer(25)]
        );

        // Update all rows
        execute(&mut tx, "UPDATE users SET name = 'Unknown'").unwrap();

        let rows = execute(&mut tx, "SELECT name FROM users").unwrap();
        assert_eq!(rows[0].values, vec![DbValue::Text("Unknown".into())]);
        assert_eq!(rows[1].values, vec![DbValue::Text("Unknown".into())]);

        tx.commit().unwrap();
    }

    #[test]
    fn delete_rows() {
        let (db, _temp) = open_db();
        let mut tx = db.begin_transaction();

        execute(
            &mut tx,
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL)",
        )
        .unwrap();

        execute(
            &mut tx,
            "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)",
        )
        .unwrap();
        execute(
            &mut tx,
            "INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)",
        )
        .unwrap();
        execute(
            &mut tx,
            "INSERT INTO users (id, name, age) VALUES (3, 'Carol', 35)",
        )
        .unwrap();

        // Delete with WHERE clause
        execute(&mut tx, "DELETE FROM users WHERE id = 1").unwrap();

        let rows = execute(&mut tx, "SELECT name FROM users").unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values, vec![DbValue::Text("Bob".into())]);
        assert_eq!(rows[1].values, vec![DbValue::Text("Carol".into())]);

        // Delete all remaining rows
        execute(&mut tx, "DELETE FROM users").unwrap();

        let rows = execute(&mut tx, "SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 0);

        tx.commit().unwrap();
    }
}
