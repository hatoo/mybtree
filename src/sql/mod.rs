use rkyv::rancor::Error;
use sqlparser::ast::{Expr, SelectItem};

use crate::{Column, ColumnType, Database, DatabaseError, DbTransaction, DbValue, Row, Schema};
use sqlparser::ast::Statement as SqlStatement;

use expr::{eval_expr_bool, eval_value_expr};
use scan::Scanner;
use table_source::TableSource;

pub mod expr;
mod scan;
pub(crate) mod table_source;

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
    #[error("no active transaction")]
    NoActiveTransaction,
    #[error("transaction already active")]
    TransactionAlreadyActive,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResultSet {
    pub rows: Vec<Vec<(String, DbValue)>>,
}

impl ResultSet {
    pub fn empty() -> Self {
        Self { rows: Vec::new() }
    }
}

fn project_row(
    src: &TableSource<'_>,
    projections: &[SelectItem],
) -> Result<Vec<(String, DbValue)>, SqlError> {
    let mut values = Vec::new();
    for item in projections {
        match item {
            SelectItem::Wildcard(_) => {
                values.extend(src.all_columns());
            }
            SelectItem::UnnamedExpr(Expr::Identifier(ident)) => {
                let val = src.resolve(&ident.value)?;
                values.push((ident.value.clone(), val));
            }
            SelectItem::UnnamedExpr(Expr::CompoundIdentifier(parts)) if parts.len() == 2 => {
                let val = src.resolve_qualified(&parts[0].value, &parts[1].value)?;
                values.push((parts[1].value.clone(), val));
            }
            SelectItem::ExprWithAlias {
                expr: Expr::Identifier(ident),
                alias,
            } => {
                let val = src.resolve(&ident.value)?;
                values.push((alias.value.clone(), val));
            }
            SelectItem::ExprWithAlias {
                expr: Expr::CompoundIdentifier(parts),
                alias,
            } if parts.len() == 2 => {
                let val = src.resolve_qualified(&parts[0].value, &parts[1].value)?;
                values.push((alias.value.clone(), val));
            }
            _ => return Err(SqlError::UnsupportedExpr),
        }
    }
    Ok(values)
}

pub fn execute<'a, const N: usize>(
    db: &'a Database<N>,
    tx: &mut Option<DbTransaction<'a, N>>,
    sql: &str,
) -> Result<ResultSet, SqlError> {
    use sqlparser::dialect::GenericDialect;
    use sqlparser::parser::Parser;

    let stmts =
        Parser::parse_sql(&GenericDialect {}, sql).map_err(|e| SqlError::Parse(e.to_string()))?;

    let mut last = ResultSet::empty();
    for stmt in stmts {
        let result = match stmt {
            SqlStatement::StartTransaction { .. } => {
                if tx.is_some() {
                    return Err(SqlError::TransactionAlreadyActive);
                }
                *tx = Some(db.begin_transaction());
                ResultSet::empty()
            }
            SqlStatement::Commit { .. } => {
                let t = tx.take().ok_or(SqlError::NoActiveTransaction)?;
                t.commit()?;
                ResultSet::empty()
            }
            SqlStatement::Rollback { .. } => {
                let t = tx.take().ok_or(SqlError::NoActiveTransaction)?;
                drop(t);
                ResultSet::empty()
            }
            other => {
                let auto = tx.is_none();
                if auto {
                    *tx = Some(db.begin_transaction());
                }
                let t = tx.as_mut().unwrap();
                let result = match other {
                    SqlStatement::Query(query) => execute_query(t, *query)?,
                    SqlStatement::CreateTable(ct) => execute_create_table(t, ct)?,
                    SqlStatement::Insert(insert) => execute_insert(t, insert)?,
                    SqlStatement::Update(update) => execute_update(t, update)?,
                    SqlStatement::Delete(delete) => execute_delete(t, delete)?,
                    SqlStatement::CreateIndex(ci) => execute_create_index(t, ci)?,
                    _ => return Err(SqlError::UnsupportedStatement),
                };
                if auto {
                    tx.take().unwrap().commit()?;
                }
                result
            }
        };
        if !result.rows.is_empty() {
            last = result;
        }
    }
    Ok(last)
}

fn execute_query<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    query: sqlparser::ast::Query,
) -> Result<ResultSet, SqlError> {
    use sqlparser::ast::{SetExpr, TableFactor};
    let SetExpr::Select(select) = *query.body else {
        return Err(SqlError::UnsupportedStatement);
    };

    if select.from.len() != 1 || !select.from[0].joins.is_empty() {
        return Err(SqlError::UnsupportedStatement);
    }

    let projections = select.projection;
    let filter = select.selection;

    match &select.from[0].relation {
        TableFactor::Table { name, alias, .. } => {
            let table_name = name
                .0
                .last()
                .and_then(|p| p.as_ident())
                .map(|i| i.value.clone())
                .ok_or(SqlError::UnsupportedStatement)?;
            let qualifier = alias
                .as_ref()
                .map(|a| a.name.value.clone())
                .unwrap_or_else(|| table_name.clone());

            tx.with_lock(|mut locked_tx| {
                let schema = locked_tx.get_schema(&table_name)?;
                let indexed_columns = locked_tx.get_indexed_columns(&table_name)?;

                let scanner =
                    Scanner::from_filter(&table_name, &schema, &indexed_columns, &filter);

                let mut rows = Vec::new();
                scanner.scan::<_, SqlError, N>(locked_tx, |_tx, _key, archived| {
                    let row: Row = rkyv::deserialize::<Row, Error>(archived)
                        .map_err(DatabaseError::Internal)?;
                    let src = TableSource::from_table(&qualifier, &schema, &row);
                    if let Some(expr) = &filter {
                        if !eval_expr_bool(expr, &src)? {
                            return Ok(false);
                        }
                    }
                    rows.push(project_row(&src, &projections)?);
                    Ok(false)
                })?;
                Ok(ResultSet { rows })
            })
        }
        TableFactor::Derived {
            subquery, alias, ..
        } => {
            let alias = alias
                .as_ref()
                .map(|a| a.name.value.clone())
                .ok_or(SqlError::UnsupportedStatement)?;

            let sub_result = execute_query(tx, *subquery.clone())?;

            let mut rows = Vec::new();
            for sub_row in &sub_result.rows {
                let src = TableSource::from_result_row(&alias, sub_row);
                if let Some(expr) = &filter {
                    if !eval_expr_bool(expr, &src)? {
                        continue;
                    }
                }
                rows.push(project_row(&src, &projections)?);
            }
            Ok(ResultSet { rows })
        }
        _ => Err(SqlError::UnsupportedStatement),
    }
}

fn execute_create_table<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    ct: sqlparser::ast::CreateTable,
) -> Result<ResultSet, SqlError> {
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
        Ok(ResultSet::empty())
    })
}

fn execute_insert<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    insert: sqlparser::ast::Insert,
) -> Result<ResultSet, SqlError> {
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

        Ok(ResultSet::empty())
    })
}

fn execute_update<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    update: sqlparser::ast::Update,
) -> Result<ResultSet, SqlError> {
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
        let indexed_columns = locked_tx.get_indexed_columns(&table_name)?;

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

        let scanner = Scanner::from_filter(&table_name, &schema, &indexed_columns, &filter);

        scanner.scan::<_, SqlError, N>(locked_tx, |mut tx, key, archived| {
            let mut row: Row =
                rkyv::deserialize::<Row, Error>(archived).map_err(DatabaseError::Internal)?;
            if let Some(expr) = &filter {
                let src = TableSource::from_table(&table_name, &schema, &row);
                if !eval_expr_bool(expr, &src)? {
                    return Ok(false);
                }
            }
            for &(col_idx, ref expr) in &assignments {
                row.values[col_idx] = eval_value_expr(expr)?;
            }
            tx.update(&table_name, key, &row)?;
            Ok(false)
        })?;
        Ok(ResultSet::empty())
    })
}

fn execute_create_index<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    ci: sqlparser::ast::CreateIndex,
) -> Result<ResultSet, SqlError> {
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
        Ok(ResultSet::empty())
    })
}

fn execute_delete<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    delete: sqlparser::ast::Delete,
) -> Result<ResultSet, SqlError> {
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
        let indexed_columns = locked_tx.get_indexed_columns(&table_name)?;

        let scanner = Scanner::from_filter(&table_name, &schema, &indexed_columns, &filter);

        scanner.scan::<_, SqlError, N>(locked_tx, |mut tx, key, archived| {
            if let Some(expr) = &filter {
                let row: Row =
                    rkyv::deserialize::<Row, Error>(archived).map_err(DatabaseError::Internal)?;
                let src = TableSource::from_table(&table_name, &schema, &row);
                if !eval_expr_bool(expr, &src)? {
                    return Ok(false);
                }
            }
            tx.delete(&table_name, key)?;
            Ok(false)
        })?;
        Ok(ResultSet::empty())
    })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::NamedTempFile;

    use crate::{Database, DbValue, Pager};

    use super::{ResultSet, execute};

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

    /// Helper to extract just the values from a ResultSet for easy assertion.
    fn values(rs: &ResultSet) -> Vec<Vec<DbValue>> {
        rs.rows.iter().map(|r| r.iter().map(|(_, v)| v.clone()).collect()).collect()
    }

    #[test]
    fn create_insert_select() {
        let (db, _temp) = open_db();

        execute(&db, &mut None, "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL)").unwrap();
        execute(&db, &mut None, "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)").unwrap();
        execute(&db, &mut None, "INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)").unwrap();

        let rs = execute(&db, &mut None, "SELECT name, age FROM users").unwrap();

        assert_eq!(rs.rows.len(), 2);
        assert_eq!(rs.rows[0][0], ("name".into(), DbValue::Text("Alice".into())));
        assert_eq!(rs.rows[0][1], ("age".into(), DbValue::Integer(30)));
        assert_eq!(
            values(&rs),
            vec![
                vec![DbValue::Text("Alice".into()), DbValue::Integer(30)],
                vec![DbValue::Text("Bob".into()), DbValue::Integer(25)],
            ]
        );
    }

    #[test]
    fn select_with_alias() {
        let (db, _temp) = open_db();

        execute(&db, &mut None, "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)").unwrap();
        execute(&db, &mut None, "INSERT INTO users (id, name) VALUES (1, 'Alice')").unwrap();

        let rs = execute(&db, &mut None, "SELECT name AS user_name FROM users").unwrap();
        assert_eq!(rs.rows.len(), 1);
        assert_eq!(rs.rows[0][0], ("user_name".into(), DbValue::Text("Alice".into())));
    }

    #[test]
    fn update_rows() {
        let (db, _temp) = open_db();

        execute(&db, &mut None, "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL)").unwrap();
        execute(&db, &mut None, "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)").unwrap();
        execute(&db, &mut None, "INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)").unwrap();

        execute(&db, &mut None, "UPDATE users SET age = 31 WHERE id = 1").unwrap();

        let rs = execute(&db, &mut None, "SELECT name, age FROM users").unwrap();
        assert_eq!(
            values(&rs),
            vec![
                vec![DbValue::Text("Alice".into()), DbValue::Integer(31)],
                vec![DbValue::Text("Bob".into()), DbValue::Integer(25)],
            ]
        );

        execute(&db, &mut None, "UPDATE users SET name = 'Unknown'").unwrap();

        let rs = execute(&db, &mut None, "SELECT name FROM users").unwrap();
        assert_eq!(
            values(&rs),
            vec![vec![DbValue::Text("Unknown".into())], vec![DbValue::Text("Unknown".into())]]
        );
    }

    #[test]
    fn delete_rows() {
        let (db, _temp) = open_db();

        execute(&db, &mut None, "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL)").unwrap();
        execute(&db, &mut None, "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)").unwrap();
        execute(&db, &mut None, "INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)").unwrap();
        execute(&db, &mut None, "INSERT INTO users (id, name, age) VALUES (3, 'Carol', 35)").unwrap();

        execute(&db, &mut None, "DELETE FROM users WHERE id = 1").unwrap();

        let rs = execute(&db, &mut None, "SELECT name FROM users").unwrap();
        assert_eq!(rs.rows.len(), 2);
        assert_eq!(values(&rs), vec![vec![DbValue::Text("Bob".into())], vec![DbValue::Text("Carol".into())]]);

        execute(&db, &mut None, "DELETE FROM users").unwrap();

        let rs = execute(&db, &mut None, "SELECT * FROM users").unwrap();
        assert_eq!(rs.rows.len(), 0);
    }

    #[test]
    fn begin_commit() {
        let (db, _temp) = open_db();

        execute(
            &db, &mut None,
            "BEGIN;
             CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
             INSERT INTO users (id, name) VALUES (1, 'Alice');
             COMMIT",
        )
        .unwrap();

        let rs = execute(&db, &mut None, "SELECT name FROM users").unwrap();
        assert_eq!(rs.rows.len(), 1);
        assert_eq!(rs.rows[0][0], ("name".into(), DbValue::Text("Alice".into())));
    }

    #[test]
    fn rollback() {
        let (db, _temp) = open_db();

        execute(
            &db, &mut None,
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
             INSERT INTO users (id, name) VALUES (1, 'Alice')",
        )
        .unwrap();

        execute(
            &db, &mut None,
            "BEGIN;
             INSERT INTO users (id, name) VALUES (2, 'Bob');
             ROLLBACK",
        )
        .unwrap();

        let rs = execute(&db, &mut None, "SELECT name FROM users").unwrap();
        assert_eq!(rs.rows.len(), 1);
        assert_eq!(rs.rows[0][0], ("name".into(), DbValue::Text("Alice".into())));
    }

    #[test]
    fn partial_transaction_across_calls() {
        let (db, _temp) = open_db();
        execute(&db, &mut None, "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)").unwrap();
        execute(&db, &mut None, "INSERT INTO users (id, name) VALUES (1, 'Alice')").unwrap();

        let mut tx = None;
        execute(&db, &mut tx, "BEGIN").unwrap();
        execute(&db, &mut tx, "INSERT INTO users (id, name) VALUES (2, 'Bob')").unwrap();

        let rs = execute(&db, &mut tx, "SELECT name FROM users").unwrap();
        assert_eq!(rs.rows.len(), 2);

        execute(&db, &mut tx, "COMMIT").unwrap();
        assert!(tx.is_none());

        let rs = execute(&db, &mut None, "SELECT name FROM users").unwrap();
        assert_eq!(rs.rows.len(), 2);
    }

    #[test]
    fn partial_transaction_rollback_across_calls() {
        let (db, _temp) = open_db();
        execute(&db, &mut None, "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)").unwrap();
        execute(&db, &mut None, "INSERT INTO users (id, name) VALUES (1, 'Alice')").unwrap();

        let mut tx = None;
        execute(&db, &mut tx, "BEGIN").unwrap();
        execute(&db, &mut tx, "INSERT INTO users (id, name) VALUES (2, 'Bob')").unwrap();
        execute(&db, &mut tx, "ROLLBACK").unwrap();
        assert!(tx.is_none());

        let rs = execute(&db, &mut None, "SELECT name FROM users").unwrap();
        assert_eq!(rs.rows.len(), 1);
        assert_eq!(rs.rows[0][0], ("name".into(), DbValue::Text("Alice".into())));
    }

    #[test]
    fn table_alias() {
        let (db, _temp) = open_db();
        execute(&db, &mut None, "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)").unwrap();
        execute(&db, &mut None, "INSERT INTO users (id, name) VALUES (1, 'Alice')").unwrap();

        // Qualified column with table alias
        let rs = execute(&db, &mut None, "SELECT u.name FROM users AS u").unwrap();
        assert_eq!(rs.rows.len(), 1);
        assert_eq!(rs.rows[0][0], ("name".into(), DbValue::Text("Alice".into())));

        // Qualified column with alias in WHERE
        let rs = execute(&db, &mut None, "SELECT u.name FROM users AS u WHERE u.id = 1").unwrap();
        assert_eq!(rs.rows.len(), 1);
        assert_eq!(rs.rows[0][0], ("name".into(), DbValue::Text("Alice".into())));

        // Qualified column with alias + AS in projection
        let rs = execute(&db, &mut None, "SELECT u.name AS user_name FROM users AS u").unwrap();
        assert_eq!(rs.rows[0][0], ("user_name".into(), DbValue::Text("Alice".into())));

        // PK range scan with alias
        execute(&db, &mut None, "INSERT INTO users (id, name) VALUES (2, 'Bob')").unwrap();
        let rs = execute(&db, &mut None, "SELECT u.name FROM users AS u WHERE u.id >= 1").unwrap();
        assert_eq!(rs.rows.len(), 2);
    }

    #[test]
    fn subquery() {
        let (db, _temp) = open_db();
        execute(&db, &mut None, "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL)").unwrap();
        execute(&db, &mut None, "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)").unwrap();
        execute(&db, &mut None, "INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)").unwrap();
        execute(&db, &mut None, "INSERT INTO users (id, name, age) VALUES (3, 'Carol', 35)").unwrap();

        // Subquery with alias
        let rs = execute(
            &db, &mut None,
            "SELECT t.name FROM (SELECT name, age FROM users WHERE age > 28) AS t",
        ).unwrap();
        assert_eq!(rs.rows.len(), 2);
        assert_eq!(rs.rows[0][0], ("name".into(), DbValue::Text("Alice".into())));
        assert_eq!(rs.rows[1][0], ("name".into(), DbValue::Text("Carol".into())));

        // Subquery with filter on outer query
        let rs = execute(
            &db, &mut None,
            "SELECT t.name FROM (SELECT name, age FROM users) AS t WHERE t.age > 28",
        ).unwrap();
        assert_eq!(rs.rows.len(), 2);
        assert_eq!(rs.rows[0][0], ("name".into(), DbValue::Text("Alice".into())));
        assert_eq!(rs.rows[1][0], ("name".into(), DbValue::Text("Carol".into())));

        // Subquery with wildcard
        let rs = execute(
            &db, &mut None,
            "SELECT * FROM (SELECT name FROM users WHERE id = 1) AS t",
        ).unwrap();
        assert_eq!(rs.rows.len(), 1);
        assert_eq!(rs.rows[0][0], ("name".into(), DbValue::Text("Alice".into())));
    }
}
