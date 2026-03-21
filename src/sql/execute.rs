use rkyv::rancor::Error;
use sqlparser::ast::Expr;

use crate::{Column, ColumnType, DatabaseError, DbTransaction, DbValue, Row, Schema};

use super::expr::{eval_expr_bool, eval_value_expr};
use super::scan::Scanner;
use super::table_source::TableSource;
use super::{ResultSet, SqlError};

pub(super) fn execute_create_table<const N: usize>(
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

pub(super) fn execute_insert<const N: usize>(
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

pub(super) fn execute_update<const N: usize>(
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

        scanner.scan::<_, SqlError, N>(&mut locked_tx, |tx, key, archived| {
            let mut row: Row =
                rkyv::deserialize::<Row, Error>(archived).map_err(DatabaseError::Internal)?;
            if let Some(expr) = &filter {
                let src = TableSource::from_table(&table_name, &schema, &row);
                if !eval_expr_bool(expr, &src, tx)? {
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

pub(super) fn execute_create_index<const N: usize>(
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

pub(super) fn execute_delete<const N: usize>(
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

        scanner.scan::<_, SqlError, N>(&mut locked_tx, |tx, key, archived| {
            if let Some(expr) = &filter {
                let row: Row =
                    rkyv::deserialize::<Row, Error>(archived).map_err(DatabaseError::Internal)?;
                let src = TableSource::from_table(&table_name, &schema, &row);
                if !eval_expr_bool(expr, &src, tx)? {
                    return Ok(false);
                }
            }
            tx.delete(&table_name, key)?;
            Ok(false)
        })?;
        Ok(ResultSet::empty())
    })
}
