use std::{borrow::Cow, cmp::Ordering, ops::Bound};

use rkyv::rancor::Error;
use sqlparser::ast::{BinaryOperator, Expr, SelectItem, Value};

use crate::{
    DatabaseError, DbTransaction, DbValue, Key, Row, Schema,
    database::{ArchivedRow, LockedDbTransaction},
};

enum Scanner {
    PkEqual {
        table: String,
        pk: Key,
    },
    PkRange {
        table: String,
        range: (Bound<Key>, Bound<Key>),
    },
    IndexEqual {
        table: String,
        index: String,
        value: Vec<u8>,
    },
    IndexRange {
        table: String,
        index: String,
        range: (Bound<Vec<u8>>, Bound<Vec<u8>>),
    },
}

enum Statement {
    Select {
        table: String,
        scanner: Scanner,
        projections: Vec<SelectItem>,
        filter: Option<Expr>,
    },
}

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

impl Scanner {
    fn scan<'a, F, E: From<DatabaseError>, const N: usize>(
        &self,
        mut locked_tx: LockedDbTransaction<'a, N>,
        mut f: F,
    ) -> Result<(), E>
    where
        F: for<'local> FnMut(
            LockedDbTransaction<'local, N>,
            Key,
            &'local ArchivedRow,
        ) -> Result<bool, E>,
    {
        match self {
            Scanner::PkEqual { table, pk } => {
                if let Some(value) = locked_tx.get_value(table, *pk)? {
                    let value = match value {
                        Cow::Borrowed(v) => v.to_vec(),
                        Cow::Owned(v) => v,
                    };
                    let value = rkyv::access::<rkyv::Archived<Row>, Error>(&value)
                        .map_err(|e| DatabaseError::Internal(e))?;
                    f(locked_tx, *pk, value)?;
                }
            }
            Scanner::PkRange { table, range } => {
                locked_tx.scan(table.as_str(), *range, f)?;
            }
            Scanner::IndexEqual {
                table,
                index,
                value,
            } => {
                let v = value.as_slice();
                locked_tx.scan_by_index(
                    table.as_str(),
                    index.as_str(),
                    v..=v,
                    |tx, archived, key| f(tx, key, archived),
                )?;
            }
            Scanner::IndexRange {
                table,
                index,
                range,
            } => {
                fn map_bound(b: &Bound<Vec<u8>>) -> Bound<&[u8]> {
                    match b {
                        Bound::Included(v) => Bound::Included(v.as_slice()),
                        Bound::Excluded(v) => Bound::Excluded(v.as_slice()),
                        Bound::Unbounded => Bound::Unbounded,
                    }
                }
                locked_tx.scan_by_index(
                    table.as_str(),
                    index.as_str(),
                    (map_bound(&range.0), map_bound(&range.1)),
                    |tx, archived, key| f(tx, key, archived),
                )?;
            }
        }
        Ok(())
    }
}

impl Statement {
    fn execute<'a, const N: usize>(
        &self,
        mut locked_tx: LockedDbTransaction<'a, N>,
    ) -> Result<Vec<Row>, SqlError> {
        match self {
            Statement::Select {
                table,
                scanner,
                projections,
                filter,
            } => {
                let schema = locked_tx.get_schema(table)?;
                let mut rows = Vec::new();
                scanner.scan::<_, SqlError, N>(locked_tx, |_tx, _key, archived| {
                    let row: Row = rkyv::deserialize::<Row, Error>(archived)
                        .map_err(DatabaseError::Internal)?;
                    if let Some(expr) = filter {
                        if !eval_expr_bool(expr, &schema, &row)? {
                            return Ok(true);
                        }
                    }
                    rows.push(project_row(&schema, &row, projections)?);
                    Ok(true)
                })?;
                Ok(rows)
            }
        }
    }
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

fn eval_value(v: &Value) -> Result<DbValue, SqlError> {
    match v {
        Value::Number(n, _) => {
            if let Ok(i) = n.parse::<i64>() {
                Ok(DbValue::Integer(i))
            } else if let Ok(f) = n.parse::<f64>() {
                Ok(DbValue::Float(f))
            } else {
                Err(SqlError::UnsupportedExpr)
            }
        }
        Value::SingleQuotedString(s) | Value::DoubleQuotedString(s) => {
            Ok(DbValue::Text(s.clone()))
        }
        Value::Boolean(b) => Ok(DbValue::Bool(*b)),
        Value::Null => Ok(DbValue::Null),
        _ => Err(SqlError::UnsupportedExpr),
    }
}

fn eval_expr_value(expr: &Expr, schema: &Schema, row: &Row) -> Result<DbValue, SqlError> {
    match expr {
        Expr::Identifier(ident) => {
            let col_idx = schema
                .columns
                .iter()
                .position(|c| c.name.eq_ignore_ascii_case(&ident.value))
                .ok_or_else(|| SqlError::ColumnNotFound(ident.value.clone()))?;
            Ok(row.values[col_idx].clone())
        }
        Expr::CompoundIdentifier(parts) => {
            let name = parts.last().map(|i| i.value.as_str()).unwrap_or("");
            let col_idx = schema
                .columns
                .iter()
                .position(|c| c.name.eq_ignore_ascii_case(name))
                .ok_or_else(|| SqlError::ColumnNotFound(name.to_string()))?;
            Ok(row.values[col_idx].clone())
        }
        Expr::Value(v) => eval_value(&v.value),
        _ => Err(SqlError::UnsupportedExpr),
    }
}

fn compare_db_values(a: &DbValue, b: &DbValue) -> Option<Ordering> {
    match (a, b) {
        (DbValue::Integer(x), DbValue::Integer(y)) => Some(x.cmp(y)),
        (DbValue::Float(x), DbValue::Float(y)) => x.partial_cmp(y),
        (DbValue::Integer(x), DbValue::Float(y)) => (*x as f64).partial_cmp(y),
        (DbValue::Float(x), DbValue::Integer(y)) => x.partial_cmp(&(*y as f64)),
        (DbValue::Text(x), DbValue::Text(y)) => Some(x.cmp(y)),
        (DbValue::Bool(x), DbValue::Bool(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

fn eval_expr_bool(expr: &Expr, schema: &Schema, row: &Row) -> Result<bool, SqlError> {
    match expr {
        Expr::BinaryOp { left, op, right } => match op {
            BinaryOperator::And => {
                Ok(eval_expr_bool(left, schema, row)? && eval_expr_bool(right, schema, row)?)
            }
            BinaryOperator::Or => {
                Ok(eval_expr_bool(left, schema, row)? || eval_expr_bool(right, schema, row)?)
            }
            BinaryOperator::Eq => {
                let lv = eval_expr_value(left, schema, row)?;
                let rv = eval_expr_value(right, schema, row)?;
                Ok(lv == rv)
            }
            BinaryOperator::NotEq => {
                let lv = eval_expr_value(left, schema, row)?;
                let rv = eval_expr_value(right, schema, row)?;
                Ok(lv != rv)
            }
            BinaryOperator::Lt => {
                let lv = eval_expr_value(left, schema, row)?;
                let rv = eval_expr_value(right, schema, row)?;
                Ok(compare_db_values(&lv, &rv) == Some(Ordering::Less))
            }
            BinaryOperator::LtEq => {
                let lv = eval_expr_value(left, schema, row)?;
                let rv = eval_expr_value(right, schema, row)?;
                Ok(matches!(
                    compare_db_values(&lv, &rv),
                    Some(Ordering::Less | Ordering::Equal)
                ))
            }
            BinaryOperator::Gt => {
                let lv = eval_expr_value(left, schema, row)?;
                let rv = eval_expr_value(right, schema, row)?;
                Ok(compare_db_values(&lv, &rv) == Some(Ordering::Greater))
            }
            BinaryOperator::GtEq => {
                let lv = eval_expr_value(left, schema, row)?;
                let rv = eval_expr_value(right, schema, row)?;
                Ok(matches!(
                    compare_db_values(&lv, &rv),
                    Some(Ordering::Greater | Ordering::Equal)
                ))
            }
            _ => Err(SqlError::UnsupportedExpr),
        },
        Expr::IsNull(e) => {
            let v = eval_expr_value(e, schema, row)?;
            Ok(v == DbValue::Null)
        }
        Expr::IsNotNull(e) => {
            let v = eval_expr_value(e, schema, row)?;
            Ok(v != DbValue::Null)
        }
        Expr::Nested(e) => eval_expr_bool(e, schema, row),
        _ => Err(SqlError::UnsupportedExpr),
    }
}

/// Try to extract a simple `pk_col = integer` pattern for PkEqual scanner optimisation.
fn try_pk_equal(table: &str, schema: &Schema, expr: &Expr) -> Option<Scanner> {
    let Expr::BinaryOp { left, op: BinaryOperator::Eq, right } = expr else {
        return None;
    };
    let pk_name = &schema.columns[schema.primary_key].name;

    let (ident, val_expr) = match (left.as_ref(), right.as_ref()) {
        (Expr::Identifier(i), v) => (i, v),
        (v, Expr::Identifier(i)) => (i, v),
        _ => return None,
    };

    if !ident.value.eq_ignore_ascii_case(pk_name) {
        return None;
    }

    let Expr::Value(v) = val_expr else {
        return None;
    };
    let Value::Number(n, _) = &v.value else {
        return None;
    };

    let pk: Key = n.parse().ok()?;
    Some(Scanner::PkEqual {
        table: table.to_string(),
        pk,
    })
}

pub fn execute<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    sql: &str,
) -> Result<Vec<Row>, SqlError> {
    use sqlparser::ast::{SetExpr, Statement as SqlStatement, TableFactor};
    use sqlparser::dialect::GenericDialect;
    use sqlparser::parser::Parser;

    let mut stmts =
        Parser::parse_sql(&GenericDialect {}, sql).map_err(|e| SqlError::Parse(e.to_string()))?;

    if stmts.len() != 1 {
        return Err(SqlError::UnsupportedStatement);
    }

    let SqlStatement::Query(query) = stmts.remove(0) else {
        return Err(SqlError::UnsupportedStatement);
    };
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

        let scanner = filter
            .as_ref()
            .and_then(|f| try_pk_equal(&table_name, &schema, f))
            .unwrap_or_else(|| Scanner::PkRange {
                table: table_name.clone(),
                range: (Bound::Unbounded, Bound::Unbounded),
            });

        let stmt = Statement::Select {
            table: table_name,
            scanner,
            projections,
            filter,
        };
        stmt.execute(locked_tx)
    })
}
