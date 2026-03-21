use std::cmp::Ordering;

use sqlparser::ast::{BinaryOperator, Expr, Value};

use crate::DbValue;
use crate::database::LockedDbTransaction;

use super::SqlError;
use super::table_source::TableSource;

pub(super) fn eval_value(v: &Value) -> Result<DbValue, SqlError> {
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
        Value::SingleQuotedString(s) | Value::DoubleQuotedString(s) => Ok(DbValue::Text(s.clone())),
        Value::Boolean(b) => Ok(DbValue::Bool(*b)),
        Value::Null => Ok(DbValue::Null),
        _ => Err(SqlError::UnsupportedExpr),
    }
}

pub(super) fn eval_value_expr(expr: &Expr) -> Result<DbValue, SqlError> {
    match expr {
        Expr::Value(v) => eval_value(&v.value),
        Expr::UnaryOp {
            op: sqlparser::ast::UnaryOperator::Minus,
            expr,
        } => match eval_value_expr(expr)? {
            DbValue::Integer(i) => Ok(DbValue::Integer(-i)),
            DbValue::Float(f) => Ok(DbValue::Float(-f)),
            _ => Err(SqlError::UnsupportedExpr),
        },
        _ => Err(SqlError::UnsupportedExpr),
    }
}

pub(super) fn eval_expr_value<const N: usize>(
    expr: &Expr,
    src: &TableSource<'_>,
    locked_tx: &mut LockedDbTransaction<'_, N>,
) -> Result<DbValue, SqlError> {
    match expr {
        Expr::Identifier(ident) => src.resolve(&ident.value),
        Expr::CompoundIdentifier(parts) => {
            if parts.len() == 2 {
                src.resolve_qualified(&parts[0].value, &parts[1].value)
            } else {
                let name = parts.last().map(|i| i.value.as_str()).unwrap_or("");
                src.resolve(name)
            }
        }
        Expr::Value(v) => eval_value(&v.value),
        Expr::Subquery(subquery) => {
            let mut val = DbValue::Null;
            super::query::scan_query_locked::<N, _>(
                locked_tx,
                *subquery.clone(),
                Some(src),
                |_tx, row| {
                    if let Some((_, v)) = row.first() {
                        val = v.clone();
                    }
                    Ok(true) // stop after first row
                },
            )?;
            Ok(val)
        }
        _ => Err(SqlError::UnsupportedExpr),
    }
}

pub(super) fn compare_db_values(a: &DbValue, b: &DbValue) -> Option<Ordering> {
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

fn compare_with_op(lv: &DbValue, rv: &DbValue, op: &BinaryOperator) -> Result<bool, SqlError> {
    match op {
        BinaryOperator::Eq => Ok(lv == rv),
        BinaryOperator::NotEq => Ok(lv != rv),
        BinaryOperator::Lt => Ok(compare_db_values(lv, rv) == Some(Ordering::Less)),
        BinaryOperator::LtEq => Ok(matches!(
            compare_db_values(lv, rv),
            Some(Ordering::Less | Ordering::Equal)
        )),
        BinaryOperator::Gt => Ok(compare_db_values(lv, rv) == Some(Ordering::Greater)),
        BinaryOperator::GtEq => Ok(matches!(
            compare_db_values(lv, rv),
            Some(Ordering::Greater | Ordering::Equal)
        )),
        _ => Err(SqlError::UnsupportedExpr),
    }
}

pub(super) fn eval_expr_bool<const N: usize>(
    expr: &Expr,
    src: &TableSource<'_>,
    locked_tx: &mut LockedDbTransaction<'_, N>,
) -> Result<bool, SqlError> {
    match expr {
        Expr::BinaryOp { left, op, right } => match op {
            BinaryOperator::And => {
                Ok(eval_expr_bool(left, src, locked_tx)? && eval_expr_bool(right, src, locked_tx)?)
            }
            BinaryOperator::Or => {
                Ok(eval_expr_bool(left, src, locked_tx)? || eval_expr_bool(right, src, locked_tx)?)
            }
            _ => {
                let lv = eval_expr_value(left, src, locked_tx)?;
                let rv = eval_expr_value(right, src, locked_tx)?;
                compare_with_op(&lv, &rv, op)
            }
        },
        Expr::Exists { subquery, negated } => {
            let mut found = false;
            super::query::scan_query_locked::<N, _>(
                locked_tx,
                *subquery.clone(),
                Some(src),
                |_tx, _row| {
                    found = true;
                    Ok(true)
                },
            )?;
            Ok(found ^ negated)
        }
        Expr::InSubquery {
            expr,
            subquery,
            negated,
        } => {
            let lv = eval_expr_value(expr, src, locked_tx)?;
            let mut found = false;
            super::query::scan_query_locked::<N, _>(
                locked_tx,
                *subquery.clone(),
                Some(src),
                |_tx, row| {
                    if row.first().is_some_and(|(_, v)| *v == lv) {
                        found = true;
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                },
            )?;
            Ok(found ^ negated)
        }
        Expr::AnyOp {
            left,
            compare_op,
            right,
            ..
        } => {
            let lv = eval_expr_value(left, src, locked_tx)?;
            let Expr::Subquery(subquery) = right.as_ref() else {
                return Err(SqlError::UnsupportedExpr);
            };
            let mut matched = false;
            super::query::scan_query_locked::<N, _>(
                locked_tx,
                *subquery.clone(),
                Some(src),
                |_tx, row| {
                    if let Some((_, rv)) = row.first() {
                        if compare_with_op(&lv, rv, compare_op)? {
                            matched = true;
                            return Ok(true);
                        }
                    }
                    Ok(false)
                },
            )?;
            Ok(matched)
        }
        Expr::AllOp {
            left,
            compare_op,
            right,
        } => {
            let lv = eval_expr_value(left, src, locked_tx)?;
            let Expr::Subquery(subquery) = right.as_ref() else {
                return Err(SqlError::UnsupportedExpr);
            };
            let mut all_match = true;
            super::query::scan_query_locked::<N, _>(
                locked_tx,
                *subquery.clone(),
                Some(src),
                |_tx, row| {
                    if let Some((_, rv)) = row.first() {
                        if !compare_with_op(&lv, rv, compare_op)? {
                            all_match = false;
                            return Ok(true);
                        }
                    }
                    Ok(false)
                },
            )?;
            Ok(all_match)
        }
        Expr::IsNull(e) => {
            let v = eval_expr_value(e, src, locked_tx)?;
            Ok(v == DbValue::Null)
        }
        Expr::IsNotNull(e) => {
            let v = eval_expr_value(e, src, locked_tx)?;
            Ok(v != DbValue::Null)
        }
        Expr::Nested(e) => eval_expr_bool(e, src, locked_tx),
        _ => Err(SqlError::UnsupportedExpr),
    }
}
