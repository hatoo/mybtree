use std::cmp::Ordering;

use sqlparser::ast::{BinaryOperator, Expr, Value};

use crate::{DbValue, Row, Schema};

use super::SqlError;

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

pub(super) fn eval_expr_value(
    expr: &Expr,
    schema: &Schema,
    row: &Row,
) -> Result<DbValue, SqlError> {
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

pub(super) fn eval_expr_bool(expr: &Expr, schema: &Schema, row: &Row) -> Result<bool, SqlError> {
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
