use std::{borrow::Cow, ops::Bound};

use rkyv::rancor::Error;
use sqlparser::ast::{Expr, Value};

use crate::{
    DatabaseError, DbValue, Key, Row, Schema,
    database::{ArchivedRow, LockedDbTransaction, db_value_to_bytes},
};

use super::expr::eval_value;

pub(super) enum Scanner {
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

impl Scanner {
    pub fn full(table: String) -> Self {
        Scanner::PkRange {
            table: table,
            range: (Bound::Unbounded, Bound::Unbounded),
        }
    }

    pub fn from_filter(
        table: &str,
        schema: &Schema,
        indexed_columns: &[String],
        filter: &Option<Expr>,
    ) -> Self {
        filter
            .as_ref()
            .and_then(|f| find_best_scanner(table, schema, indexed_columns, f))
            .unwrap_or_else(|| Scanner::full(table.to_string()))
    }

    pub(super) fn scan<'a, F, E: From<DatabaseError>, const N: usize>(
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

/// Get the column name from an Identifier or CompoundIdentifier (last part).
fn column_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Identifier(i) => Some(&i.value),
        Expr::CompoundIdentifier(parts) => parts.last().map(|i| i.value.as_str()),
        _ => None,
    }
}

/// Extract a `col op literal` or `literal op col` pair from a binary expression.
/// Returns `(column_name, operator_from_column_perspective, literal_value)`.
fn extract_col_op_value<'a>(
    left: &'a Expr,
    op: &sqlparser::ast::BinaryOperator,
    right: &'a Expr,
) -> Option<(&'a str, sqlparser::ast::BinaryOperator, &'a Value)> {
    use sqlparser::ast::BinaryOperator;

    let (col_name, val_expr, effective_op) = match (column_name(left), column_name(right)) {
        (Some(name), None) => {
            let Expr::Value(v) = right else { return None };
            (name, v, op.clone())
        }
        (None, Some(name)) => {
            let Expr::Value(v) = left else { return None };
            let flipped = match op {
                BinaryOperator::Eq => BinaryOperator::Eq,
                BinaryOperator::Lt => BinaryOperator::Gt,
                BinaryOperator::LtEq => BinaryOperator::GtEq,
                BinaryOperator::Gt => BinaryOperator::Lt,
                BinaryOperator::GtEq => BinaryOperator::LtEq,
                _ => return None,
            };
            (name, v, flipped)
        }
        _ => return None,
    };

    Some((col_name, effective_op, &val_expr.value))
}

/// Try to parse a literal value as a primary-key (`Key = i64`).
fn value_as_key(v: &Value) -> Option<Key> {
    match v {
        Value::Number(n, _) => n.parse().ok(),
        _ => None,
    }
}

/// Try to convert a SQL literal into index bytes via `DbValue`.
fn value_as_index_bytes(v: &Value) -> Option<Vec<u8>> {
    let db_val = eval_value(v).ok()?;
    if matches!(db_val, DbValue::Null) {
        return None;
    }
    Some(db_value_to_bytes(&db_val))
}

/// Analyse a WHERE expression and choose the most efficient `Scanner`.
///
/// Priority:
/// 1. Primary-key equality  (`pk = literal`)   → `PkEqual`
/// 2. Primary-key range     (`pk > literal`, …) → `PkRange`
/// 3. Secondary-index equality (`indexed_col = literal`) → `IndexEqual`
/// 4. Secondary-index range (`indexed_col > literal`, …) → `IndexRange`
/// 5. Full table scan                                     → `PkRange { Unbounded, Unbounded }`
///
/// For AND expressions the function recurses into both sides and merges
/// the tightest bounds it can find for the same column.
pub(super) fn find_best_scanner(
    table: &str,
    schema: &Schema,
    indexed_columns: &[String],
    expr: &Expr,
) -> Option<Scanner> {
    use sqlparser::ast::BinaryOperator;

    match expr {
        // ── AND: merge constraints from both sides ──────────────────
        Expr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            let l = find_best_scanner(table, schema, indexed_columns, left);
            let r = find_best_scanner(table, schema, indexed_columns, right);
            // Pick the more specific scanner (PkEqual > PkRange > IndexEqual > IndexRange).
            fn rank(s: &Scanner) -> u8 {
                match s {
                    Scanner::PkEqual { .. } => 4,
                    Scanner::PkRange { .. } => 3,
                    Scanner::IndexEqual { .. } => 2,
                    Scanner::IndexRange { .. } => 1,
                }
            }
            match (l, r) {
                (Some(a), Some(b)) => {
                    // Try to merge two PkRange scanners on the same table.
                    if let (
                        Scanner::PkRange {
                            range: (lo_a, hi_a),
                            ..
                        },
                        Scanner::PkRange {
                            range: (lo_b, hi_b),
                            ..
                        },
                    ) = (&a, &b)
                    {
                        let lo = tighter_lower_bound_key(lo_a, lo_b);
                        let hi = tighter_upper_bound_key(hi_a, hi_b);
                        return Some(Scanner::PkRange {
                            table: table.to_string(),
                            range: (lo, hi),
                        });
                    }
                    // Try to merge two IndexRange scanners on the same index.
                    if let (
                        Scanner::IndexRange {
                            index: idx_a,
                            range: (lo_a, hi_a),
                            ..
                        },
                        Scanner::IndexRange {
                            index: idx_b,
                            range: (lo_b, hi_b),
                            ..
                        },
                    ) = (&a, &b)
                    {
                        if idx_a == idx_b {
                            let lo = tighter_lower_bound_bytes(lo_a, lo_b);
                            let hi = tighter_upper_bound_bytes(hi_a, hi_b);
                            return Some(Scanner::IndexRange {
                                table: table.to_string(),
                                index: idx_a.clone(),
                                range: (lo, hi),
                            });
                        }
                    }
                    // Otherwise pick the higher-priority one.
                    if rank(&a) >= rank(&b) {
                        Some(a)
                    } else {
                        Some(b)
                    }
                }
                (Some(s), None) | (None, Some(s)) => Some(s),
                (None, None) => None,
            }
        }

        // ── Nested parenthesised expression ─────────────────────────
        Expr::Nested(inner) => find_best_scanner(table, schema, indexed_columns, inner),

        // ── Single comparison ───────────────────────────────────────
        Expr::BinaryOp { left, op, right } => {
            let (col_name, effective_op, lit) = extract_col_op_value(left, op, right)?;

            let pk_name = &schema.columns[schema.primary_key].name;

            // --- Primary-key optimisation ---
            if col_name.eq_ignore_ascii_case(pk_name) {
                let key = value_as_key(lit)?;
                return match effective_op {
                    BinaryOperator::Eq => Some(Scanner::PkEqual {
                        table: table.to_string(),
                        pk: key,
                    }),
                    BinaryOperator::Gt => Some(Scanner::PkRange {
                        table: table.to_string(),
                        range: (Bound::Excluded(key), Bound::Unbounded),
                    }),
                    BinaryOperator::GtEq => Some(Scanner::PkRange {
                        table: table.to_string(),
                        range: (Bound::Included(key), Bound::Unbounded),
                    }),
                    BinaryOperator::Lt => Some(Scanner::PkRange {
                        table: table.to_string(),
                        range: (Bound::Unbounded, Bound::Excluded(key)),
                    }),
                    BinaryOperator::LtEq => Some(Scanner::PkRange {
                        table: table.to_string(),
                        range: (Bound::Unbounded, Bound::Included(key)),
                    }),
                    _ => None,
                };
            }

            // --- Secondary-index optimisation ---
            if indexed_columns
                .iter()
                .any(|c| c.eq_ignore_ascii_case(col_name))
            {
                let bytes = value_as_index_bytes(lit)?;
                return match effective_op {
                    BinaryOperator::Eq => Some(Scanner::IndexEqual {
                        table: table.to_string(),
                        index: col_name.to_string(),
                        value: bytes,
                    }),
                    BinaryOperator::Gt => Some(Scanner::IndexRange {
                        table: table.to_string(),
                        index: col_name.to_string(),
                        range: (Bound::Excluded(bytes), Bound::Unbounded),
                    }),
                    BinaryOperator::GtEq => Some(Scanner::IndexRange {
                        table: table.to_string(),
                        index: col_name.to_string(),
                        range: (Bound::Included(bytes), Bound::Unbounded),
                    }),
                    BinaryOperator::Lt => Some(Scanner::IndexRange {
                        table: table.to_string(),
                        index: col_name.to_string(),
                        range: (Bound::Unbounded, Bound::Excluded(bytes)),
                    }),
                    BinaryOperator::LtEq => Some(Scanner::IndexRange {
                        table: table.to_string(),
                        index: col_name.to_string(),
                        range: (Bound::Unbounded, Bound::Included(bytes)),
                    }),
                    _ => None,
                };
            }

            None
        }

        _ => None,
    }
}

// ── Bound-merging helpers ───────────────────────────────────────────

fn tighter_lower_bound_key(a: &Bound<Key>, b: &Bound<Key>) -> Bound<Key> {
    match (a, b) {
        (Bound::Unbounded, other) | (other, Bound::Unbounded) => other.clone(),
        (Bound::Included(x), Bound::Included(y)) => Bound::Included(*x.max(y)),
        (Bound::Excluded(x), Bound::Excluded(y)) => Bound::Excluded(*x.max(y)),
        (Bound::Included(x), Bound::Excluded(y)) | (Bound::Excluded(y), Bound::Included(x)) => {
            if y >= x {
                Bound::Excluded(*y)
            } else {
                Bound::Included(*x)
            }
        }
    }
}

fn tighter_upper_bound_key(a: &Bound<Key>, b: &Bound<Key>) -> Bound<Key> {
    match (a, b) {
        (Bound::Unbounded, other) | (other, Bound::Unbounded) => other.clone(),
        (Bound::Included(x), Bound::Included(y)) => Bound::Included(*x.min(y)),
        (Bound::Excluded(x), Bound::Excluded(y)) => Bound::Excluded(*x.min(y)),
        (Bound::Included(x), Bound::Excluded(y)) | (Bound::Excluded(y), Bound::Included(x)) => {
            if y <= x {
                Bound::Excluded(*y)
            } else {
                Bound::Included(*x)
            }
        }
    }
}

fn tighter_lower_bound_bytes(a: &Bound<Vec<u8>>, b: &Bound<Vec<u8>>) -> Bound<Vec<u8>> {
    match (a, b) {
        (Bound::Unbounded, other) | (other, Bound::Unbounded) => other.clone(),
        (Bound::Included(x), Bound::Included(y)) => Bound::Included(x.max(y).clone()),
        (Bound::Excluded(x), Bound::Excluded(y)) => Bound::Excluded(x.max(y).clone()),
        (Bound::Included(x), Bound::Excluded(y)) | (Bound::Excluded(y), Bound::Included(x)) => {
            if y >= x {
                Bound::Excluded(y.clone())
            } else {
                Bound::Included(x.clone())
            }
        }
    }
}

fn tighter_upper_bound_bytes(a: &Bound<Vec<u8>>, b: &Bound<Vec<u8>>) -> Bound<Vec<u8>> {
    match (a, b) {
        (Bound::Unbounded, other) | (other, Bound::Unbounded) => other.clone(),
        (Bound::Included(x), Bound::Included(y)) => Bound::Included(x.min(y).clone()),
        (Bound::Excluded(x), Bound::Excluded(y)) => Bound::Excluded(x.min(y).clone()),
        (Bound::Included(x), Bound::Excluded(y)) | (Bound::Excluded(y), Bound::Included(x)) => {
            if y <= x {
                Bound::Excluded(y.clone())
            } else {
                Bound::Included(x.clone())
            }
        }
    }
}
