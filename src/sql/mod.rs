use std::{borrow::Cow, ops::Bound};

use rkyv::rancor::Error;
use sqlparser::ast::{Expr, SelectItem};

use crate::{
    DatabaseError, DbTransaction, Key, Row, Schema,
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
    },
}

#[derive(thiserror::Error, Debug)]
pub enum SqlError {
    #[error("Database error: {0}")]
    Database(#[from] crate::DatabaseError),
    #[error("column not found: {0}")]
    ColumnNotFound(String),
    #[error("unsupported select item")]
    UnsupportedSelectItem,
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
            } => {
                let schema = locked_tx.get_schema(table)?;
                let mut rows = Vec::new();
                scanner.scan::<_, SqlError, N>(locked_tx, |_tx, _key, archived| {
                    let row: Row = rkyv::deserialize::<Row, Error>(archived)
                        .map_err(DatabaseError::Internal)?;
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
            _ => return Err(SqlError::UnsupportedSelectItem),
        }
    }
    Ok(Row { values })
}

pub fn execute<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    sql: &str,
) -> Result<Vec<Row>, SqlError> {
    todo!()
}
