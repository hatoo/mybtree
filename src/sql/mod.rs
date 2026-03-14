use std::{borrow::Cow, ops::Bound};

use rkyv::rancor::Error;
use sqlparser::ast::SelectItem;

use crate::{
    DatabaseError, DbTransaction, Key, Row,
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
        locked_tx: LockedDbTransaction<'a, N>,
    ) -> Result<Vec<Row>, SqlError> {
        match self {
            Statement::Select {
                table,
                scanner,
                projections,
            } => {
                let mut rows = Vec::new();
                scanner.scan::<_, SqlError, N>(locked_tx, |_tx, key, row| todo!())?;
                Ok(rows)
            }
        }
    }
}

pub fn execute<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    sql: &str,
) -> Result<Vec<Row>, SqlError> {
    todo!()
}
