use std::ops::Bound;

use rkyv::rancor::Error;
use sqlparser::ast::SelectItem;

use crate::{DatabaseError, DbTransaction, Key, Row, database::LockedDbTransaction};

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
        locked_tx: &mut LockedDbTransaction<'a, N>,
        mut f: F,
    ) -> Result<(), E>
    where
        F: for<'local> FnMut(&mut LockedDbTransaction<'local, N>, Key, Row) -> Result<bool, E>,
    {
        match self {
            Scanner::PkEqual { table, pk } => {
                if let Some(row) = locked_tx.get(table, *pk)? {
                    f(locked_tx, *pk, row)?;
                }
            }
            Scanner::PkRange { table, range } => {
                locked_tx.scan(table.as_str(), *range, |mut tx, key, value| {
                    let value = rkyv::deserialize::<Row, Error>(value)
                        .map_err(|e| DatabaseError::Internal(e))?;
                    f(&mut tx, key, value)
                })?;
            }
            _ => unimplemented!(),
        }
        Ok(())
    }
}

pub fn execute<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    sql: &str,
) -> Result<Vec<Row>, SqlError> {
    todo!()
}
