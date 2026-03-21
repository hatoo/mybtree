use crate::{Database, DbTransaction, DbValue};
use sqlparser::ast::Statement as SqlStatement;

use execute::{
    execute_create_index, execute_create_table, execute_delete, execute_insert, execute_update,
};
use query::execute_query;

mod execute;
pub mod expr;
mod query;
mod scan;
pub(crate) mod table_source;
#[cfg(test)]
mod tests;

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
