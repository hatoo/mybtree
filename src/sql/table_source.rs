use crate::{DbValue, Row, Schema};

use super::SqlError;

/// A named row source used during query evaluation.
/// Wraps either a physical table row (Schema + Row) or a subquery result row.
/// Optionally chains to a parent scope for correlated subquery column resolution.
pub(crate) struct TableSource<'a> {
    /// The qualifier name: table name or alias (e.g. "u" in `FROM users AS u`).
    pub qualifier: &'a str,
    columns: Columns<'a>,
    /// Parent scope for correlated subqueries — column lookup falls back here.
    parent: Option<&'a TableSource<'a>>,
}

enum Columns<'a> {
    /// Physical table: column definitions + positional values.
    Physical { schema: &'a Schema, row: &'a Row },
    /// Subquery or derived table: named pairs.
    Named(&'a [(String, DbValue)]),
}

impl<'a> TableSource<'a> {
    /// Create from a physical table schema and row.
    pub fn from_table(qualifier: &'a str, schema: &'a Schema, row: &'a Row) -> Self {
        Self {
            qualifier,
            columns: Columns::Physical { schema, row },
            parent: None,
        }
    }

    /// Create from a subquery result row (named pairs).
    pub fn from_result_row(qualifier: &'a str, row: &'a [(String, DbValue)]) -> Self {
        Self {
            qualifier,
            columns: Columns::Named(row),
            parent: None,
        }
    }

    /// Return a new TableSource with a parent scope for correlated subquery resolution.
    pub fn with_parent(mut self, parent: &'a TableSource<'a>) -> Self {
        self.parent = Some(parent);
        self
    }

    /// Resolve an unqualified column name to its value.
    /// Falls back to parent scope if not found locally.
    pub fn resolve(&self, col_name: &str) -> Result<DbValue, SqlError> {
        match self.resolve_local(col_name) {
            Ok(val) => Ok(val),
            Err(_) if self.parent.is_some() => self.parent.unwrap().resolve(col_name),
            Err(e) => Err(e),
        }
    }

    /// Resolve a qualified column reference (e.g. `u.name`).
    /// Falls back to parent scope if qualifier doesn't match locally.
    pub fn resolve_qualified(&self, qualifier: &str, col_name: &str) -> Result<DbValue, SqlError> {
        if self.qualifier.eq_ignore_ascii_case(qualifier) {
            return self.resolve_local(col_name);
        }
        if let Some(parent) = self.parent {
            return parent.resolve_qualified(qualifier, col_name);
        }
        Err(SqlError::ColumnNotFound(format!("{qualifier}.{col_name}")))
    }

    /// Resolve locally without falling back to parent.
    fn resolve_local(&self, col_name: &str) -> Result<DbValue, SqlError> {
        match &self.columns {
            Columns::Physical { schema, row } => {
                let idx = schema
                    .columns
                    .iter()
                    .position(|c| c.name.eq_ignore_ascii_case(col_name))
                    .ok_or_else(|| SqlError::ColumnNotFound(col_name.to_string()))?;
                Ok(row.values[idx].clone())
            }
            Columns::Named(pairs) => {
                let (_, val) = pairs
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(col_name))
                    .ok_or_else(|| SqlError::ColumnNotFound(col_name.to_string()))?;
                Ok(val.clone())
            }
        }
    }

    /// List all visible columns as (name, value) pairs, respecting implicit PK hiding.
    pub fn all_columns(&self) -> Vec<(String, DbValue)> {
        match &self.columns {
            Columns::Physical { schema, row } => schema
                .columns
                .iter()
                .enumerate()
                .filter(|(i, _)| !(schema.implicit_pk && *i == schema.primary_key))
                .map(|(i, col)| (col.name.clone(), row.values[i].clone()))
                .collect(),
            Columns::Named(pairs) => pairs.to_vec(),
        }
    }
}
