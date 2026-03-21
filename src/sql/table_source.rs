use crate::{DbValue, Row, Schema};

use super::SqlError;

/// A named row source used during query evaluation.
/// Wraps either a physical table row (Schema + Row) or a subquery result row.
pub(super) struct TableSource<'a> {
    /// The qualifier name: table name or alias (e.g. "u" in `FROM users AS u`).
    pub qualifier: &'a str,
    columns: Columns<'a>,
}

enum Columns<'a> {
    /// Physical table: column definitions + positional values.
    Physical {
        schema: &'a Schema,
        row: &'a Row,
    },
    /// Subquery or derived table: named pairs.
    Named(&'a [(String, DbValue)]),
}

impl<'a> TableSource<'a> {
    /// Create from a physical table schema and row.
    pub fn from_table(qualifier: &'a str, schema: &'a Schema, row: &'a Row) -> Self {
        Self {
            qualifier,
            columns: Columns::Physical { schema, row },
        }
    }

    /// Create from a subquery result row (named pairs).
    pub fn from_result_row(qualifier: &'a str, row: &'a [(String, DbValue)]) -> Self {
        Self {
            qualifier,
            columns: Columns::Named(row),
        }
    }

    /// Resolve an unqualified column name to its value.
    pub fn resolve(&self, col_name: &str) -> Result<DbValue, SqlError> {
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

    /// Resolve a qualified column reference (e.g. `u.name`).
    /// Returns Err if the qualifier doesn't match, or the column isn't found.
    pub fn resolve_qualified(&self, qualifier: &str, col_name: &str) -> Result<DbValue, SqlError> {
        if !self.qualifier.eq_ignore_ascii_case(qualifier) {
            return Err(SqlError::ColumnNotFound(format!("{qualifier}.{col_name}")));
        }
        self.resolve(col_name)
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

    /// Get the schema (only available for physical tables).
    pub fn schema(&self) -> Option<&Schema> {
        match &self.columns {
            Columns::Physical { schema, .. } => Some(schema),
            Columns::Named(_) => None,
        }
    }
}
