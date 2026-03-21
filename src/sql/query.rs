use rkyv::rancor::Error;
use sqlparser::ast::{Expr, SelectItem};

use crate::database::LockedDbTransaction;
use crate::{DatabaseError, DbTransaction, DbValue, Row};

use super::expr::{eval_expr_bool, eval_value_expr};
use super::scan::Scanner;
use super::table_source::TableSource;
use super::{ResultSet, SqlError};

pub(super) fn project_row(
    src: &TableSource<'_>,
    projections: &[SelectItem],
) -> Result<Vec<(String, DbValue)>, SqlError> {
    let mut values = Vec::new();
    for item in projections {
        match item {
            SelectItem::Wildcard(_) => {
                values.extend(src.all_columns());
            }
            SelectItem::UnnamedExpr(Expr::Identifier(ident)) => {
                let val = src.resolve(&ident.value)?;
                values.push((ident.value.clone(), val));
            }
            SelectItem::UnnamedExpr(Expr::CompoundIdentifier(parts)) if parts.len() == 2 => {
                let val = src.resolve_qualified(&parts[0].value, &parts[1].value)?;
                values.push((parts[1].value.clone(), val));
            }
            SelectItem::ExprWithAlias {
                expr: Expr::Identifier(ident),
                alias,
            } => {
                let val = src.resolve(&ident.value)?;
                values.push((alias.value.clone(), val));
            }
            SelectItem::ExprWithAlias {
                expr: Expr::CompoundIdentifier(parts),
                alias,
            } if parts.len() == 2 => {
                let val = src.resolve_qualified(&parts[0].value, &parts[1].value)?;
                values.push((alias.value.clone(), val));
            }
            _ => return Err(SqlError::UnsupportedExpr),
        }
    }
    Ok(values)
}

pub(super) fn execute_query<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    query: sqlparser::ast::Query,
) -> Result<ResultSet, SqlError> {
    tx.with_lock(|mut locked_tx| execute_query_locked(&mut locked_tx, query, None))
}

pub(super) fn execute_query_locked<const N: usize>(
    locked_tx: &mut LockedDbTransaction<'_, N>,
    query: sqlparser::ast::Query,
    outer_src: Option<&TableSource<'_>>,
) -> Result<ResultSet, SqlError> {
    let mut rows = Vec::new();
    scan_query_locked::<N>(locked_tx, query, outer_src, &mut |_tx, row| {
        rows.push(row.to_vec());
        Ok(false)
    })?;
    Ok(ResultSet { rows })
}

/// Stream query results row by row through a callback.
/// The callback receives `&mut LockedDbTransaction` and the projected row.
/// Returns `Ok(true)` to stop early, `Ok(false)` to continue.
pub(super) fn scan_query_locked<const N: usize>(
    locked_tx: &mut LockedDbTransaction<'_, N>,
    query: sqlparser::ast::Query,
    outer_src: Option<&TableSource<'_>>,
    f: &mut dyn FnMut(&mut LockedDbTransaction<'_, N>, &[(String, DbValue)]) -> Result<bool, SqlError>,
) -> Result<(), SqlError> {
    use sqlparser::ast::{SetExpr, TableFactor};
    let SetExpr::Select(select) = *query.body else {
        return Err(SqlError::UnsupportedStatement);
    };

    if select.from.len() != 1 || !select.from[0].joins.is_empty() {
        return Err(SqlError::UnsupportedStatement);
    }

    let projections = select.projection;
    let filter = select.selection;

    match &select.from[0].relation {
        TableFactor::Table { name, alias, .. } => {
            let table_name = name
                .0
                .last()
                .and_then(|p| p.as_ident())
                .map(|i| i.value.clone())
                .ok_or(SqlError::UnsupportedStatement)?;
            let qualifier = alias
                .as_ref()
                .map(|a| a.name.value.clone())
                .unwrap_or_else(|| table_name.clone());

            let schema = locked_tx.get_schema(&table_name)?;
            let indexed_columns = locked_tx.get_indexed_columns(&table_name)?;

            let scanner =
                Scanner::from_filter(&table_name, &schema, &indexed_columns, &filter);

            scanner.scan::<_, SqlError, N>(locked_tx, |tx, _key, archived| {
                let row: Row = rkyv::deserialize::<Row, Error>(archived)
                    .map_err(DatabaseError::Internal)?;
                let src = TableSource::from_table(&qualifier, &schema, &row);
                let src = match outer_src {
                    Some(outer) => src.with_parent(outer),
                    None => src,
                };
                if let Some(expr) = &filter {
                    if !eval_expr_bool(expr, &src, tx)? {
                        return Ok(false);
                    }
                }
                let projected = project_row(&src, &projections)?;
                f(tx, &projected)
            })?;
            Ok(())
        }
        TableFactor::Derived {
            subquery, alias, ..
        } => {
            let alias = alias
                .as_ref()
                .map(|a| a.name.value.clone())
                .ok_or(SqlError::UnsupportedStatement)?;

            scan_query_locked::<N>(locked_tx, *subquery.clone(), outer_src, &mut |tx, sub_row| {
                let src = TableSource::from_result_row(&alias, sub_row);
                let src = match outer_src {
                    Some(outer) => src.with_parent(outer),
                    None => src,
                };
                if let Some(expr) = &filter {
                    if !eval_expr_bool(expr, &src, tx)? {
                        return Ok(false);
                    }
                }
                let projected = project_row(&src, &projections)?;
                f(tx, &projected)
            })?;
            Ok(())
        }
        _ => Err(SqlError::UnsupportedStatement),
    }
}
