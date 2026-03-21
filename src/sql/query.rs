use rkyv::rancor::Error;
use sqlparser::ast::{Expr, JoinConstraint, JoinOperator, SelectItem, TableFactor};

use crate::database::LockedDbTransaction;
use crate::{DatabaseError, DbTransaction, DbValue, Row};

use super::expr::eval_expr_bool;
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
    scan_query_locked::<N, _>(locked_tx, query, outer_src, |_tx, row| {
        rows.push(row.to_vec());
        Ok(false)
    })?;
    Ok(ResultSet { rows })
}

/// Stream query results row by row through a callback.
/// The callback receives `&mut LockedDbTransaction` and the projected row.
/// Returns `Ok(true)` to stop early, `Ok(false)` to continue.
pub(super) fn scan_query_locked<const N: usize, F>(
    locked_tx: &mut LockedDbTransaction<'_, N>,
    query: sqlparser::ast::Query,
    outer_src: Option<&TableSource<'_>>,
    mut f: F,
) -> Result<(), SqlError>
where
    F: FnMut(&mut LockedDbTransaction<'_, N>, &[(String, DbValue)]) -> Result<bool, SqlError>,
{
    use sqlparser::ast::SetExpr;
    let SetExpr::Select(select) = *query.body else {
        return Err(SqlError::UnsupportedStatement);
    };

    if select.from.len() != 1 {
        return Err(SqlError::UnsupportedStatement);
    }

    let projections = select.projection;
    let filter = select.selection;
    let from = &select.from[0];

    // Extract join ON conditions and right-side table factors.
    let mut joins: Vec<(&TableFactor, Option<&Expr>)> = Vec::new();
    for join in &from.joins {
        let on_expr = match &join.join_operator {
            JoinOperator::Inner(JoinConstraint::On(expr))
            | JoinOperator::Join(JoinConstraint::On(expr)) => Some(expr),
            JoinOperator::Inner(JoinConstraint::None)
            | JoinOperator::Join(JoinConstraint::None)
            | JoinOperator::CrossJoin(_) => None,
            _ => return Err(SqlError::UnsupportedStatement),
        };
        joins.push((&join.relation, on_expr));
    }

    // Scan the base relation, then for each row, nest into joins.
    scan_relation::<N>(
        locked_tx,
        &from.relation,
        outer_src,
        &filter,
        |tx, base_src| {
            if joins.is_empty() {
                // No joins — apply WHERE and project.
                if let Some(expr) = &filter {
                    if !eval_expr_bool(expr, &base_src, tx)? {
                        return Ok(false);
                    }
                }
                let projected = project_row(&base_src, &projections)?;
                return f(tx, &projected);
            }

            // Process joins by nesting.
            scan_joins::<N, _>(tx, &joins, 0, &base_src, &filter, &projections, &mut f)
        },
    )
}

/// Recursively process joins. For each join level, scan the right table
/// and chain it as a child of the accumulated source.
fn scan_joins<const N: usize, F>(
    locked_tx: &mut LockedDbTransaction<'_, N>,
    joins: &[(&TableFactor, Option<&Expr>)],
    idx: usize,
    left_src: &TableSource<'_>,
    filter: &Option<Expr>,
    projections: &[SelectItem],
    f: &mut F,
) -> Result<bool, SqlError>
where
    F: FnMut(&mut LockedDbTransaction<'_, N>, &[(String, DbValue)]) -> Result<bool, SqlError>,
{
    let (relation, on_expr) = &joins[idx];
    let last = idx + 1 == joins.len();

    let mut stopped = false;
    let on_expr = *on_expr;

    scan_relation::<N>(locked_tx, relation, Some(left_src), &None, |tx, right_src| {
        // Chain right on top of left for column resolution.
        let joined_src = right_src.with_parent(left_src);

        // Evaluate ON condition.
        if let Some(on) = on_expr {
            if !eval_expr_bool(on, &joined_src, tx)? {
                return Ok(false);
            }
        }

        let result = if last {
            // Last join — apply WHERE and project.
            if let Some(expr) = filter {
                if !eval_expr_bool(expr, &joined_src, tx)? {
                    return Ok(false);
                }
            }
            let projected = project_row(&joined_src, projections)?;
            f(tx, &projected)?
        } else {
            // More joins to process.
            scan_joins::<N, _>(tx, joins, idx + 1, &joined_src, filter, projections, f)?
        };
        if result {
            stopped = true;
        }
        Ok(result)
    })?;
    Ok(stopped)
}

/// Scan a single table factor (physical table or derived subquery),
/// calling `f` for each row with a `TableSource` that has `parent` set.
fn scan_relation<'a, const N: usize>(
    locked_tx: &mut LockedDbTransaction<'_, N>,
    relation: &TableFactor,
    parent: Option<&'a TableSource<'a>>,
    scanner_filter: &Option<Expr>,
    mut f: impl FnMut(&mut LockedDbTransaction<'_, N>, TableSource<'_>) -> Result<bool, SqlError>,
) -> Result<(), SqlError> {
    match relation {
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
                Scanner::from_filter(&table_name, &schema, &indexed_columns, scanner_filter);

            scanner.scan::<_, SqlError, N>(locked_tx, |tx, _key, archived| {
                let row: Row =
                    rkyv::deserialize::<Row, Error>(archived).map_err(DatabaseError::Internal)?;
                let mut src = TableSource::from_table(&qualifier, &schema, &row);
                if let Some(p) = parent {
                    src.set_parent(p);
                }
                f(tx, src)
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

            let inner: &mut dyn FnMut(
                &mut LockedDbTransaction<'_, N>,
                &[(String, DbValue)],
            ) -> Result<bool, SqlError> = &mut |tx, sub_row| {
                let mut src = TableSource::from_result_row(&alias, sub_row);
                if let Some(p) = parent {
                    src.set_parent(p);
                }
                f(tx, src)
            };
            scan_query_locked::<N, _>(locked_tx, *subquery.clone(), parent, inner)?;
            Ok(())
        }
        _ => Err(SqlError::UnsupportedStatement),
    }
}
