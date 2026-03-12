use sqlparser::ast::{
    BinaryOperator, ColumnOption, DataType, Expr, SelectItem, SetExpr, Statement, TableFactor,
    UnaryOperator, Value,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use std::ops::Bound;

use crate::database::{
    Column, ColumnType, DatabaseError, DbTransaction, DbValue, LockedDbTransaction, Row, Schema,
    db_value_to_bytes,
};

#[derive(Debug, thiserror::Error)]
pub enum SqlError {
    #[error("parse error: {0}")]
    Parse(#[from] sqlparser::parser::ParserError),
    #[error("unsupported SQL type: {0}")]
    UnsupportedType(String),
    #[error("unsupported statement")]
    UnsupportedStatement,
    #[error("unsupported expression: {0}")]
    UnsupportedExpression(String),
    #[error("invalid value: {0}")]
    InvalidValue(String),
    #[error("database error: {0}")]
    Database(#[from] DatabaseError),
}

fn map_data_type(data_type: &DataType) -> Result<ColumnType, SqlError> {
    match data_type {
        DataType::Int(_) | DataType::Integer(_) | DataType::BigInt(_) => Ok(ColumnType::Integer),
        DataType::Text | DataType::Varchar(_) | DataType::Char(_) => Ok(ColumnType::Text),
        DataType::Float(_) | DataType::Double(_) | DataType::Real => Ok(ColumnType::Float),
        DataType::Boolean | DataType::Bool => Ok(ColumnType::Bool),
        other => Err(SqlError::UnsupportedType(other.to_string())),
    }
}

fn expr_to_dbvalue(expr: &Expr) -> Result<DbValue, SqlError> {
    match expr {
        Expr::Value(v) => match &v.value {
            Value::Number(s, _) => {
                if s.contains('.') {
                    let f: f64 = s.parse().map_err(|_| SqlError::InvalidValue(s.clone()))?;
                    Ok(DbValue::Float(f))
                } else {
                    let i: i64 = s.parse().map_err(|_| SqlError::InvalidValue(s.clone()))?;
                    Ok(DbValue::Integer(i))
                }
            }
            Value::SingleQuotedString(s) | Value::DoubleQuotedString(s) => {
                Ok(DbValue::Text(s.clone()))
            }
            Value::Boolean(b) => Ok(DbValue::Bool(*b)),
            Value::Null => Ok(DbValue::Null),
            other => Err(SqlError::UnsupportedExpression(other.to_string())),
        },
        Expr::UnaryOp { op, expr } if *op == UnaryOperator::Minus => match expr_to_dbvalue(expr)? {
            DbValue::Integer(i) => Ok(DbValue::Integer(-i)),
            DbValue::Float(f) => Ok(DbValue::Float(-f)),
            other => Err(SqlError::UnsupportedExpression(format!("-{:?}", other))),
        },
        other => Err(SqlError::UnsupportedExpression(other.to_string())),
    }
}

fn resolve_column(name: &str, schema: &Schema) -> Result<usize, SqlError> {
    schema
        .columns
        .iter()
        .position(|c| c.name == name)
        .ok_or_else(|| {
            SqlError::Database(DatabaseError::SchemaMismatch(format!(
                "column '{}' not found",
                name
            )))
        })
}

fn resolve_projection(projection: &[SelectItem], schema: &Schema) -> Result<Vec<usize>, SqlError> {
    let mut indices = Vec::new();
    for item in projection {
        match item {
            SelectItem::Wildcard(_) => {
                indices.extend(0..schema.columns.len());
            }
            SelectItem::UnnamedExpr(Expr::Identifier(ident)) => {
                indices.push(resolve_column(&ident.value, schema)?);
            }
            SelectItem::ExprWithAlias {
                expr: Expr::Identifier(ident),
                ..
            } => {
                indices.push(resolve_column(&ident.value, schema)?);
            }
            other => {
                return Err(SqlError::UnsupportedExpression(other.to_string()));
            }
        }
    }
    Ok(indices)
}

fn eval_expr(expr: &Expr, row: &Row, schema: &Schema) -> Result<DbValue, SqlError> {
    match expr {
        Expr::Identifier(ident) => {
            let idx = resolve_column(&ident.value, schema)?;
            Ok(row.values[idx].clone())
        }
        other => expr_to_dbvalue(other),
    }
}

fn compare_dbvalues(a: &DbValue, b: &DbValue) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (DbValue::Null, _) | (_, DbValue::Null) => None,
        (DbValue::Integer(a), DbValue::Integer(b)) => a.partial_cmp(b),
        (DbValue::Float(a), DbValue::Float(b)) => a.partial_cmp(b),
        (DbValue::Integer(a), DbValue::Float(b)) => (*a as f64).partial_cmp(b),
        (DbValue::Float(a), DbValue::Integer(b)) => a.partial_cmp(&(*b as f64)),
        (DbValue::Text(a), DbValue::Text(b)) => Some(a.cmp(b)),
        (DbValue::Bool(a), DbValue::Bool(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

/// Try to extract a simple `col = literal` or `literal = col` pattern from a WHERE expression.
/// Returns `(column_name, value)` if the expression is a simple equality check.
fn try_extract_eq_condition(expr: &Expr) -> Option<(String, DbValue)> {
    if let Expr::BinaryOp { left, op, right } = expr {
        if *op != BinaryOperator::Eq {
            return None;
        }
        // col = literal
        if let Expr::Identifier(ident) = left.as_ref()
            && let Ok(val) = expr_to_dbvalue(right)
        {
            return Some((ident.value.clone(), val));
        }
        // literal = col
        if let Expr::Identifier(ident) = right.as_ref()
            && let Ok(val) = expr_to_dbvalue(left)
        {
            return Some((ident.value.clone(), val));
        }
    }
    None
}

fn select_pk_eq_range(selection: Option<&Expr>, schema: &Schema) -> (Bound<u64>, Bound<u64>) {
    let Some((column_name, value)) = selection.and_then(try_extract_eq_condition) else {
        return (Bound::Unbounded, Bound::Unbounded);
    };

    let Some(pk_column) = schema.columns.get(schema.primary_key) else {
        return (Bound::Unbounded, Bound::Unbounded);
    };

    if column_name != pk_column.name {
        return (Bound::Unbounded, Bound::Unbounded);
    }

    let DbValue::Integer(key) = value else {
        return (Bound::Unbounded, Bound::Unbounded);
    };
    if key < 0 {
        return (Bound::Included(1), Bound::Excluded(1));
    }

    let key = key as u64;
    (Bound::Included(key), Bound::Included(key))
}

fn eval_where(expr: &Expr, row: &Row, schema: &Schema) -> Result<bool, SqlError> {
    match expr {
        Expr::BinaryOp { left, op, right } => match op {
            BinaryOperator::And => {
                Ok(eval_where(left, row, schema)? && eval_where(right, row, schema)?)
            }
            BinaryOperator::Or => {
                Ok(eval_where(left, row, schema)? || eval_where(right, row, schema)?)
            }
            _ => {
                let lval = eval_expr(left, row, schema)?;
                let rval = eval_expr(right, row, schema)?;
                let ord = compare_dbvalues(&lval, &rval);
                let result = match op {
                    BinaryOperator::Eq => ord == Some(std::cmp::Ordering::Equal),
                    BinaryOperator::NotEq => {
                        ord.is_some() && ord != Some(std::cmp::Ordering::Equal)
                    }
                    BinaryOperator::Lt => ord == Some(std::cmp::Ordering::Less),
                    BinaryOperator::LtEq => matches!(
                        ord,
                        Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
                    ),
                    BinaryOperator::Gt => ord == Some(std::cmp::Ordering::Greater),
                    BinaryOperator::GtEq => matches!(
                        ord,
                        Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
                    ),
                    _ => return Err(SqlError::UnsupportedExpression(format!("{}", op))),
                };
                Ok(result)
            }
        },
        Expr::IsNull(inner) => {
            let val = eval_expr(inner, row, schema)?;
            Ok(val == DbValue::Null)
        }
        Expr::IsNotNull(inner) => {
            let val = eval_expr(inner, row, schema)?;
            Ok(val != DbValue::Null)
        }
        Expr::Nested(inner) => eval_where(inner, row, schema),
        _ => Err(SqlError::UnsupportedExpression(expr.to_string())),
    }
}

fn collect_scan<const N: usize>(
    tx: &mut LockedDbTransaction<'_, N>,
    table_name: &str,
) -> Result<Vec<(crate::types::Key, Row)>, SqlError> {
    let mut rows = Vec::new();
    tx.scan(table_name, .., |_, key, row| {
        rows.push((
            key,
            rkyv::deserialize::<Row, rkyv::rancor::Error>(row).unwrap(),
        ));
        Ok::<_, SqlError>(false)
    })?;
    Ok(rows)
}

fn collect_scan_by_index<const N: usize>(
    tx: &mut LockedDbTransaction<'_, N>,
    table_name: &str,
    column_name: &str,
    value: &[u8],
) -> Result<Vec<(crate::types::Key, Row)>, SqlError> {
    let mut rows = Vec::new();
    tx.scan_by_index(table_name, column_name, value..=value, |_, row, key| {
        rows.push((
            key,
            rkyv::deserialize::<Row, rkyv::rancor::Error>(row).unwrap(),
        ));
        Ok::<_, SqlError>(false)
    })?;
    Ok(rows)
}

fn execute_locked<const N: usize>(
    tx: &mut LockedDbTransaction<'_, N>,
    sql: &str,
) -> Result<Vec<Row>, SqlError> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)?;

    let mut result = Vec::new();

    for stmt in statements {
        match stmt {
            Statement::CreateTable(ct) => {
                let mut columns = Vec::new();
                let mut primary_key = None;
                for (i, col_def) in ct.columns.iter().enumerate() {
                    let column_type = map_data_type(&col_def.data_type)?;
                    let is_pk = col_def
                        .options
                        .iter()
                        .any(|opt| matches!(opt.option, ColumnOption::PrimaryKey { .. }));
                    let nullable = if is_pk {
                        false
                    } else {
                        !col_def
                            .options
                            .iter()
                            .any(|opt| matches!(opt.option, ColumnOption::NotNull))
                    };
                    if is_pk {
                        if primary_key.is_some() {
                            return Err(SqlError::UnsupportedExpression(
                                "multiple PRIMARY KEY columns not supported".into(),
                            ));
                        }
                        primary_key = Some(i);
                    }
                    columns.push(Column {
                        name: col_def.name.value.clone(),
                        column_type,
                        nullable,
                    });
                }
                let schema = Schema {
                    columns,
                    primary_key: primary_key.unwrap_or(0),
                    implicit_pk: primary_key.is_none(),
                };
                tx.create_table(&ct.name.to_string(), schema, primary_key)?;
            }
            Statement::Insert(ins) => {
                let table_name = ins.table.to_string();
                let source = ins.source.as_ref().ok_or(SqlError::UnsupportedStatement)?;
                let rows_exprs = match source.body.as_ref() {
                    SetExpr::Values(values) => &values.rows,
                    _ => return Err(SqlError::UnsupportedStatement),
                };

                let schema = tx.get_schema(&table_name)?;

                let total = schema.columns.len();
                let map = if ins.columns.is_empty() {
                    if schema.implicit_pk {
                        (1..total).collect()
                    } else {
                        (0..total).collect()
                    }
                } else {
                    let mut map: Vec<usize> = Vec::with_capacity(ins.columns.len());
                    for col in &ins.columns {
                        let pos = schema
                            .columns
                            .iter()
                            .position(|c| c.name == col.value)
                            .ok_or_else(|| {
                                DatabaseError::SchemaMismatch(format!(
                                    "column '{}' not found",
                                    col.value
                                ))
                            })?;
                        map.push(pos);
                    }
                    map
                };

                for row_exprs in rows_exprs {
                    let values: Vec<DbValue> = row_exprs
                        .iter()
                        .map(expr_to_dbvalue)
                        .collect::<Result<_, _>>()?;

                    let row = {
                        let mut full = vec![DbValue::Null; total];
                        for (i, pos) in map.iter().enumerate() {
                            full[*pos] = values[i].clone();
                        }
                        Row { values: full }
                    };

                    tx.insert(&table_name, &row)?;
                }
            }
            Statement::Query(query) => {
                let select = match query.body.as_ref() {
                    SetExpr::Select(select) => select,
                    _ => return Err(SqlError::UnsupportedStatement),
                };

                if select.from.len() != 1 || !select.from[0].joins.is_empty() {
                    return Err(SqlError::UnsupportedStatement);
                }
                let table_name = match &select.from[0].relation {
                    TableFactor::Table { name, .. } => name.to_string(),
                    _ => return Err(SqlError::UnsupportedStatement),
                };

                let schema = tx.get_schema(&table_name)?;
                let col_indices = resolve_projection(&select.projection, &schema)?;

                // Try to use an index or PK for simple `col = value` WHERE clauses.
                let eq_cond = select.selection.as_ref().and_then(try_extract_eq_condition);

                let rows: Vec<(crate::types::Key, Row)> =
                    if let Some((ref col_name, ref value)) = eq_cond {
                        let col_idx = resolve_column(col_name, &schema)?;
                        let pk_idx = schema.primary_key;

                        if pk_idx == col_idx {
                            // Primary key point lookup
                            if let DbValue::Integer(i) = value {
                                if *i >= 0 {
                                    match tx.get(&table_name, *i as u64)? {
                                        Some(row) => vec![(*i as u64, row)],
                                        None => vec![],
                                    }
                                } else {
                                    vec![]
                                }
                            } else {
                                // PK is always integer; non-integer value can't match
                                vec![]
                            }
                        } else {
                            let indexed_cols = tx.get_indexed_columns(&table_name)?;
                            if indexed_cols.contains(&col_name.to_string()) {
                                // Index scan
                                let bytes = db_value_to_bytes(value);
                                collect_scan_by_index(tx, &table_name, col_name, bytes.as_slice())?
                            } else {
                                // No index — fall back to full scan
                                tx.scan(&table_name, .., |_, _key, row| {
                                    let row =
                                        rkyv::deserialize::<Row, rkyv::rancor::Error>(row).unwrap();
                                    let matches = match &select.selection {
                                        Some(where_expr) => eval_where(where_expr, &row, &schema)?,
                                        None => true,
                                    };
                                    if matches {
                                        let projected = Row {
                                            values: col_indices
                                                .iter()
                                                .map(|&i| row.values[i].clone())
                                                .collect(),
                                        };
                                        result.push(projected);
                                    }
                                    Ok::<_, SqlError>(false)
                                })?;
                                return Ok(result);
                            }
                        }
                    } else {
                        collect_scan(tx, &table_name)?
                    };

                for (_, row) in &rows {
                    let matches = match &select.selection {
                        Some(where_expr) => eval_where(where_expr, row, &schema)?,
                        None => true,
                    };
                    if matches {
                        let projected = Row {
                            values: col_indices.iter().map(|&i| row.values[i].clone()).collect(),
                        };
                        result.push(projected);
                    }
                }
            }
            Statement::CreateIndex(ci) => {
                let table_name = ci.table_name.to_string();
                if ci.columns.len() != 1 {
                    return Err(SqlError::UnsupportedExpression(
                        "only single-column indexes are supported".into(),
                    ));
                }
                let column_name = match &ci.columns[0].column.expr {
                    Expr::Identifier(ident) => ident.value.clone(),
                    other => {
                        return Err(SqlError::UnsupportedExpression(other.to_string()));
                    }
                };
                tx.create_index(&table_name, &column_name)?;
            }
            Statement::Delete(del) => {
                let table_name = match del.from {
                    sqlparser::ast::FromTable::WithFromKeyword(table) => table
                        .first()
                        .map(|t| t.relation.to_string())
                        .ok_or(SqlError::UnsupportedStatement)?,
                    _ => return Err(SqlError::UnsupportedStatement),
                };

                let schema = tx.get_schema(&table_name)?;
                let pk_range = select_pk_eq_range(del.selection.as_ref(), &schema);
                tx.delete_range_where(&table_name, pk_range, |_, row| {
                    let row = rkyv::deserialize::<Row, rkyv::rancor::Error>(row).unwrap();
                    let matches = match &del.selection {
                        Some(where_expr) => eval_where(where_expr, &row, &schema)?,
                        None => true,
                    };
                    Ok::<_, SqlError>((matches, false))
                })?;
            }
            Statement::Update(upd) => {
                // Extract table name from the table reference
                let table_name = match &upd.table.relation {
                    sqlparser::ast::TableFactor::Table { name, .. } => name.to_string(),
                    _ => return Err(SqlError::UnsupportedStatement),
                };

                let schema = tx.get_schema(&table_name)?;

                // Build a map of column index to new value expression
                let mut assignments: std::collections::HashMap<usize, Expr> =
                    std::collections::HashMap::new();
                for assign in &upd.assignments {
                    // Each assignment's target is the column(s) to update
                    // For simple case, it's a single column identifier
                    if let sqlparser::ast::AssignmentTarget::ColumnName(name) = &assign.target {
                        // ObjectName converts to string for table.column format
                        let col_name = name.to_string();
                        // Handle qualified names (table.column) by taking just the column part
                        let col_name = if col_name.contains('.') {
                            col_name
                                .split('.')
                                .next_back()
                                .unwrap_or(&col_name)
                                .to_string()
                        } else {
                            col_name
                        };
                        let col_idx = resolve_column(&col_name, &schema)?;
                        assignments.insert(col_idx, assign.value.clone());
                    } else {
                        return Err(SqlError::UnsupportedExpression(
                            "complex column references in UPDATE not supported".into(),
                        ));
                    }
                }

                // Scan all rows
                let rows = collect_scan(tx, &table_name)?;

                // Update matching rows
                for (key, mut row) in rows {
                    let matches = match &upd.selection {
                        Some(where_expr) => eval_where(where_expr, &row, &schema)?,
                        None => true,
                    };

                    if matches {
                        // Apply assignments
                        for (&col_idx, expr) in &assignments {
                            let new_value = eval_expr(expr, &row, &schema)?;
                            row.values[col_idx] = new_value;
                        }
                        tx.update(&table_name, key, &row)?;
                    }
                }
            }
            _ => return Err(SqlError::UnsupportedStatement),
        }
    }

    Ok(result)
}

pub fn execute<const N: usize>(
    tx: &mut DbTransaction<'_, N>,
    sql: &str,
) -> Result<Vec<Row>, SqlError> {
    tx.with_lock(|mut tx| execute_locked(&mut tx, sql))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Database, Pager};
    use std::fs;
    use std::ops::RangeBounds;
    use tempfile::NamedTempFile;

    fn open_db() -> (Database<4096>, NamedTempFile) {
        let temp = NamedTempFile::new().unwrap();
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(temp.path())
            .unwrap();
        let pager = Pager::<4096>::new(file);
        let db = Database::create(pager).unwrap();
        (db, temp)
    }

    fn with_locked<const N: usize, T>(
        tx: &mut DbTransaction<'_, N>,
        f: impl FnOnce(&mut LockedDbTransaction<'_, N>) -> T,
    ) -> T {
        tx.with_lock(|mut locked| f(&mut locked))
    }

    fn insert<const N: usize>(
        tx: &mut DbTransaction<'_, N>,
        table_name: &str,
        row: &Row,
    ) -> Result<crate::types::Key, DatabaseError> {
        with_locked(tx, |tx| tx.insert(table_name, row))
    }

    fn get<const N: usize>(
        tx: &mut DbTransaction<'_, N>,
        table_name: &str,
        key: crate::types::Key,
    ) -> Result<Option<Row>, DatabaseError> {
        with_locked(tx, |tx| tx.get(table_name, key))
    }

    fn scan<const N: usize, R: RangeBounds<crate::types::Key>>(
        tx: &mut DbTransaction<'_, N>,
        table_name: &str,
        range: R,
    ) -> Result<Vec<(crate::types::Key, Row)>, DatabaseError> {
        let mut rows = Vec::new();
        with_locked(tx, |tx| {
            tx.scan(table_name, range, |_, key, row| {
                rows.push((key, rkyv::deserialize::<Row, rkyv::rancor::Error>(row)?));
                Ok::<_, DatabaseError>(false)
            })
        })?;
        Ok(rows)
    }

    fn scan_by_index<'b, const N: usize, R: RangeBounds<&'b [u8]>>(
        tx: &mut DbTransaction<'_, N>,
        table_name: &str,
        column_name: &str,
        range: R,
    ) -> Result<Vec<(crate::types::Key, Row)>, DatabaseError> {
        let mut rows = Vec::new();
        with_locked(tx, |tx| {
            tx.scan_by_index(table_name, column_name, range, |_, row, key| {
                rows.push((key, rkyv::deserialize::<Row, rkyv::rancor::Error>(row)?));
                Ok::<_, DatabaseError>(false)
            })
        })?;
        Ok(rows)
    }

    fn get_schema<const N: usize>(
        tx: &mut DbTransaction<'_, N>,
        table_name: &str,
    ) -> Result<Schema, DatabaseError> {
        with_locked(tx, |tx| tx.get_schema(table_name))
    }

    #[test]
    fn test_create_table_various_types() {
        use crate::{DbValue, Row};

        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(
            &mut tx,
            "CREATE TABLE items (
                id INTEGER NOT NULL,
                name VARCHAR(255) NOT NULL,
                price FLOAT,
                active BOOLEAN
            )",
        )
        .unwrap();

        // Verify by inserting a valid row with all types (prepend _rowid Null)
        let key = insert(
            &mut tx,
            "items",
            &Row {
                values: vec![
                    DbValue::Null, // _rowid
                    DbValue::Integer(1),
                    DbValue::Text("widget".into()),
                    DbValue::Float(9.99),
                    DbValue::Bool(true),
                ],
            },
        )
        .unwrap();
        let row = get(&mut tx, "items", key).unwrap().unwrap();
        assert_eq!(row.values[1], DbValue::Integer(1));
        assert_eq!(row.values[2], DbValue::Text("widget".into()));
        assert_eq!(row.values[3], DbValue::Float(9.99));
        assert_eq!(row.values[4], DbValue::Bool(true));

        // Nullable columns accept null
        insert(
            &mut tx,
            "items",
            &Row {
                values: vec![
                    DbValue::Null, // _rowid
                    DbValue::Integer(2),
                    DbValue::Text("gadget".into()),
                    DbValue::Null,
                    DbValue::Null,
                ],
            },
        )
        .unwrap();

        // NOT NULL columns reject null (id column)
        let err = insert(
            &mut tx,
            "items",
            &Row {
                values: vec![
                    DbValue::Null, // _rowid
                    DbValue::Null,
                    DbValue::Text("bad".into()),
                    DbValue::Null,
                    DbValue::Null,
                ],
            },
        )
        .unwrap_err();
        assert!(matches!(err, DatabaseError::SchemaMismatch(_)));
    }

    #[test]
    fn test_create_table_nullable_vs_not_null() {
        use crate::{DbValue, Row};

        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(
            &mut tx,
            "CREATE TABLE t (a INT NOT NULL, b INT, c TEXT NOT NULL, d TEXT)",
        )
        .unwrap();

        // b and d are nullable, a and c are NOT NULL
        insert(
            &mut tx,
            "t",
            &Row {
                values: vec![
                    DbValue::Null, // _rowid
                    DbValue::Integer(1),
                    DbValue::Null, // b nullable
                    DbValue::Text("x".into()),
                    DbValue::Null, // d nullable
                ],
            },
        )
        .unwrap();

        // a is NOT NULL — should reject
        let err = insert(
            &mut tx,
            "t",
            &Row {
                values: vec![
                    DbValue::Null, // _rowid
                    DbValue::Null,
                    DbValue::Integer(1),
                    DbValue::Text("x".into()),
                    DbValue::Null,
                ],
            },
        )
        .unwrap_err();
        assert!(matches!(err, DatabaseError::SchemaMismatch(_)));

        // c is NOT NULL — should reject
        let err = insert(
            &mut tx,
            "t",
            &Row {
                values: vec![
                    DbValue::Null, // _rowid
                    DbValue::Integer(1),
                    DbValue::Null,
                    DbValue::Null,
                    DbValue::Null,
                ],
            },
        )
        .unwrap_err();
        assert!(matches!(err, DatabaseError::SchemaMismatch(_)));
    }

    #[test]
    fn test_insert_basic() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(
            &mut tx,
            "CREATE TABLE users (name TEXT NOT NULL, age INTEGER NOT NULL)",
        )
        .unwrap();
        execute(&mut tx, "INSERT INTO users VALUES ('Alice', 30)").unwrap();

        let rows = scan(&mut tx, "users", ..).unwrap();
        assert_eq!(rows.len(), 1);
        // values[0] = _rowid, values[1] = name, values[2] = age
        assert_eq!(rows[0].1.values[1], DbValue::Text("Alice".into()));
        assert_eq!(rows[0].1.values[2], DbValue::Integer(30));
    }

    #[test]
    fn test_insert_multiple_rows() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(
            &mut tx,
            "CREATE TABLE users (name TEXT NOT NULL, age INTEGER NOT NULL)",
        )
        .unwrap();
        execute(
            &mut tx,
            "INSERT INTO users VALUES ('Alice', 30), ('Bob', 25)",
        )
        .unwrap();

        let rows = scan(&mut tx, "users", ..).unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_insert_with_column_list() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(
            &mut tx,
            "CREATE TABLE users (name TEXT NOT NULL, age INTEGER, active BOOLEAN)",
        )
        .unwrap();
        execute(
            &mut tx,
            "INSERT INTO users (active, name) VALUES (true, 'Alice')",
        )
        .unwrap();

        let rows = scan(&mut tx, "users", ..).unwrap();
        assert_eq!(rows.len(), 1);
        // values[0] = _rowid, values[1] = name, values[2] = age, values[3] = active
        assert_eq!(rows[0].1.values[1], DbValue::Text("Alice".into()));
        assert_eq!(rows[0].1.values[2], DbValue::Null); // age omitted
        assert_eq!(rows[0].1.values[3], DbValue::Bool(true));
    }

    #[test]
    fn test_insert_all_types() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(
            &mut tx,
            "CREATE TABLE t (i INTEGER, f FLOAT, t TEXT, b BOOLEAN)",
        )
        .unwrap();
        execute(&mut tx, "INSERT INTO t VALUES (42, 3.14, 'hello', true)").unwrap();

        let rows = scan(&mut tx, "t", ..).unwrap();
        assert_eq!(rows[0].1.values[1], DbValue::Integer(42));
        assert_eq!(rows[0].1.values[2], DbValue::Float(3.14));
        assert_eq!(rows[0].1.values[3], DbValue::Text("hello".into()));
        assert_eq!(rows[0].1.values[4], DbValue::Bool(true));
    }

    #[test]
    fn test_insert_null() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (a INTEGER, b TEXT)").unwrap();
        execute(&mut tx, "INSERT INTO t VALUES (NULL, NULL)").unwrap();

        let rows = scan(&mut tx, "t", ..).unwrap();
        assert_eq!(rows[0].1.values[1], DbValue::Null);
        assert_eq!(rows[0].1.values[2], DbValue::Null);
    }

    #[test]
    fn test_insert_negative_numbers() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (i INTEGER, f FLOAT)").unwrap();
        execute(&mut tx, "INSERT INTO t VALUES (-42, -3.14)").unwrap();

        let rows = scan(&mut tx, "t", ..).unwrap();
        assert_eq!(rows[0].1.values[1], DbValue::Integer(-42));
        assert_eq!(rows[0].1.values[2], DbValue::Float(-3.14));
    }

    #[test]
    fn test_insert_schema_mismatch() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (a INTEGER NOT NULL)").unwrap();
        let err = execute(&mut tx, "INSERT INTO t VALUES (NULL)").unwrap_err();
        assert!(matches!(err, SqlError::Database(_)));
    }

    #[test]
    fn test_unsupported_type() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        let err = execute(&mut tx, "CREATE TABLE t (a BLOB)").unwrap_err();
        assert!(matches!(err, SqlError::UnsupportedType(_)));
    }

    #[test]
    fn test_invalid_sql() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        let err = execute(&mut tx, "NOT VALID SQL AT ALL ???").unwrap_err();
        assert!(matches!(err, SqlError::Parse(_)));
    }

    #[test]
    fn test_unsupported_statement() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        // Try to delete from non-existent table - should fail with table not found
        let err = execute(&mut tx, "DELETE FROM nonexistent WHERE id = 1").unwrap_err();
        assert!(matches!(
            err,
            SqlError::Database(DatabaseError::TableNotFound(_))
        ));
    }

    #[test]
    fn test_create_index() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE users (name TEXT, age INTEGER)").unwrap();
        execute(&mut tx, "CREATE INDEX idx_name ON users (name)").unwrap();
        execute(
            &mut tx,
            "INSERT INTO users VALUES ('Alice', 30), ('Bob', 25)",
        )
        .unwrap();

        // Index should be usable via scan_by_index
        let rows = scan_by_index(
            &mut tx,
            "users",
            "name",
            b"Alice".as_ref()..=b"Alice".as_ref(),
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1.values[1], DbValue::Text("Alice".into()));
    }

    #[test]
    fn test_create_index_backfills() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (x INTEGER)").unwrap();
        execute(&mut tx, "INSERT INTO t VALUES (10), (20), (30)").unwrap();
        execute(&mut tx, "CREATE INDEX idx_x ON t (x)").unwrap();

        let key_20 = 20i64.to_be_bytes().to_vec();
        let rows = scan_by_index(&mut tx, "t", "x", key_20.as_slice()..=key_20.as_slice()).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1.values[1], DbValue::Integer(20));
    }

    #[test]
    fn test_create_index_persists_after_commit() {
        let (db, _tmp) = open_db();

        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (name TEXT, age INTEGER)").unwrap();
        execute(&mut tx, "CREATE INDEX idx_name ON t (name)").unwrap();
        execute(&mut tx, "INSERT INTO t VALUES ('Alice', 30)").unwrap();
        tx.commit().unwrap();

        let mut tx = db.begin_transaction();
        execute(&mut tx, "INSERT INTO t VALUES ('Bob', 25)").unwrap();
        tx.commit().unwrap();

        let mut tx = db.begin_transaction();
        let rows = scan_by_index(&mut tx, "t", "name", b"Bob".as_ref()..=b"Bob".as_ref()).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1.values[2], DbValue::Integer(25));
    }

    #[test]
    fn test_create_index_duplicate_error() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (x INTEGER)").unwrap();
        execute(&mut tx, "CREATE INDEX idx1 ON t (x)").unwrap();
        let err = execute(&mut tx, "CREATE INDEX idx2 ON t (x)").unwrap_err();
        assert!(matches!(err, SqlError::Database(_)));
    }

    #[test]
    fn test_delete_uses_range_delete_and_updates_indexes() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE users (name TEXT, age INTEGER)").unwrap();
        execute(&mut tx, "CREATE INDEX idx_name ON users (name)").unwrap();
        execute(
            &mut tx,
            "INSERT INTO users VALUES ('Alice', 30), ('Bob', 25), ('Charlie', 40)",
        )
        .unwrap();

        execute(&mut tx, "DELETE FROM users WHERE age >= 30").unwrap();

        let rows = scan(&mut tx, "users", ..).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1.values[1], DbValue::Text("Bob".into()));

        let alice = scan_by_index(
            &mut tx,
            "users",
            "name",
            b"Alice".as_ref()..=b"Alice".as_ref(),
        )
        .unwrap();
        let bob =
            scan_by_index(&mut tx, "users", "name", b"Bob".as_ref()..=b"Bob".as_ref()).unwrap();
        let charlie = scan_by_index(
            &mut tx,
            "users",
            "name",
            b"Charlie".as_ref()..=b"Charlie".as_ref(),
        )
        .unwrap();

        assert!(alice.is_empty());
        assert_eq!(bob.len(), 1);
        assert!(charlie.is_empty());
    }

    #[test]
    fn test_select_star() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(
            &mut tx,
            "CREATE TABLE users (name TEXT NOT NULL, age INTEGER NOT NULL)",
        )
        .unwrap();
        execute(
            &mut tx,
            "INSERT INTO users VALUES ('Alice', 30), ('Bob', 25)",
        )
        .unwrap();

        let rows = execute(&mut tx, "SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2);
        // 3 columns: _rowid + name + age
        assert_eq!(rows[0].values.len(), 3);
    }

    #[test]
    fn test_select_columns() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(
            &mut tx,
            "CREATE TABLE users (name TEXT, age INTEGER, active BOOLEAN)",
        )
        .unwrap();
        execute(&mut tx, "INSERT INTO users VALUES ('Alice', 30, true)").unwrap();

        let rows = execute(&mut tx, "SELECT age, name FROM users").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values.len(), 2);
        assert_eq!(rows[0].values[0], DbValue::Integer(30));
        assert_eq!(rows[0].values[1], DbValue::Text("Alice".into()));
    }

    #[test]
    fn test_select_where_eq() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE users (name TEXT, age INTEGER)").unwrap();
        execute(
            &mut tx,
            "INSERT INTO users VALUES ('Alice', 30), ('Bob', 25), ('Charlie', 30)",
        )
        .unwrap();

        let rows = execute(&mut tx, "SELECT * FROM users WHERE age = 30").unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_select_where_comparison() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (x INTEGER)").unwrap();
        execute(&mut tx, "INSERT INTO t VALUES (10), (20), (30), (40)").unwrap();

        assert_eq!(
            execute(&mut tx, "SELECT * FROM t WHERE x > 20")
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            execute(&mut tx, "SELECT * FROM t WHERE x >= 20")
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            execute(&mut tx, "SELECT * FROM t WHERE x < 20")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            execute(&mut tx, "SELECT * FROM t WHERE x <= 20")
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            execute(&mut tx, "SELECT * FROM t WHERE x <> 20")
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    fn test_select_where_and_or() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (a INTEGER, b INTEGER)").unwrap();
        execute(&mut tx, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)").unwrap();

        let rows = execute(&mut tx, "SELECT * FROM t WHERE a >= 2 AND b <= 20").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[1], DbValue::Integer(2));

        let rows = execute(&mut tx, "SELECT * FROM t WHERE a = 1 OR a = 3").unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_select_where_is_null() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (a INTEGER, b TEXT)").unwrap();
        execute(
            &mut tx,
            "INSERT INTO t VALUES (1, 'hello'), (2, NULL), (NULL, 'world')",
        )
        .unwrap();

        let rows = execute(&mut tx, "SELECT * FROM t WHERE b IS NULL").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[1], DbValue::Integer(2));

        let rows = execute(&mut tx, "SELECT * FROM t WHERE a IS NOT NULL").unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_select_where_null_comparison() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (a INTEGER)").unwrap();
        execute(&mut tx, "INSERT INTO t VALUES (1), (NULL)").unwrap();

        // NULL = NULL should be false (SQL semantics)
        let rows = execute(&mut tx, "SELECT * FROM t WHERE a = NULL").unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn test_select_where_string() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (name TEXT)").unwrap();
        execute(
            &mut tx,
            "INSERT INTO t VALUES ('Alice'), ('Bob'), ('Charlie')",
        )
        .unwrap();

        let rows = execute(&mut tx, "SELECT * FROM t WHERE name = 'Bob'").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[1], DbValue::Text("Bob".into()));
    }

    #[test]
    fn test_select_empty_result() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (a INTEGER)").unwrap();

        let rows = execute(&mut tx, "SELECT * FROM t").unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_insert_visible_after_commit() {
        let (db, _tmp) = open_db();

        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE users (name TEXT, age INTEGER)").unwrap();
        execute(
            &mut tx,
            "INSERT INTO users VALUES ('Alice', 30), ('Bob', 25)",
        )
        .unwrap();
        tx.commit().unwrap();

        let mut tx = db.begin_transaction();
        let rows = execute(&mut tx, "SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values[1], DbValue::Text("Alice".into()));
        assert_eq!(rows[0].values[2], DbValue::Integer(30));
        assert_eq!(rows[1].values[1], DbValue::Text("Bob".into()));
        assert_eq!(rows[1].values[2], DbValue::Integer(25));
    }

    #[test]
    fn test_insert_not_visible_after_rollback() {
        let (db, _tmp) = open_db();

        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE users (name TEXT, age INTEGER)").unwrap();
        tx.commit().unwrap();

        let mut tx = db.begin_transaction();
        execute(&mut tx, "INSERT INTO users VALUES ('Alice', 30)").unwrap();
        drop(tx); // rollback

        let mut tx = db.begin_transaction();
        let rows = execute(&mut tx, "SELECT * FROM users").unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_create_table_visible_after_commit() {
        let (db, _tmp) = open_db();

        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (x INTEGER)").unwrap();
        tx.commit().unwrap();

        let mut tx = db.begin_transaction();
        execute(&mut tx, "INSERT INTO t VALUES (42)").unwrap();
        let rows = execute(&mut tx, "SELECT * FROM t").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[1], DbValue::Integer(42));
    }

    #[test]
    fn test_select_across_commits() {
        let (db, _tmp) = open_db();

        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (x INTEGER)").unwrap();
        execute(&mut tx, "INSERT INTO t VALUES (1)").unwrap();
        tx.commit().unwrap();

        let mut tx = db.begin_transaction();
        execute(&mut tx, "INSERT INTO t VALUES (2)").unwrap();
        tx.commit().unwrap();

        let mut tx = db.begin_transaction();
        execute(&mut tx, "INSERT INTO t VALUES (3)").unwrap();
        tx.commit().unwrap();

        let mut tx = db.begin_transaction();
        let rows = execute(&mut tx, "SELECT * FROM t WHERE x >= 2").unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_type_aliases() {
        use crate::{DbValue, Row};

        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(
            &mut tx,
            "CREATE TABLE t (
                a BIGINT,
                b CHAR(10),
                c DOUBLE,
                d REAL,
                e BOOL
            )",
        )
        .unwrap();

        // Verify types by inserting matching values (prepend _rowid Null)
        let key = insert(
            &mut tx,
            "t",
            &Row {
                values: vec![
                    DbValue::Null,              // _rowid
                    DbValue::Integer(42),       // BIGINT -> Integer
                    DbValue::Text("hi".into()), // CHAR -> Text
                    DbValue::Float(1.0),        // DOUBLE -> Float
                    DbValue::Float(2.0),        // REAL -> Float
                    DbValue::Bool(false),       // BOOL -> Bool
                ],
            },
        )
        .unwrap();
        let row = get(&mut tx, "t", key).unwrap().unwrap();
        assert_eq!(row.values[1], DbValue::Integer(42));
        assert_eq!(row.values[2], DbValue::Text("hi".into()));
        assert_eq!(row.values[3], DbValue::Float(1.0));
        assert_eq!(row.values[4], DbValue::Float(2.0));
        assert_eq!(row.values[5], DbValue::Bool(false));
    }

    // ── Primary key SQL tests ──────────────────────────────────────

    #[test]
    fn test_sql_create_table_with_primary_key() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(
            &mut tx,
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
        )
        .unwrap();

        let schema = get_schema(&mut tx, "items").unwrap();
        assert_eq!(schema.columns.len(), 2);
        assert_eq!(schema.columns[0].name, "id");
        assert_eq!(schema.primary_key, 0);
    }

    #[test]
    fn test_sql_insert_with_explicit_pk() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(
            &mut tx,
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
        )
        .unwrap();
        execute(&mut tx, "INSERT INTO items VALUES (10, 'widget')").unwrap();
        execute(&mut tx, "INSERT INTO items VALUES (20, 'gadget')").unwrap();

        // B-tree key should match PK value
        let row = get(&mut tx, "items", 10).unwrap().unwrap();
        assert_eq!(row.values[0], DbValue::Integer(10));
        assert_eq!(row.values[1], DbValue::Text("widget".into()));

        let row = get(&mut tx, "items", 20).unwrap().unwrap();
        assert_eq!(row.values[1], DbValue::Text("gadget".into()));
    }

    #[test]
    fn test_sql_duplicate_pk_error() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(
            &mut tx,
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)",
        )
        .unwrap();
        execute(&mut tx, "INSERT INTO items VALUES (1, 'a')").unwrap();
        let err = execute(&mut tx, "INSERT INTO items VALUES (1, 'b')").unwrap_err();
        assert!(matches!(
            err,
            SqlError::Database(DatabaseError::DuplicateKey(1))
        ));
    }

    #[test]
    fn test_sql_select_with_pk_table() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(
            &mut tx,
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
        )
        .unwrap();
        execute(&mut tx, "INSERT INTO items VALUES (1, 'Alice'), (2, 'Bob')").unwrap();

        let rows = execute(&mut tx, "SELECT name FROM items WHERE id = 2").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[0], DbValue::Text("Bob".into()));
    }

    #[test]
    fn test_sql_implicit_rowid_not_in_insert() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (x INTEGER, y TEXT)").unwrap();
        // INSERT without column list should not require _rowid
        execute(&mut tx, "INSERT INTO t VALUES (1, 'a'), (2, 'b')").unwrap();

        let rows = execute(&mut tx, "SELECT x, y FROM t").unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values[0], DbValue::Integer(1));
        assert_eq!(rows[0].values[1], DbValue::Text("a".into()));
    }

    #[test]
    fn test_sql_select_rowid() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (x INTEGER)").unwrap();
        execute(&mut tx, "INSERT INTO t VALUES (100)").unwrap();

        // _rowid is queryable via SELECT
        let rows = execute(&mut tx, "SELECT _rowid, x FROM t").unwrap();
        assert_eq!(rows.len(), 1);
        // _rowid should be auto-assigned (0)
        assert_eq!(rows[0].values[0], DbValue::Integer(0));
        assert_eq!(rows[0].values[1], DbValue::Integer(100));
    }

    // ── Index-optimized SELECT tests ─────────────────────────────

    #[test]
    fn test_select_uses_index_for_eq() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE users (name TEXT, age INTEGER)").unwrap();
        execute(&mut tx, "CREATE INDEX idx_name ON users (name)").unwrap();
        execute(
            &mut tx,
            "INSERT INTO users VALUES ('Alice', 30), ('Bob', 25), ('Charlie', 30)",
        )
        .unwrap();

        let rows = execute(&mut tx, "SELECT * FROM users WHERE name = 'Bob'").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[1], DbValue::Text("Bob".into()));
        assert_eq!(rows[0].values[2], DbValue::Integer(25));
    }

    #[test]
    fn test_select_uses_index_multiple_matches() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (x INTEGER, y TEXT)").unwrap();
        execute(&mut tx, "CREATE INDEX idx_x ON t (x)").unwrap();
        execute(
            &mut tx,
            "INSERT INTO t VALUES (1, 'a'), (2, 'b'), (1, 'c'), (3, 'd'), (1, 'e')",
        )
        .unwrap();

        let rows = execute(&mut tx, "SELECT y FROM t WHERE x = 1").unwrap();
        assert_eq!(rows.len(), 3);
        let mut vals: Vec<String> = rows
            .iter()
            .map(|r| match &r.values[0] {
                DbValue::Text(s) => s.clone(),
                _ => panic!("expected text"),
            })
            .collect();
        vals.sort();
        assert_eq!(vals, vec!["a", "c", "e"]);
    }

    #[test]
    fn test_select_pk_point_lookup() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(
            &mut tx,
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
        )
        .unwrap();
        execute(
            &mut tx,
            "INSERT INTO items VALUES (10, 'widget'), (20, 'gadget'), (30, 'doohickey')",
        )
        .unwrap();

        let rows = execute(&mut tx, "SELECT name FROM items WHERE id = 20").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[0], DbValue::Text("gadget".into()));

        // Non-existent PK
        let rows = execute(&mut tx, "SELECT name FROM items WHERE id = 99").unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_select_eq_no_index_falls_back() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (x INTEGER, y TEXT)").unwrap();
        // No index created
        execute(&mut tx, "INSERT INTO t VALUES (1, 'a'), (2, 'b'), (1, 'c')").unwrap();

        // Should still work via full scan fallback
        let rows = execute(&mut tx, "SELECT y FROM t WHERE x = 1").unwrap();
        assert_eq!(rows.len(), 2);
    }

    // ── UPDATE statement tests ────────────────────────────────────

    #[test]
    fn test_update_basic() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE users (name TEXT, age INTEGER)").unwrap();
        execute(
            &mut tx,
            "INSERT INTO users VALUES ('Alice', 30), ('Bob', 25)",
        )
        .unwrap();

        // Update a single column
        execute(&mut tx, "UPDATE users SET age = 31 WHERE name = 'Alice'").unwrap();

        let rows = execute(&mut tx, "SELECT * FROM users WHERE name = 'Alice'").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[1], DbValue::Text("Alice".into()));
        assert_eq!(rows[0].values[2], DbValue::Integer(31));
    }

    #[test]
    fn test_update_multiple_columns() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(
            &mut tx,
            "CREATE TABLE users (name TEXT, age INTEGER, city TEXT)",
        )
        .unwrap();
        execute(
            &mut tx,
            "INSERT INTO users VALUES ('Alice', 30, 'NYC'), ('Bob', 25, 'LA')",
        )
        .unwrap();

        // Update multiple columns
        execute(
            &mut tx,
            "UPDATE users SET age = 31, city = 'Boston' WHERE name = 'Alice'",
        )
        .unwrap();

        let rows = execute(&mut tx, "SELECT * FROM users WHERE name = 'Alice'").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[2], DbValue::Integer(31));
        assert_eq!(rows[0].values[3], DbValue::Text("Boston".into()));
    }

    #[test]
    fn test_update_all_rows() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (x INTEGER)").unwrap();
        execute(&mut tx, "INSERT INTO t VALUES (1), (2), (3)").unwrap();

        // Update without WHERE clause updates all rows
        execute(&mut tx, "UPDATE t SET x = 10").unwrap();

        let rows = execute(&mut tx, "SELECT * FROM t").unwrap();
        assert_eq!(rows.len(), 3);
        for row in rows {
            assert_eq!(row.values[1], DbValue::Integer(10));
        }
    }

    #[test]
    fn test_update_multiple_rows() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (x INTEGER, active BOOLEAN)").unwrap();
        execute(
            &mut tx,
            "INSERT INTO t VALUES (1, true), (2, true), (3, false)",
        )
        .unwrap();

        // Update multiple matching rows
        execute(&mut tx, "UPDATE t SET active = false WHERE x <= 2").unwrap();

        let rows = execute(&mut tx, "SELECT * FROM t").unwrap();
        assert_eq!(rows.len(), 3);

        // First two should be false now
        for i in 0..2 {
            assert_eq!(rows[i].values[2], DbValue::Bool(false));
        }
        // Third should still be false
        assert_eq!(rows[2].values[2], DbValue::Bool(false));
    }

    #[test]
    fn test_update_with_expression() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (x INTEGER, y TEXT)").unwrap();
        execute(&mut tx, "INSERT INTO t VALUES (1, 'hello'), (2, 'world')").unwrap();

        // Update with literal value (not computed expressions yet, but test existing value)
        execute(&mut tx, "UPDATE t SET y = 'updated' WHERE x = 1").unwrap();

        let rows = execute(&mut tx, "SELECT * FROM t WHERE x = 1").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[2], DbValue::Text("updated".into()));
    }

    #[test]
    fn test_update_to_null() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (x INTEGER, y TEXT)").unwrap();
        execute(&mut tx, "INSERT INTO t VALUES (1, 'hello')").unwrap();

        // Update to NULL
        execute(&mut tx, "UPDATE t SET y = NULL WHERE x = 1").unwrap();

        let rows = execute(&mut tx, "SELECT * FROM t").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[2], DbValue::Null);
    }

    #[test]
    fn test_update_no_matching_rows() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (x INTEGER)").unwrap();
        execute(&mut tx, "INSERT INTO t VALUES (1), (2)").unwrap();

        // This should succeed but update no rows
        execute(&mut tx, "UPDATE t SET x = 99 WHERE x = 100").unwrap();

        let rows = execute(&mut tx, "SELECT * FROM t").unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values[1], DbValue::Integer(1));
        assert_eq!(rows[1].values[1], DbValue::Integer(2));
    }

    #[test]
    fn test_update_with_complex_where() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (a INTEGER, b INTEGER, c TEXT)").unwrap();
        execute(
            &mut tx,
            "INSERT INTO t VALUES (1, 10, 'x'), (2, 20, 'y'), (3, 15, 'z')",
        )
        .unwrap();

        // Update with AND condition
        execute(
            &mut tx,
            "UPDATE t SET c = 'updated' WHERE a >= 2 AND b <= 20",
        )
        .unwrap();

        let rows = execute(&mut tx, "SELECT * FROM t").unwrap();
        assert_eq!(rows.len(), 3);
        // Should update rows 2 and 3 (a >= 2 AND b <= 20)
        assert_eq!(rows[0].values[3], DbValue::Text("x".into())); // a=1, not updated
        assert_eq!(rows[1].values[3], DbValue::Text("updated".into())); // a=2, b=20, updated
        assert_eq!(rows[2].values[3], DbValue::Text("updated".into())); // a=3, b=15, updated
    }

    #[test]
    fn test_update_with_pk_table() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(
            &mut tx,
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, price FLOAT)",
        )
        .unwrap();
        execute(
            &mut tx,
            "INSERT INTO items VALUES (1, 'widget', 9.99), (2, 'gadget', 19.99)",
        )
        .unwrap();

        // Update by PK
        execute(&mut tx, "UPDATE items SET price = 15.50 WHERE id = 2").unwrap();

        let row = get(&mut tx, "items", 2).unwrap().unwrap();
        assert_eq!(row.values[0], DbValue::Integer(2)); // id column
        assert_eq!(row.values[1], DbValue::Text("gadget".into())); // name column
        assert_eq!(row.values[2], DbValue::Float(15.50)); // price column
    }

    #[test]
    fn test_update_different_types() {
        let (db, _tmp) = open_db();
        let mut tx = db.begin_transaction();
        execute(
            &mut tx,
            "CREATE TABLE t (i INTEGER, f FLOAT, t TEXT, b BOOLEAN)",
        )
        .unwrap();
        execute(&mut tx, "INSERT INTO t VALUES (1, 1.0, 'a', true)").unwrap();

        // Update different column types
        execute(
            &mut tx,
            "UPDATE t SET i = 42, f = 3.14, t = 'updated', b = false WHERE i = 1",
        )
        .unwrap();

        let rows = execute(&mut tx, "SELECT * FROM t").unwrap();
        assert_eq!(rows[0].values[1], DbValue::Integer(42));
        assert_eq!(rows[0].values[2], DbValue::Float(3.14));
        assert_eq!(rows[0].values[3], DbValue::Text("updated".into()));
        assert_eq!(rows[0].values[4], DbValue::Bool(false));
    }

    #[test]
    fn test_update_visible_after_commit() {
        let (db, _tmp) = open_db();

        let mut tx = db.begin_transaction();
        execute(&mut tx, "CREATE TABLE t (x INTEGER)").unwrap();
        execute(&mut tx, "INSERT INTO t VALUES (1)").unwrap();
        tx.commit().unwrap();

        let mut tx = db.begin_transaction();
        execute(&mut tx, "UPDATE t SET x = 99").unwrap();
        tx.commit().unwrap();

        let mut tx = db.begin_transaction();
        let rows = execute(&mut tx, "SELECT * FROM t").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[1], DbValue::Integer(99));
    }
}
