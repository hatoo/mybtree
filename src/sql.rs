use sqlparser::ast::{
    BinaryOperator, ColumnOption, DataType, Expr, SelectItem, SetExpr, Statement, TableFactor,
    UnaryOperator, Value,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::database::{Column, ColumnType, DatabaseError, DbTransaction, DbValue, Row, Schema};

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

pub fn execute<const N: usize>(tx: &DbTransaction<'_, N>, sql: &str) -> Result<Vec<Row>, SqlError> {
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
                    let is_pk = col_def.options.iter().any(|opt| {
                        matches!(opt.option, ColumnOption::PrimaryKey { .. })
                    });
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
                    primary_key,
                };
                tx.create_table(&ct.name.to_string(), schema)?;
            }
            Statement::Insert(ins) => {
                let table_name = ins.table.to_string();
                let source = ins.source.as_ref().ok_or(SqlError::UnsupportedStatement)?;
                let rows_exprs = match source.body.as_ref() {
                    SetExpr::Values(values) => &values.rows,
                    _ => return Err(SqlError::UnsupportedStatement),
                };

                let schema = tx.get_schema(&table_name)?;

                // Detect whether the table has an implicit _rowid column
                // that the user is not explicitly providing.
                let has_implicit_rowid = schema.columns.first().map_or(false, |c| c.name == "_rowid");

                let column_map = if ins.columns.is_empty() {
                    None
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
                    Some((map, schema.columns.len()))
                };

                for row_exprs in rows_exprs {
                    let values: Vec<DbValue> = row_exprs
                        .iter()
                        .map(expr_to_dbvalue)
                        .collect::<Result<_, _>>()?;

                    let row = if let Some((ref map, total)) = column_map {
                        let mut full = vec![DbValue::Null; total];
                        for (i, pos) in map.iter().enumerate() {
                            full[*pos] = values[i].clone();
                        }
                        Row { values: full }
                    } else if has_implicit_rowid {
                        // User provided values without column list and table has
                        // implicit _rowid — prepend Null so _rowid auto-assigns.
                        let mut full = Vec::with_capacity(values.len() + 1);
                        full.push(DbValue::Null);
                        full.extend(values);
                        Row { values: full }
                    } else {
                        Row { values }
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
                let all_rows = tx.scan(&table_name, ..)?;

                for (_, row) in &all_rows {
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
            _ => return Err(SqlError::UnsupportedStatement),
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Database, Pager};
    use std::fs;
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

    #[test]
    fn test_create_table_various_types() {
        use crate::{DbValue, Row};

        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(
            &tx,
            "CREATE TABLE items (
                id INTEGER NOT NULL,
                name VARCHAR(255) NOT NULL,
                price FLOAT,
                active BOOLEAN
            )",
        )
        .unwrap();

        // Verify by inserting a valid row with all types (prepend _rowid Null)
        let key = tx
            .insert(
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
        let row = tx.get("items", key).unwrap().unwrap();
        assert_eq!(row.values[1], DbValue::Integer(1));
        assert_eq!(row.values[2], DbValue::Text("widget".into()));
        assert_eq!(row.values[3], DbValue::Float(9.99));
        assert_eq!(row.values[4], DbValue::Bool(true));

        // Nullable columns accept null
        tx.insert(
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
        let err = tx
            .insert(
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
        let tx = db.begin_transaction();
        execute(
            &tx,
            "CREATE TABLE t (a INT NOT NULL, b INT, c TEXT NOT NULL, d TEXT)",
        )
        .unwrap();

        // b and d are nullable, a and c are NOT NULL
        tx.insert(
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
        let err = tx
            .insert(
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
        let err = tx
            .insert(
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
        let tx = db.begin_transaction();
        execute(
            &tx,
            "CREATE TABLE users (name TEXT NOT NULL, age INTEGER NOT NULL)",
        )
        .unwrap();
        execute(&tx, "INSERT INTO users VALUES ('Alice', 30)").unwrap();

        let rows = tx.scan("users", ..).unwrap();
        assert_eq!(rows.len(), 1);
        // values[0] = _rowid, values[1] = name, values[2] = age
        assert_eq!(rows[0].1.values[1], DbValue::Text("Alice".into()));
        assert_eq!(rows[0].1.values[2], DbValue::Integer(30));
    }

    #[test]
    fn test_insert_multiple_rows() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(
            &tx,
            "CREATE TABLE users (name TEXT NOT NULL, age INTEGER NOT NULL)",
        )
        .unwrap();
        execute(&tx, "INSERT INTO users VALUES ('Alice', 30), ('Bob', 25)").unwrap();

        let rows = tx.scan("users", ..).unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_insert_with_column_list() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(
            &tx,
            "CREATE TABLE users (name TEXT NOT NULL, age INTEGER, active BOOLEAN)",
        )
        .unwrap();
        execute(
            &tx,
            "INSERT INTO users (active, name) VALUES (true, 'Alice')",
        )
        .unwrap();

        let rows = tx.scan("users", ..).unwrap();
        assert_eq!(rows.len(), 1);
        // values[0] = _rowid, values[1] = name, values[2] = age, values[3] = active
        assert_eq!(rows[0].1.values[1], DbValue::Text("Alice".into()));
        assert_eq!(rows[0].1.values[2], DbValue::Null); // age omitted
        assert_eq!(rows[0].1.values[3], DbValue::Bool(true));
    }

    #[test]
    fn test_insert_all_types() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(
            &tx,
            "CREATE TABLE t (i INTEGER, f FLOAT, t TEXT, b BOOLEAN)",
        )
        .unwrap();
        execute(&tx, "INSERT INTO t VALUES (42, 3.14, 'hello', true)").unwrap();

        let rows = tx.scan("t", ..).unwrap();
        assert_eq!(rows[0].1.values[1], DbValue::Integer(42));
        assert_eq!(rows[0].1.values[2], DbValue::Float(3.14));
        assert_eq!(rows[0].1.values[3], DbValue::Text("hello".into()));
        assert_eq!(rows[0].1.values[4], DbValue::Bool(true));
    }

    #[test]
    fn test_insert_null() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE t (a INTEGER, b TEXT)").unwrap();
        execute(&tx, "INSERT INTO t VALUES (NULL, NULL)").unwrap();

        let rows = tx.scan("t", ..).unwrap();
        assert_eq!(rows[0].1.values[1], DbValue::Null);
        assert_eq!(rows[0].1.values[2], DbValue::Null);
    }

    #[test]
    fn test_insert_negative_numbers() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE t (i INTEGER, f FLOAT)").unwrap();
        execute(&tx, "INSERT INTO t VALUES (-42, -3.14)").unwrap();

        let rows = tx.scan("t", ..).unwrap();
        assert_eq!(rows[0].1.values[1], DbValue::Integer(-42));
        assert_eq!(rows[0].1.values[2], DbValue::Float(-3.14));
    }

    #[test]
    fn test_insert_schema_mismatch() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE t (a INTEGER NOT NULL)").unwrap();
        let err = execute(&tx, "INSERT INTO t VALUES (NULL)").unwrap_err();
        assert!(matches!(err, SqlError::Database(_)));
    }

    #[test]
    fn test_unsupported_type() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        let err = execute(&tx, "CREATE TABLE t (a BLOB)").unwrap_err();
        assert!(matches!(err, SqlError::UnsupportedType(_)));
    }

    #[test]
    fn test_invalid_sql() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        let err = execute(&tx, "NOT VALID SQL AT ALL ???").unwrap_err();
        assert!(matches!(err, SqlError::Parse(_)));
    }

    #[test]
    fn test_unsupported_statement() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        let err = execute(&tx, "DELETE FROM t WHERE id = 1").unwrap_err();
        assert!(matches!(err, SqlError::UnsupportedStatement));
    }

    #[test]
    fn test_create_index() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE users (name TEXT, age INTEGER)").unwrap();
        execute(&tx, "CREATE INDEX idx_name ON users (name)").unwrap();
        execute(&tx, "INSERT INTO users VALUES ('Alice', 30), ('Bob', 25)").unwrap();

        // Index should be usable via scan_by_index
        let rows = tx
            .scan_by_index("users", "name", b"Alice".as_ref()..=b"Alice".as_ref())
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1.values[1], DbValue::Text("Alice".into()));
    }

    #[test]
    fn test_create_index_backfills() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE t (x INTEGER)").unwrap();
        execute(&tx, "INSERT INTO t VALUES (10), (20), (30)").unwrap();
        execute(&tx, "CREATE INDEX idx_x ON t (x)").unwrap();

        let key_20 = 20i64.to_be_bytes().to_vec();
        let rows = tx
            .scan_by_index("t", "x", key_20.as_slice()..=key_20.as_slice())
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1.values[1], DbValue::Integer(20));
    }

    #[test]
    fn test_create_index_persists_after_commit() {
        let (db, _tmp) = open_db();

        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE t (name TEXT, age INTEGER)").unwrap();
        execute(&tx, "CREATE INDEX idx_name ON t (name)").unwrap();
        execute(&tx, "INSERT INTO t VALUES ('Alice', 30)").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        execute(&tx, "INSERT INTO t VALUES ('Bob', 25)").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let rows = tx
            .scan_by_index("t", "name", b"Bob".as_ref()..=b"Bob".as_ref())
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1.values[2], DbValue::Integer(25));
    }

    #[test]
    fn test_create_index_duplicate_error() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE t (x INTEGER)").unwrap();
        execute(&tx, "CREATE INDEX idx1 ON t (x)").unwrap();
        let err = execute(&tx, "CREATE INDEX idx2 ON t (x)").unwrap_err();
        assert!(matches!(err, SqlError::Database(_)));
    }

    #[test]
    fn test_select_star() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(
            &tx,
            "CREATE TABLE users (name TEXT NOT NULL, age INTEGER NOT NULL)",
        )
        .unwrap();
        execute(&tx, "INSERT INTO users VALUES ('Alice', 30), ('Bob', 25)").unwrap();

        let rows = execute(&tx, "SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2);
        // 3 columns: _rowid + name + age
        assert_eq!(rows[0].values.len(), 3);
    }

    #[test]
    fn test_select_columns() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(
            &tx,
            "CREATE TABLE users (name TEXT, age INTEGER, active BOOLEAN)",
        )
        .unwrap();
        execute(&tx, "INSERT INTO users VALUES ('Alice', 30, true)").unwrap();

        let rows = execute(&tx, "SELECT age, name FROM users").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values.len(), 2);
        assert_eq!(rows[0].values[0], DbValue::Integer(30));
        assert_eq!(rows[0].values[1], DbValue::Text("Alice".into()));
    }

    #[test]
    fn test_select_where_eq() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE users (name TEXT, age INTEGER)").unwrap();
        execute(
            &tx,
            "INSERT INTO users VALUES ('Alice', 30), ('Bob', 25), ('Charlie', 30)",
        )
        .unwrap();

        let rows = execute(&tx, "SELECT * FROM users WHERE age = 30").unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_select_where_comparison() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE t (x INTEGER)").unwrap();
        execute(&tx, "INSERT INTO t VALUES (10), (20), (30), (40)").unwrap();

        assert_eq!(
            execute(&tx, "SELECT * FROM t WHERE x > 20").unwrap().len(),
            2
        );
        assert_eq!(
            execute(&tx, "SELECT * FROM t WHERE x >= 20").unwrap().len(),
            3
        );
        assert_eq!(
            execute(&tx, "SELECT * FROM t WHERE x < 20").unwrap().len(),
            1
        );
        assert_eq!(
            execute(&tx, "SELECT * FROM t WHERE x <= 20").unwrap().len(),
            2
        );
        assert_eq!(
            execute(&tx, "SELECT * FROM t WHERE x <> 20").unwrap().len(),
            3
        );
    }

    #[test]
    fn test_select_where_and_or() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE t (a INTEGER, b INTEGER)").unwrap();
        execute(&tx, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)").unwrap();

        let rows = execute(&tx, "SELECT * FROM t WHERE a >= 2 AND b <= 20").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[1], DbValue::Integer(2));

        let rows = execute(&tx, "SELECT * FROM t WHERE a = 1 OR a = 3").unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_select_where_is_null() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE t (a INTEGER, b TEXT)").unwrap();
        execute(
            &tx,
            "INSERT INTO t VALUES (1, 'hello'), (2, NULL), (NULL, 'world')",
        )
        .unwrap();

        let rows = execute(&tx, "SELECT * FROM t WHERE b IS NULL").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[1], DbValue::Integer(2));

        let rows = execute(&tx, "SELECT * FROM t WHERE a IS NOT NULL").unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_select_where_null_comparison() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE t (a INTEGER)").unwrap();
        execute(&tx, "INSERT INTO t VALUES (1), (NULL)").unwrap();

        // NULL = NULL should be false (SQL semantics)
        let rows = execute(&tx, "SELECT * FROM t WHERE a = NULL").unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn test_select_where_string() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE t (name TEXT)").unwrap();
        execute(&tx, "INSERT INTO t VALUES ('Alice'), ('Bob'), ('Charlie')").unwrap();

        let rows = execute(&tx, "SELECT * FROM t WHERE name = 'Bob'").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[1], DbValue::Text("Bob".into()));
    }

    #[test]
    fn test_select_empty_result() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE t (a INTEGER)").unwrap();

        let rows = execute(&tx, "SELECT * FROM t").unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_insert_visible_after_commit() {
        let (db, _tmp) = open_db();

        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE users (name TEXT, age INTEGER)").unwrap();
        execute(&tx, "INSERT INTO users VALUES ('Alice', 30), ('Bob', 25)").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let rows = execute(&tx, "SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values[1], DbValue::Text("Alice".into()));
        assert_eq!(rows[0].values[2], DbValue::Integer(30));
        assert_eq!(rows[1].values[1], DbValue::Text("Bob".into()));
        assert_eq!(rows[1].values[2], DbValue::Integer(25));
    }

    #[test]
    fn test_insert_not_visible_after_rollback() {
        let (db, _tmp) = open_db();

        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE users (name TEXT, age INTEGER)").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        execute(&tx, "INSERT INTO users VALUES ('Alice', 30)").unwrap();
        drop(tx); // rollback

        let tx = db.begin_transaction();
        let rows = execute(&tx, "SELECT * FROM users").unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_create_table_visible_after_commit() {
        let (db, _tmp) = open_db();

        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE t (x INTEGER)").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        execute(&tx, "INSERT INTO t VALUES (42)").unwrap();
        let rows = execute(&tx, "SELECT * FROM t").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[1], DbValue::Integer(42));
    }

    #[test]
    fn test_select_across_commits() {
        let (db, _tmp) = open_db();

        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE t (x INTEGER)").unwrap();
        execute(&tx, "INSERT INTO t VALUES (1)").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        execute(&tx, "INSERT INTO t VALUES (2)").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        execute(&tx, "INSERT INTO t VALUES (3)").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let rows = execute(&tx, "SELECT * FROM t WHERE x >= 2").unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_type_aliases() {
        use crate::{DbValue, Row};

        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(
            &tx,
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
        let key = tx
            .insert(
                "t",
                &Row {
                    values: vec![
                        DbValue::Null,               // _rowid
                        DbValue::Integer(42),         // BIGINT → Integer
                        DbValue::Text("hi".into()),   // CHAR → Text
                        DbValue::Float(1.0),          // DOUBLE → Float
                        DbValue::Float(2.0),          // REAL → Float
                        DbValue::Bool(false),         // BOOL → Bool
                    ],
                },
            )
            .unwrap();
        let row = tx.get("t", key).unwrap().unwrap();
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
        let tx = db.begin_transaction();
        execute(
            &tx,
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
        )
        .unwrap();

        let schema = tx.get_schema("items").unwrap();
        assert_eq!(schema.columns.len(), 2);
        assert_eq!(schema.columns[0].name, "id");
        assert_eq!(schema.primary_key, Some(0));
    }

    #[test]
    fn test_sql_insert_with_explicit_pk() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(
            &tx,
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
        )
        .unwrap();
        execute(&tx, "INSERT INTO items VALUES (10, 'widget')").unwrap();
        execute(&tx, "INSERT INTO items VALUES (20, 'gadget')").unwrap();

        // B-tree key should match PK value
        let row = tx.get("items", 10).unwrap().unwrap();
        assert_eq!(row.values[0], DbValue::Integer(10));
        assert_eq!(row.values[1], DbValue::Text("widget".into()));

        let row = tx.get("items", 20).unwrap().unwrap();
        assert_eq!(row.values[1], DbValue::Text("gadget".into()));
    }

    #[test]
    fn test_sql_duplicate_pk_error() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(
            &tx,
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)",
        )
        .unwrap();
        execute(&tx, "INSERT INTO items VALUES (1, 'a')").unwrap();
        let err = execute(&tx, "INSERT INTO items VALUES (1, 'b')").unwrap_err();
        assert!(matches!(err, SqlError::Database(DatabaseError::DuplicateKey(1))));
    }

    #[test]
    fn test_sql_select_with_pk_table() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(
            &tx,
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
        )
        .unwrap();
        execute(&tx, "INSERT INTO items VALUES (1, 'Alice'), (2, 'Bob')").unwrap();

        let rows = execute(&tx, "SELECT name FROM items WHERE id = 2").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[0], DbValue::Text("Bob".into()));
    }

    #[test]
    fn test_sql_implicit_rowid_not_in_insert() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE t (x INTEGER, y TEXT)").unwrap();
        // INSERT without column list should not require _rowid
        execute(&tx, "INSERT INTO t VALUES (1, 'a'), (2, 'b')").unwrap();

        let rows = execute(&tx, "SELECT x, y FROM t").unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values[0], DbValue::Integer(1));
        assert_eq!(rows[0].values[1], DbValue::Text("a".into()));
    }

    #[test]
    fn test_sql_select_rowid() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE t (x INTEGER)").unwrap();
        execute(&tx, "INSERT INTO t VALUES (100)").unwrap();

        // _rowid is queryable via SELECT
        let rows = execute(&tx, "SELECT _rowid, x FROM t").unwrap();
        assert_eq!(rows.len(), 1);
        // _rowid should be auto-assigned (0)
        assert_eq!(rows[0].values[0], DbValue::Integer(0));
        assert_eq!(rows[0].values[1], DbValue::Integer(100));
    }
}
