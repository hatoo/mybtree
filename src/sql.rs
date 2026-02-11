use sqlparser::ast::{
    ColumnOption, DataType, Expr, SetExpr, Statement, UnaryOperator, Value,
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
                    let f: f64 = s
                        .parse()
                        .map_err(|_| SqlError::InvalidValue(s.clone()))?;
                    Ok(DbValue::Float(f))
                } else {
                    let i: i64 = s
                        .parse()
                        .map_err(|_| SqlError::InvalidValue(s.clone()))?;
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
        Expr::UnaryOp { op, expr } if *op == UnaryOperator::Minus => {
            match expr_to_dbvalue(expr)? {
                DbValue::Integer(i) => Ok(DbValue::Integer(-i)),
                DbValue::Float(f) => Ok(DbValue::Float(-f)),
                other => Err(SqlError::UnsupportedExpression(format!("-{:?}", other))),
            }
        }
        other => Err(SqlError::UnsupportedExpression(other.to_string())),
    }
}

pub fn execute(tx: &DbTransaction, sql: &str) -> Result<(), SqlError> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)?;

    for stmt in statements {
        match stmt {
            Statement::CreateTable(ct) => {
                let mut columns = Vec::new();
                for col_def in &ct.columns {
                    let column_type = map_data_type(&col_def.data_type)?;
                    let nullable = !col_def
                        .options
                        .iter()
                        .any(|opt| matches!(opt.option, ColumnOption::NotNull));
                    columns.push(Column {
                        name: col_def.name.value.clone(),
                        column_type,
                        nullable,
                    });
                }
                let schema = Schema { columns };
                tx.create_table(&ct.name.to_string(), schema)?;
            }
            Statement::Insert(ins) => {
                let table_name = ins.table.to_string();
                let source = ins
                    .source
                    .as_ref()
                    .ok_or(SqlError::UnsupportedStatement)?;
                let rows_exprs = match source.body.as_ref() {
                    SetExpr::Values(values) => &values.rows,
                    _ => return Err(SqlError::UnsupportedStatement),
                };

                let column_map = if ins.columns.is_empty() {
                    None
                } else {
                    let schema = tx.get_schema(&table_name)?;
                    let mut map = Vec::with_capacity(ins.columns.len());
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
                    } else {
                        Row { values }
                    };

                    tx.insert(&table_name, &row)?;
                }
            }
            _ => return Err(SqlError::UnsupportedStatement),
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Database, Pager};
    use std::fs;
    use tempfile::NamedTempFile;

    fn open_db() -> (Database, NamedTempFile) {
        let temp = NamedTempFile::new().unwrap();
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(temp.path())
            .unwrap();
        let pager = Pager::new(file, 256);
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

        // Verify by inserting a valid row with all types
        let key = tx
            .insert(
                "items",
                &Row {
                    values: vec![
                        DbValue::Integer(1),
                        DbValue::Text("widget".into()),
                        DbValue::Float(9.99),
                        DbValue::Bool(true),
                    ],
                },
            )
            .unwrap();
        let row = tx.get("items", key).unwrap().unwrap();
        assert_eq!(row.values[0], DbValue::Integer(1));
        assert_eq!(row.values[1], DbValue::Text("widget".into()));
        assert_eq!(row.values[2], DbValue::Float(9.99));
        assert_eq!(row.values[3], DbValue::Bool(true));

        // Nullable columns accept null
        tx.insert(
            "items",
            &Row {
                values: vec![
                    DbValue::Integer(2),
                    DbValue::Text("gadget".into()),
                    DbValue::Null,
                    DbValue::Null,
                ],
            },
        )
        .unwrap();

        // NOT NULL columns reject null
        let err = tx
            .insert(
                "items",
                &Row {
                    values: vec![
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
        execute(
            &tx,
            "INSERT INTO users VALUES ('Alice', 30)",
        )
        .unwrap();

        let rows = tx.scan("users", ..).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1.values[0], DbValue::Text("Alice".into()));
        assert_eq!(rows[0].1.values[1], DbValue::Integer(30));
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
        execute(
            &tx,
            "INSERT INTO users VALUES ('Alice', 30), ('Bob', 25)",
        )
        .unwrap();

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
        assert_eq!(rows[0].1.values[0], DbValue::Text("Alice".into()));
        assert_eq!(rows[0].1.values[1], DbValue::Null); // age omitted
        assert_eq!(rows[0].1.values[2], DbValue::Bool(true));
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
        execute(
            &tx,
            "INSERT INTO t VALUES (42, 3.14, 'hello', true)",
        )
        .unwrap();

        let rows = tx.scan("t", ..).unwrap();
        assert_eq!(rows[0].1.values[0], DbValue::Integer(42));
        assert_eq!(rows[0].1.values[1], DbValue::Float(3.14));
        assert_eq!(rows[0].1.values[2], DbValue::Text("hello".into()));
        assert_eq!(rows[0].1.values[3], DbValue::Bool(true));
    }

    #[test]
    fn test_insert_null() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE t (a INTEGER, b TEXT)").unwrap();
        execute(&tx, "INSERT INTO t VALUES (NULL, NULL)").unwrap();

        let rows = tx.scan("t", ..).unwrap();
        assert_eq!(rows[0].1.values[0], DbValue::Null);
        assert_eq!(rows[0].1.values[1], DbValue::Null);
    }

    #[test]
    fn test_insert_negative_numbers() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(&tx, "CREATE TABLE t (i INTEGER, f FLOAT)").unwrap();
        execute(&tx, "INSERT INTO t VALUES (-42, -3.14)").unwrap();

        let rows = tx.scan("t", ..).unwrap();
        assert_eq!(rows[0].1.values[0], DbValue::Integer(-42));
        assert_eq!(rows[0].1.values[1], DbValue::Float(-3.14));
    }

    #[test]
    fn test_insert_schema_mismatch() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        execute(
            &tx,
            "CREATE TABLE t (a INTEGER NOT NULL)",
        )
        .unwrap();
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
        let err = execute(&tx, "SELECT 1").unwrap_err();
        assert!(matches!(err, SqlError::UnsupportedStatement));
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

        // Verify types by inserting matching values
        let key = tx
            .insert(
                "t",
                &Row {
                    values: vec![
                        DbValue::Integer(42),  // BIGINT → Integer
                        DbValue::Text("hi".into()), // CHAR → Text
                        DbValue::Float(1.0),   // DOUBLE → Float
                        DbValue::Float(2.0),   // REAL → Float
                        DbValue::Bool(false),   // BOOL → Bool
                    ],
                },
            )
            .unwrap();
        let row = tx.get("t", key).unwrap().unwrap();
        assert_eq!(row.values[0], DbValue::Integer(42));
        assert_eq!(row.values[1], DbValue::Text("hi".into()));
        assert_eq!(row.values[2], DbValue::Float(1.0));
        assert_eq!(row.values[3], DbValue::Float(2.0));
        assert_eq!(row.values[4], DbValue::Bool(false));
    }
}
