use std::ops::RangeBounds;

use rkyv::rancor::Error;
use rkyv::{Archive, Deserialize, Serialize};

use crate::Pager;
use crate::transaction::{TransactionError, TransactionStore};
use crate::tree::Btree;
use crate::types::{Key, NodePtr};

// ── Column types ────────────────────────────────────────────────────

#[derive(Archive, Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    Integer,
    Text,
    Float,
    Bool,
}

#[derive(Archive, Deserialize, Serialize, Debug, Clone, PartialEq)]
pub enum DbValue {
    Integer(i64),
    Text(String),
    Float(f64),
    Bool(bool),
    Null,
}

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
pub struct Column {
    pub name: String,
    pub column_type: ColumnType,
    pub nullable: bool,
}

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
pub struct Schema {
    pub columns: Vec<Column>,
}

#[derive(Archive, Deserialize, Serialize, Debug, Clone, PartialEq)]
pub struct Row {
    pub values: Vec<DbValue>,
}

// ── Errors ──────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum DatabaseError {
    #[error("table not found: {0}")]
    TableNotFound(String),
    #[error("table already exists: {0}")]
    TableAlreadyExists(String),
    #[error("schema mismatch: {0}")]
    SchemaMismatch(String),
    #[error("transaction error: {0}")]
    Transaction(#[from] TransactionError),
    #[error("internal error: {0}")]
    Internal(#[from] Error),
}

// ── Table metadata serialisation ────────────────────────────────────

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
struct TableMeta {
    name: String,
    schema: Schema,
    root_page: NodePtr,
}

// ── Database ────────────────────────────────────────────────────────

const CATALOG_PAGE_NUM: NodePtr = 1;

pub struct Database {
    store: TransactionStore,
}

impl Database {
    pub fn create(pager: Pager) -> Result<Self, DatabaseError> {
        let mut btree = Btree::new(pager);
        btree.init()?;
        btree.init_table()?;

        Ok(Database {
            store: TransactionStore::new(btree),
        })
    }

    pub fn open(pager: Pager) -> Result<Self, DatabaseError> {
        let btree = Btree::new(pager);

        Ok(Database {
            store: TransactionStore::new(btree),
        })
    }

    fn find_table_meta(&self, name: &str) -> Result<Option<TableMeta>, DatabaseError> {
        let tx = self.store.begin_transaction();
        let entries = tx.read_range(CATALOG_PAGE_NUM, ..)?;
        for (_key, value) in &entries {
            if let Ok(archived) = rkyv::access::<rkyv::Archived<TableMeta>, Error>(value) {
                if archived.name == name {
                    let meta: TableMeta = rkyv::deserialize::<TableMeta, Error>(archived)?;
                    return Ok(Some(meta));
                }
            }
        }
        Ok(None)
    }

    pub fn create_table(&self, name: &str, schema: Schema) -> Result<(), DatabaseError> {
        if self.find_table_meta(name)?.is_some() {
            return Err(DatabaseError::TableAlreadyExists(name.to_string()));
        }

        let root_page = self.store.init_table()?;
        let meta = TableMeta {
            name: name.to_string(),
            schema,
            root_page,
        };

        // Persist to catalog via transaction
        let tx = self.store.begin_transaction();
        tx.insert(CATALOG_PAGE_NUM, rkyv::to_bytes::<Error>(&meta)?.to_vec())?;
        tx.commit()?;

        Ok(())
    }

    pub fn drop_table(&self, name: &str) -> Result<(), DatabaseError> {
        // Find catalog key for this table via transaction
        let tx = self.store.begin_transaction();
        let entries = tx.read_range(CATALOG_PAGE_NUM, ..)?;
        let found_key = entries.iter().find_map(|(key, value)| {
            let meta = rkyv::access::<rkyv::Archived<TableMeta>, Error>(value).ok()?;
            if meta.name == name { Some(*key) } else { None }
        });

        match found_key {
            Some(key) => {
                tx.remove(CATALOG_PAGE_NUM, key);
                tx.commit()?;
                Ok(())
            }
            None => Err(DatabaseError::TableNotFound(name.to_string())),
        }
    }

    pub fn begin_transaction(&self) -> DbTransaction<'_> {
        DbTransaction {
            tx: self.store.begin_transaction(),
        }
    }
}

// ── Schema validation ───────────────────────────────────────────────

fn validate_row(row: &Row, schema: &Schema) -> Result<(), DatabaseError> {
    if row.values.len() != schema.columns.len() {
        return Err(DatabaseError::SchemaMismatch(format!(
            "expected {} columns, got {}",
            schema.columns.len(),
            row.values.len()
        )));
    }
    for (i, (val, col)) in row.values.iter().zip(schema.columns.iter()).enumerate() {
        match val {
            DbValue::Null => {
                if !col.nullable {
                    return Err(DatabaseError::SchemaMismatch(format!(
                        "column '{}' (index {i}) is not nullable",
                        col.name
                    )));
                }
            }
            DbValue::Integer(_) => {
                if col.column_type != ColumnType::Integer {
                    return Err(DatabaseError::SchemaMismatch(format!(
                        "column '{}' (index {i}) expected {:?}, got Integer",
                        col.name, col.column_type
                    )));
                }
            }
            DbValue::Text(_) => {
                if col.column_type != ColumnType::Text {
                    return Err(DatabaseError::SchemaMismatch(format!(
                        "column '{}' (index {i}) expected {:?}, got Text",
                        col.name, col.column_type
                    )));
                }
            }
            DbValue::Float(_) => {
                if col.column_type != ColumnType::Float {
                    return Err(DatabaseError::SchemaMismatch(format!(
                        "column '{}' (index {i}) expected {:?}, got Float",
                        col.name, col.column_type
                    )));
                }
            }
            DbValue::Bool(_) => {
                if col.column_type != ColumnType::Bool {
                    return Err(DatabaseError::SchemaMismatch(format!(
                        "column '{}' (index {i}) expected {:?}, got Bool",
                        col.name, col.column_type
                    )));
                }
            }
        }
    }
    Ok(())
}

// ── DbTransaction ───────────────────────────────────────────────────

pub struct DbTransaction<'a> {
    tx: crate::Transaction<'a>,
}

impl<'a> DbTransaction<'a> {
    fn get_meta(&self, table_name: &str) -> Result<TableMeta, DatabaseError> {
        let entries = self.tx.read_range(CATALOG_PAGE_NUM, ..)?;
        for (_key, value) in &entries {
            if let Ok(archived) = rkyv::access::<rkyv::Archived<TableMeta>, Error>(value) {
                if archived.name == table_name {
                    return Ok(rkyv::deserialize::<TableMeta, Error>(archived)?);
                }
            }
        }
        Err(DatabaseError::TableNotFound(table_name.to_string()))
    }

    pub fn insert(&self, table_name: &str, row: &Row) -> Result<Key, DatabaseError> {
        let meta = self.get_meta(table_name)?;
        validate_row(row, &meta.schema)?;
        let key = self
            .tx
            .insert(meta.root_page, rkyv::to_bytes::<Error>(row)?.to_vec())?;
        Ok(key)
    }

    pub fn get(&self, table_name: &str, key: Key) -> Result<Option<Row>, DatabaseError> {
        let meta = self.get_meta(table_name)?;
        let data = self.tx.read(meta.root_page, key)?;
        match data {
            Some(bytes) => {
                let archived = rkyv::access::<rkyv::Archived<Row>, Error>(&bytes)?;
                Ok(Some(rkyv::deserialize::<Row, Error>(archived)?))
            }
            None => Ok(None),
        }
    }

    pub fn scan(
        &self,
        table_name: &str,
        range: impl RangeBounds<Key>,
    ) -> Result<Vec<(Key, Row)>, DatabaseError> {
        let meta = self.get_meta(table_name)?;
        let raw = self.tx.read_range(meta.root_page, range)?;
        let mut result = Vec::with_capacity(raw.len());
        for (key, bytes) in raw {
            let archived = rkyv::access::<rkyv::Archived<Row>, Error>(&bytes)?;
            result.push((key, rkyv::deserialize::<Row, Error>(archived)?));
        }
        Ok(result)
    }

    pub fn delete(&self, table_name: &str, key: Key) -> Result<(), DatabaseError> {
        let meta = self.get_meta(table_name)?;
        self.tx.read(meta.root_page, key)?;
        self.tx.remove(meta.root_page, key);
        Ok(())
    }

    pub fn update(&self, table_name: &str, key: Key, row: &Row) -> Result<(), DatabaseError> {
        let meta = self.get_meta(table_name)?;
        validate_row(row, &meta.schema)?;
        self.tx
            .write(meta.root_page, key, rkyv::to_bytes::<Error>(row)?.to_vec());
        Ok(())
    }

    pub fn commit(self) -> Result<(), DatabaseError> {
        self.tx.commit()?;
        Ok(())
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
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

    fn users_schema() -> Schema {
        Schema {
            columns: vec![
                Column {
                    name: "name".into(),
                    column_type: ColumnType::Text,
                    nullable: false,
                },
                Column {
                    name: "age".into(),
                    column_type: ColumnType::Integer,
                    nullable: false,
                },
            ],
        }
    }

    // 1. Create table + verify schema
    #[test]
    fn test_create_table() {
        let (db, _tmp) = open_db();
        db.create_table("users", users_schema()).unwrap();
        let meta = db.find_table_meta("users").unwrap().unwrap();
        assert_eq!(meta.schema.columns.len(), 2);
    }

    #[test]
    fn test_create_table_already_exists() {
        let (db, _tmp) = open_db();
        db.create_table("users", users_schema()).unwrap();
        let err = db.create_table("users", users_schema()).unwrap_err();
        assert!(matches!(err, DatabaseError::TableAlreadyExists(_)));
    }

    // 2. Insert row + get by key
    #[test]
    fn test_insert_and_get() {
        let (db, _tmp) = open_db();
        db.create_table("users", users_schema()).unwrap();

        let row = Row {
            values: vec![DbValue::Text("Alice".into()), DbValue::Integer(30)],
        };

        let tx = db.begin_transaction();
        let key = tx.insert("users", &row).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let fetched = tx.get("users", key).unwrap().unwrap();
        assert_eq!(fetched, row);
    }

    // 3. Insert with schema mismatch
    #[test]
    fn test_schema_mismatch_wrong_type() {
        let (db, _tmp) = open_db();
        db.create_table("users", users_schema()).unwrap();

        let bad_row = Row {
            values: vec![DbValue::Integer(123), DbValue::Integer(30)],
        };

        let tx = db.begin_transaction();
        let err = tx.insert("users", &bad_row).unwrap_err();
        assert!(matches!(err, DatabaseError::SchemaMismatch(_)));
    }

    #[test]
    fn test_schema_mismatch_wrong_column_count() {
        let (db, _tmp) = open_db();
        db.create_table("users", users_schema()).unwrap();

        let bad_row = Row {
            values: vec![DbValue::Text("Alice".into())],
        };

        let tx = db.begin_transaction();
        let err = tx.insert("users", &bad_row).unwrap_err();
        assert!(matches!(err, DatabaseError::SchemaMismatch(_)));
    }

    // 4. Scan range of rows
    #[test]
    fn test_scan_range() {
        let (db, _tmp) = open_db();
        db.create_table("users", users_schema()).unwrap();

        let tx = db.begin_transaction();
        let k1 = tx
            .insert(
                "users",
                &Row {
                    values: vec![DbValue::Text("Alice".into()), DbValue::Integer(30)],
                },
            )
            .unwrap();
        let k2 = tx
            .insert(
                "users",
                &Row {
                    values: vec![DbValue::Text("Bob".into()), DbValue::Integer(25)],
                },
            )
            .unwrap();
        let _k3 = tx
            .insert(
                "users",
                &Row {
                    values: vec![DbValue::Text("Charlie".into()), DbValue::Integer(35)],
                },
            )
            .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let results = tx.scan("users", k1..=k2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1.values[0], DbValue::Text("Alice".into()));
        assert_eq!(results[1].1.values[0], DbValue::Text("Bob".into()));
    }

    // 5. Update and delete rows
    #[test]
    fn test_update_row() {
        let (db, _tmp) = open_db();
        db.create_table("users", users_schema()).unwrap();

        let tx = db.begin_transaction();
        let key = tx
            .insert(
                "users",
                &Row {
                    values: vec![DbValue::Text("Alice".into()), DbValue::Integer(30)],
                },
            )
            .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        tx.update(
            "users",
            key,
            &Row {
                values: vec![DbValue::Text("Alice".into()), DbValue::Integer(31)],
            },
        )
        .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let fetched = tx.get("users", key).unwrap().unwrap();
        assert_eq!(fetched.values[1], DbValue::Integer(31));
    }

    #[test]
    fn test_delete_row() {
        let (db, _tmp) = open_db();
        db.create_table("users", users_schema()).unwrap();

        let tx = db.begin_transaction();
        let key = tx
            .insert(
                "users",
                &Row {
                    values: vec![DbValue::Text("Alice".into()), DbValue::Integer(30)],
                },
            )
            .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        tx.delete("users", key).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        assert!(tx.get("users", key).unwrap().is_none());
    }

    // 6. Multiple tables in same database
    #[test]
    fn test_multiple_tables() {
        let (db, _tmp) = open_db();
        db.create_table("users", users_schema()).unwrap();
        db.create_table(
            "products",
            Schema {
                columns: vec![
                    Column {
                        name: "title".into(),
                        column_type: ColumnType::Text,
                        nullable: false,
                    },
                    Column {
                        name: "price".into(),
                        column_type: ColumnType::Float,
                        nullable: false,
                    },
                ],
            },
        )
        .unwrap();

        let tx = db.begin_transaction();
        let uk = tx
            .insert(
                "users",
                &Row {
                    values: vec![DbValue::Text("Alice".into()), DbValue::Integer(30)],
                },
            )
            .unwrap();
        let pk = tx
            .insert(
                "products",
                &Row {
                    values: vec![DbValue::Text("Widget".into()), DbValue::Float(9.99)],
                },
            )
            .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let user = tx.get("users", uk).unwrap().unwrap();
        let product = tx.get("products", pk).unwrap().unwrap();
        assert_eq!(user.values[0], DbValue::Text("Alice".into()));
        assert_eq!(product.values[0], DbValue::Text("Widget".into()));
    }

    // 7. Transaction commit/rollback behavior
    #[test]
    fn test_rollback_on_drop() {
        let (db, _tmp) = open_db();
        db.create_table("users", users_schema()).unwrap();

        let tx = db.begin_transaction();
        let key = tx
            .insert(
                "users",
                &Row {
                    values: vec![DbValue::Text("Ghost".into()), DbValue::Integer(0)],
                },
            )
            .unwrap();
        drop(tx); // rollback

        let tx = db.begin_transaction();
        assert!(tx.get("users", key).unwrap().is_none());
    }

    #[test]
    fn test_commit_persists() {
        let (db, _tmp) = open_db();
        db.create_table("users", users_schema()).unwrap();

        let tx = db.begin_transaction();
        let key = tx
            .insert(
                "users",
                &Row {
                    values: vec![DbValue::Text("Persist".into()), DbValue::Integer(1)],
                },
            )
            .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        assert!(tx.get("users", key).unwrap().is_some());
    }

    // 8. Nullable column handling
    #[test]
    fn test_nullable_column() {
        let (db, _tmp) = open_db();
        db.create_table(
            "events",
            Schema {
                columns: vec![
                    Column {
                        name: "title".into(),
                        column_type: ColumnType::Text,
                        nullable: false,
                    },
                    Column {
                        name: "description".into(),
                        column_type: ColumnType::Text,
                        nullable: true,
                    },
                ],
            },
        )
        .unwrap();

        let tx = db.begin_transaction();
        let key = tx
            .insert(
                "events",
                &Row {
                    values: vec![DbValue::Text("Launch".into()), DbValue::Null],
                },
            )
            .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let row = tx.get("events", key).unwrap().unwrap();
        assert_eq!(row.values[1], DbValue::Null);
    }

    #[test]
    fn test_non_nullable_rejects_null() {
        let (db, _tmp) = open_db();
        db.create_table("users", users_schema()).unwrap();

        let tx = db.begin_transaction();
        let err = tx
            .insert(
                "users",
                &Row {
                    values: vec![DbValue::Null, DbValue::Integer(30)],
                },
            )
            .unwrap_err();
        assert!(matches!(err, DatabaseError::SchemaMismatch(_)));
    }

    // 9. Drop table
    #[test]
    fn test_drop_table() {
        let (db, _tmp) = open_db();
        db.create_table("users", users_schema()).unwrap();
        db.drop_table("users").unwrap();
        assert!(db.find_table_meta("users").unwrap().is_none());
    }

    #[test]
    fn test_drop_nonexistent_table() {
        let (db, _tmp) = open_db();
        let err = db.drop_table("nope").unwrap_err();
        assert!(matches!(err, DatabaseError::TableNotFound(_)));
    }

    // Extra: table not found in transaction
    #[test]
    fn test_table_not_found() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        let err = tx.get("missing", 0).unwrap_err();
        assert!(matches!(err, DatabaseError::TableNotFound(_)));
    }

    // Bool column round-trip
    #[test]
    fn test_bool_column() {
        let (db, _tmp) = open_db();
        db.create_table(
            "flags",
            Schema {
                columns: vec![Column {
                    name: "active".into(),
                    column_type: ColumnType::Bool,
                    nullable: false,
                }],
            },
        )
        .unwrap();

        let tx = db.begin_transaction();
        let k1 = tx
            .insert(
                "flags",
                &Row {
                    values: vec![DbValue::Bool(true)],
                },
            )
            .unwrap();
        let k2 = tx
            .insert(
                "flags",
                &Row {
                    values: vec![DbValue::Bool(false)],
                },
            )
            .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        assert_eq!(
            tx.get("flags", k1).unwrap().unwrap().values[0],
            DbValue::Bool(true)
        );
        assert_eq!(
            tx.get("flags", k2).unwrap().unwrap().values[0],
            DbValue::Bool(false)
        );
    }
}
