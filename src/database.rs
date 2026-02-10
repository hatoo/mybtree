use std::collections::HashMap;
use std::ops::RangeBounds;

use rkyv::rancor::Error;

use crate::Pager;
use crate::transaction::{TransactionError, TransactionStore};
use crate::tree::Btree;
use crate::types::{Key, NodePtr};

// ── Column types ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    Integer,
    Text,
    Float,
    Bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DbValue {
    Integer(i64),
    Text(String),
    Float(f64),
    Bool(bool),
    Null,
}

#[derive(Debug, Clone)]
pub struct Column {
    pub name: String,
    pub column_type: ColumnType,
    pub nullable: bool,
}

#[derive(Debug, Clone)]
pub struct Schema {
    pub columns: Vec<Column>,
}

#[derive(Debug, Clone, PartialEq)]
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

// ── Row serialisation ───────────────────────────────────────────────

const TAG_NULL: u8 = 0x00;
const TAG_INTEGER: u8 = 0x01;
const TAG_TEXT: u8 = 0x02;
const TAG_FLOAT: u8 = 0x03;
const TAG_BOOL: u8 = 0x04;

fn serialize_row(row: &Row) -> Vec<u8> {
    let mut buf = Vec::new();
    for v in &row.values {
        match v {
            DbValue::Null => buf.push(TAG_NULL),
            DbValue::Integer(i) => {
                buf.push(TAG_INTEGER);
                buf.extend_from_slice(&i.to_le_bytes());
            }
            DbValue::Text(s) => {
                buf.push(TAG_TEXT);
                let bytes = s.as_bytes();
                buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(bytes);
            }
            DbValue::Float(f) => {
                buf.push(TAG_FLOAT);
                buf.extend_from_slice(&f.to_le_bytes());
            }
            DbValue::Bool(b) => {
                buf.push(TAG_BOOL);
                buf.push(if *b { 1 } else { 0 });
            }
        }
    }
    buf
}

fn deserialize_row(data: &[u8]) -> Result<Row, DatabaseError> {
    let mut values = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let tag = data[pos];
        pos += 1;
        match tag {
            TAG_NULL => values.push(DbValue::Null),
            TAG_INTEGER => {
                let bytes: [u8; 8] = data[pos..pos + 8]
                    .try_into()
                    .map_err(|_| DatabaseError::SchemaMismatch("truncated integer".into()))?;
                values.push(DbValue::Integer(i64::from_le_bytes(bytes)));
                pos += 8;
            }
            TAG_TEXT => {
                let len_bytes: [u8; 4] = data[pos..pos + 4]
                    .try_into()
                    .map_err(|_| DatabaseError::SchemaMismatch("truncated text length".into()))?;
                let len = u32::from_le_bytes(len_bytes) as usize;
                pos += 4;
                let s = std::str::from_utf8(&data[pos..pos + len])
                    .map_err(|_| DatabaseError::SchemaMismatch("invalid utf-8".into()))?;
                values.push(DbValue::Text(s.to_string()));
                pos += len;
            }
            TAG_FLOAT => {
                let bytes: [u8; 8] = data[pos..pos + 8]
                    .try_into()
                    .map_err(|_| DatabaseError::SchemaMismatch("truncated float".into()))?;
                values.push(DbValue::Float(f64::from_le_bytes(bytes)));
                pos += 8;
            }
            TAG_BOOL => {
                values.push(DbValue::Bool(data[pos] != 0));
                pos += 1;
            }
            _ => {
                return Err(DatabaseError::SchemaMismatch(format!(
                    "unknown type tag: {tag:#x}"
                )));
            }
        }
    }
    Ok(Row { values })
}

// ── Table metadata serialisation ────────────────────────────────────

#[derive(Debug, Clone)]
struct TableMeta {
    name: String,
    schema: Schema,
    root_page: NodePtr,
}

fn serialize_table_meta(meta: &TableMeta) -> Vec<u8> {
    let mut buf = Vec::new();
    // table name
    let name_bytes = meta.name.as_bytes();
    buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(name_bytes);
    // column count
    buf.extend_from_slice(&(meta.schema.columns.len() as u32).to_le_bytes());
    for col in &meta.schema.columns {
        let col_name = col.name.as_bytes();
        buf.extend_from_slice(&(col_name.len() as u32).to_le_bytes());
        buf.extend_from_slice(col_name);
        buf.push(match col.column_type {
            ColumnType::Integer => 0,
            ColumnType::Text => 1,
            ColumnType::Float => 2,
            ColumnType::Bool => 3,
        });
        buf.push(if col.nullable { 1 } else { 0 });
    }
    // root page
    buf.extend_from_slice(&meta.root_page.to_le_bytes());
    buf
}

fn deserialize_table_meta(data: &[u8]) -> Result<TableMeta, DatabaseError> {
    let mut pos = 0;

    let read_u32 = |pos: &mut usize, data: &[u8]| -> Result<u32, DatabaseError> {
        let bytes: [u8; 4] = data[*pos..*pos + 4]
            .try_into()
            .map_err(|_| DatabaseError::SchemaMismatch("truncated u32".into()))?;
        *pos += 4;
        Ok(u32::from_le_bytes(bytes))
    };

    // table name
    let name_len = read_u32(&mut pos, data)? as usize;
    let name = std::str::from_utf8(&data[pos..pos + name_len])
        .map_err(|_| DatabaseError::SchemaMismatch("invalid table name".into()))?
        .to_string();
    pos += name_len;

    // columns
    let col_count = read_u32(&mut pos, data)? as usize;
    let mut columns = Vec::with_capacity(col_count);
    for _ in 0..col_count {
        let col_name_len = read_u32(&mut pos, data)? as usize;
        let col_name = std::str::from_utf8(&data[pos..pos + col_name_len])
            .map_err(|_| DatabaseError::SchemaMismatch("invalid column name".into()))?
            .to_string();
        pos += col_name_len;

        let type_byte = data[pos];
        pos += 1;
        let column_type = match type_byte {
            0 => ColumnType::Integer,
            1 => ColumnType::Text,
            2 => ColumnType::Float,
            3 => ColumnType::Bool,
            _ => {
                return Err(DatabaseError::SchemaMismatch(format!(
                    "unknown column type: {type_byte}"
                )));
            }
        };

        let nullable = data[pos] != 0;
        pos += 1;

        columns.push(Column {
            name: col_name,
            column_type,
            nullable,
        });
    }

    // root page
    let root_bytes: [u8; 8] = data[pos..pos + 8]
        .try_into()
        .map_err(|_| DatabaseError::SchemaMismatch("truncated root page".into()))?;
    let root_page = u64::from_le_bytes(root_bytes);

    Ok(TableMeta {
        name,
        schema: Schema { columns },
        root_page,
    })
}

// ── Database ────────────────────────────────────────────────────────

pub struct Database {
    store: TransactionStore,
    catalog_root: NodePtr,
    data_root: NodePtr,
    tables: HashMap<String, TableMeta>,
    next_table_id: u64,
}

impl Database {
    pub fn open(pager: Pager) -> Result<Self, DatabaseError> {
        let mut btree = Btree::new(pager);
        btree.init()?;
        let catalog_root = btree.init_table()?;
        let data_root = btree.init_table()?;

        Ok(Database {
            store: TransactionStore::new(btree),
            catalog_root,
            data_root,
            tables: HashMap::new(),
            next_table_id: 0,
        })
    }

    pub fn create_table(&mut self, name: &str, schema: Schema) -> Result<(), DatabaseError> {
        if self.tables.contains_key(name) {
            return Err(DatabaseError::TableAlreadyExists(name.to_string()));
        }

        let meta = TableMeta {
            name: name.to_string(),
            schema,
            root_page: self.data_root,
        };

        let id = self.next_table_id;
        self.next_table_id += 1;

        // Persist to catalog via transaction
        let serialized = serialize_table_meta(&meta);
        let tx = self.store.begin_transaction();
        tx.write(id, serialized);
        tx.commit(self.catalog_root)?;

        self.tables.insert(name.to_string(), meta);
        Ok(())
    }

    pub fn drop_table(&mut self, name: &str) -> Result<(), DatabaseError> {
        if !self.tables.contains_key(name) {
            return Err(DatabaseError::TableNotFound(name.to_string()));
        }

        // Find catalog key for this table via transaction
        let tx = self.store.begin_transaction();
        let entries = tx.read_range(self.catalog_root, ..)?;
        let found_key = entries.iter().find_map(|(key, value)| {
            if let Ok(meta) = deserialize_table_meta(value)
                && meta.name == name
            {
                Some(*key)
            } else {
                None
            }
        });

        if let Some(key) = found_key {
            tx.remove(key);
        }
        tx.commit(self.catalog_root)?;

        self.tables.remove(name);
        Ok(())
    }

    pub fn begin_transaction(&self) -> DbTransaction<'_> {
        DbTransaction {
            tx: self.store.begin_transaction(),
            tables: &self.tables,
            data_root: self.data_root,
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
    tables: &'a HashMap<String, TableMeta>,
    data_root: NodePtr,
}

impl<'a> DbTransaction<'a> {
    fn get_meta(&self, table_name: &str) -> Result<&TableMeta, DatabaseError> {
        self.tables
            .get(table_name)
            .ok_or_else(|| DatabaseError::TableNotFound(table_name.to_string()))
    }

    pub fn insert(&self, table_name: &str, row: &Row) -> Result<Key, DatabaseError> {
        let meta = self.get_meta(table_name)?;
        validate_row(row, &meta.schema)?;
        let bytes = serialize_row(row);
        let key = self.tx.insert(self.data_root, bytes)?;
        Ok(key)
    }

    pub fn get(&self, table_name: &str, key: Key) -> Result<Option<Row>, DatabaseError> {
        self.get_meta(table_name)?;
        let data = self.tx.read(self.data_root, key)?;
        match data {
            Some(bytes) => Ok(Some(deserialize_row(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn scan(
        &self,
        table_name: &str,
        range: impl RangeBounds<Key>,
    ) -> Result<Vec<(Key, Row)>, DatabaseError> {
        self.get_meta(table_name)?;
        let raw = self.tx.read_range(self.data_root, range)?;
        let mut result = Vec::with_capacity(raw.len());
        for (key, bytes) in raw {
            result.push((key, deserialize_row(&bytes)?));
        }
        Ok(result)
    }

    pub fn delete(&self, table_name: &str, key: Key) -> Result<(), DatabaseError> {
        self.get_meta(table_name)?;
        self.tx.read(self.data_root, key)?;
        self.tx.remove(key);
        Ok(())
    }

    pub fn update(&self, table_name: &str, key: Key, row: &Row) -> Result<(), DatabaseError> {
        let meta = self.get_meta(table_name)?;
        validate_row(row, &meta.schema)?;
        let bytes = serialize_row(row);
        self.tx.write(key, bytes);
        Ok(())
    }

    pub fn commit(self) -> Result<(), DatabaseError> {
        self.tx.commit(self.data_root)?;
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
        let db = Database::open(pager).unwrap();
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
        let (mut db, _tmp) = open_db();
        db.create_table("users", users_schema()).unwrap();
        assert!(db.tables.contains_key("users"));
        assert_eq!(db.tables["users"].schema.columns.len(), 2);
    }

    #[test]
    fn test_create_table_already_exists() {
        let (mut db, _tmp) = open_db();
        db.create_table("users", users_schema()).unwrap();
        let err = db.create_table("users", users_schema()).unwrap_err();
        assert!(matches!(err, DatabaseError::TableAlreadyExists(_)));
    }

    // 2. Insert row + get by key
    #[test]
    fn test_insert_and_get() {
        let (mut db, _tmp) = open_db();
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
        let (mut db, _tmp) = open_db();
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
        let (mut db, _tmp) = open_db();
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
        let (mut db, _tmp) = open_db();
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
        let (mut db, _tmp) = open_db();
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
        let (mut db, _tmp) = open_db();
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
        let (mut db, _tmp) = open_db();
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
        let (mut db, _tmp) = open_db();
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
        let (mut db, _tmp) = open_db();
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
        let (mut db, _tmp) = open_db();
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
        let (mut db, _tmp) = open_db();
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
        let (mut db, _tmp) = open_db();
        db.create_table("users", users_schema()).unwrap();
        db.drop_table("users").unwrap();
        assert!(!db.tables.contains_key("users"));
    }

    #[test]
    fn test_drop_nonexistent_table() {
        let (mut db, _tmp) = open_db();
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
        let (mut db, _tmp) = open_db();
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
