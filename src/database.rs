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
    /// Index of the primary key column. `None` means no explicit PK was given
    /// (an implicit `_rowid` column will be prepended by `create_table`).
    pub primary_key: Option<usize>,
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
    #[error("duplicate primary key: {0}")]
    DuplicateKey(Key),
    #[error("transaction error: {0}")]
    Transaction(#[from] TransactionError),
    #[error("internal error: {0}")]
    Internal(#[from] Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("tree error: {0}")]
    TreeError(#[from] crate::tree::TreeError),
}

// ── Table metadata serialisation ────────────────────────────────────

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
struct TableMeta {
    name: String,
    schema: Schema,
    root_page: NodePtr,
    /// Index trees: `(column_name, index_root_page)` pairs.
    index_trees: Vec<(String, NodePtr)>,
}

// ── Database ────────────────────────────────────────────────────────

const CATALOG_PAGE_NUM: NodePtr = 1;
/// Index tree on the catalog: table name bytes → catalog key.
const CATALOG_INDEX_PAGE_NUM: NodePtr = 2;

pub struct Database<const N: usize> {
    store: TransactionStore<N>,
}

impl<const N: usize> Database<N> {
    pub fn create(pager: Pager<N>) -> Result<Self, DatabaseError> {
        let mut btree = Btree::new(pager);
        btree.init()?;
        btree.init_tree()?; // page 1 = catalog
        btree.init_index()?; // page 2 = catalog name index

        Ok(Database {
            store: TransactionStore::new(btree),
        })
    }

    pub fn open(pager: Pager<N>) -> Result<Self, DatabaseError> {
        let btree = Btree::new(pager);

        Ok(Database {
            store: TransactionStore::new(btree),
        })
    }

    pub fn begin_transaction(&self) -> DbTransaction<'_, N> {
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

/// Convert a `DbValue` into a byte representation suitable for index tree keys.
pub fn db_value_to_bytes(value: &DbValue) -> Vec<u8> {
    match value {
        DbValue::Integer(i) => i.to_be_bytes().to_vec(),
        DbValue::Text(s) => s.as_bytes().to_vec(),
        DbValue::Float(f) => f.to_be_bytes().to_vec(),
        DbValue::Bool(b) => vec![*b as u8],
        DbValue::Null => vec![],
    }
}

// ── DbTransaction ───────────────────────────────────────────────────

pub struct DbTransaction<'a, const N: usize> {
    tx: crate::Transaction<'a, N>,
}

impl<'a, const N: usize> DbTransaction<'a, N> {
    fn find_table_meta(&self, name: &str) -> Result<Option<TableMeta>, DatabaseError> {
        if let Some(catalog_key) = self
            .tx
            .index_read(CATALOG_INDEX_PAGE_NUM, name.as_bytes())?
        {
            if let Some(data) = self.tx.read(CATALOG_PAGE_NUM, catalog_key)? {
                let archived = rkyv::access::<rkyv::Archived<TableMeta>, Error>(&data)?;
                return Ok(Some(rkyv::deserialize::<TableMeta, Error>(archived)?));
            }
        }
        Ok(None)
    }

    pub fn get_schema(&self, name: &str) -> Result<Schema, DatabaseError> {
        let meta = self
            .find_table_meta(name)?
            .ok_or_else(|| DatabaseError::TableNotFound(name.to_string()))?;
        Ok(meta.schema)
    }

    pub fn create_table(&self, name: &str, mut schema: Schema) -> Result<(), DatabaseError> {
        if self.find_table_meta(name)?.is_some() {
            return Err(DatabaseError::TableAlreadyExists(name.to_string()));
        }

        // If no explicit primary key, prepend an implicit `_rowid` column.
        if schema.primary_key.is_none() {
            schema.columns.insert(
                0,
                Column {
                    name: "_rowid".to_string(),
                    column_type: ColumnType::Integer,
                    nullable: false,
                },
            );
            schema.primary_key = Some(0);
        }

        // Validate PK column is integer and not nullable.
        let pk_idx = schema.primary_key.unwrap();
        let pk_col = &schema.columns[pk_idx];
        if pk_col.column_type != ColumnType::Integer {
            return Err(DatabaseError::SchemaMismatch(
                "primary key column must be INTEGER".to_string(),
            ));
        }
        if pk_col.nullable {
            return Err(DatabaseError::SchemaMismatch(
                "primary key column must be NOT NULL".to_string(),
            ));
        }

        let root_page = self.tx.init_tree()?;
        let meta = TableMeta {
            name: name.to_string(),
            schema,
            root_page,
            index_trees: vec![],
        };

        let catalog_key = self
            .tx
            .insert(CATALOG_PAGE_NUM, rkyv::to_bytes::<Error>(&meta)?.to_vec())?;
        self.tx.index_insert(
            CATALOG_INDEX_PAGE_NUM,
            catalog_key,
            name.as_bytes().to_vec(),
        )?;

        Ok(())
    }

    pub fn drop_table(&self, name: &str) -> Result<(), DatabaseError> {
        let catalog_key = self
            .tx
            .index_read(CATALOG_INDEX_PAGE_NUM, name.as_bytes())?
            .ok_or_else(|| DatabaseError::TableNotFound(name.to_string()))?;

        let data = self
            .tx
            .read(CATALOG_PAGE_NUM, catalog_key)?
            .ok_or_else(|| DatabaseError::TableNotFound(name.to_string()))?;
        let archived = rkyv::access::<rkyv::Archived<TableMeta>, Error>(&data)?;
        let meta: TableMeta = rkyv::deserialize::<TableMeta, Error>(archived)?;

        self.tx.remove(CATALOG_PAGE_NUM, catalog_key);
        self.tx
            .index_remove(CATALOG_INDEX_PAGE_NUM, name.as_bytes(), catalog_key)?;

        // Free all pages belonging to the table's tree
        self.tx.free_tree(meta.root_page)?;
        // Free all index trees
        for (_, idx_root) in &meta.index_trees {
            self.tx.free_index_tree(*idx_root)?;
        }
        Ok(())
    }

    pub fn create_index(&self, table_name: &str, column_name: &str) -> Result<(), DatabaseError> {
        let mut meta = self
            .find_table_meta(table_name)?
            .ok_or_else(|| DatabaseError::TableNotFound(table_name.to_string()))?;

        // Validate column exists
        let col_idx = meta
            .schema
            .columns
            .iter()
            .position(|c| c.name == column_name)
            .ok_or_else(|| {
                DatabaseError::SchemaMismatch(format!("column '{}' not found", column_name))
            })?;

        // Check not already indexed
        if meta.index_trees.iter().any(|(c, _)| c == column_name) {
            return Err(DatabaseError::SchemaMismatch(format!(
                "index already exists on column '{}'",
                column_name
            )));
        }

        let idx_root = self.tx.init_index()?;

        // Back-fill: scan existing rows and insert into index tree
        let rows = self.tx.read_range(meta.root_page, ..)?;
        for (row_key, row_bytes) in &rows {
            if let Ok(archived) = rkyv::access::<rkyv::Archived<Row>, Error>(row_bytes) {
                let row: Row = rkyv::deserialize::<Row, Error>(archived)?;
                let col_bytes = db_value_to_bytes(&row.values[col_idx]);
                self.tx.index_insert(idx_root, *row_key, col_bytes)?;
            }
        }

        // Update catalog
        meta.index_trees.push((column_name.to_string(), idx_root));
        self.update_table_meta(table_name, &meta)?;

        Ok(())
    }

    pub fn drop_index(&self, table_name: &str, column_name: &str) -> Result<(), DatabaseError> {
        let mut meta = self
            .find_table_meta(table_name)?
            .ok_or_else(|| DatabaseError::TableNotFound(table_name.to_string()))?;

        let idx_pos = meta
            .index_trees
            .iter()
            .position(|(c, _)| c == column_name)
            .ok_or_else(|| {
                DatabaseError::SchemaMismatch(format!("no index on column '{}'", column_name))
            })?;

        let (_, idx_root) = meta.index_trees.remove(idx_pos);
        self.tx.free_index_tree(idx_root)?;
        self.update_table_meta(table_name, &meta)?;

        Ok(())
    }

    /// Rewrite the catalog entry for `table_name` with the given `meta`.
    fn update_table_meta(&self, table_name: &str, meta: &TableMeta) -> Result<(), DatabaseError> {
        let catalog_key = self
            .tx
            .index_read(CATALOG_INDEX_PAGE_NUM, table_name.as_bytes())?
            .ok_or_else(|| DatabaseError::TableNotFound(table_name.to_string()))?;

        self.tx.write(
            CATALOG_PAGE_NUM,
            catalog_key,
            rkyv::to_bytes::<Error>(meta)?.to_vec(),
        );
        Ok(())
    }

    pub fn insert(&self, table_name: &str, row: &Row) -> Result<Key, DatabaseError> {
        let meta = self
            .find_table_meta(table_name)?
            .ok_or_else(|| DatabaseError::TableNotFound(table_name.to_string()))?;

        let pk_idx = meta.schema.primary_key.unwrap();
        let mut row = row.clone();

        // Handle auto-assign for implicit _rowid (Null in PK position).
        if row.values[pk_idx] == DbValue::Null {
            let next = self.tx.available_key(meta.root_page)?;
            row.values[pk_idx] = DbValue::Integer(next as i64);
        }

        validate_row(&row, &meta.schema)?;

        // Extract PK value and convert to B-tree key.
        let key = match &row.values[pk_idx] {
            DbValue::Integer(i) => {
                if *i < 0 {
                    return Err(DatabaseError::SchemaMismatch(
                        "primary key value must be non-negative".to_string(),
                    ));
                }
                *i as Key
            }
            _ => unreachable!("PK column validated as integer"),
        };

        // Check for duplicate key.
        if self.tx.read(meta.root_page, key)?.is_some() {
            return Err(DatabaseError::DuplicateKey(key));
        }

        self.tx
            .write(meta.root_page, key, rkyv::to_bytes::<Error>(&row)?.to_vec());

        // Update index trees
        for (col_name, idx_root) in &meta.index_trees {
            let col_idx = meta
                .schema
                .columns
                .iter()
                .position(|c| c.name == *col_name)
                .unwrap();
            let col_bytes = db_value_to_bytes(&row.values[col_idx]);
            self.tx.index_insert(*idx_root, key, col_bytes)?;
        }

        Ok(key)
    }

    pub fn get(&self, table_name: &str, key: Key) -> Result<Option<Row>, DatabaseError> {
        let meta = self
            .find_table_meta(table_name)?
            .ok_or_else(|| DatabaseError::TableNotFound(table_name.to_string()))?;
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
        let meta = self
            .find_table_meta(table_name)?
            .ok_or_else(|| DatabaseError::TableNotFound(table_name.to_string()))?;
        let raw = self.tx.read_range(meta.root_page, range)?;
        let mut result = Vec::with_capacity(raw.len());
        for (key, bytes) in raw {
            let archived = rkyv::access::<rkyv::Archived<Row>, Error>(&bytes)?;
            result.push((key, rkyv::deserialize::<Row, Error>(archived)?));
        }
        Ok(result)
    }

    pub fn delete(&self, table_name: &str, key: Key) -> Result<(), DatabaseError> {
        let meta = self
            .find_table_meta(table_name)?
            .ok_or_else(|| DatabaseError::TableNotFound(table_name.to_string()))?;
        let old_data = self.tx.read(meta.root_page, key)?;

        // Remove from index trees
        if let Some(old_bytes) = &old_data {
            let old_row: Row = {
                let archived = rkyv::access::<rkyv::Archived<Row>, Error>(old_bytes)?;
                rkyv::deserialize::<Row, Error>(archived)?
            };
            for (col_name, idx_root) in &meta.index_trees {
                let col_idx = meta
                    .schema
                    .columns
                    .iter()
                    .position(|c| c.name == *col_name)
                    .unwrap();
                let col_bytes = db_value_to_bytes(&old_row.values[col_idx]);
                self.tx.index_remove(*idx_root, &col_bytes, key)?;
            }
        }

        self.tx.remove(meta.root_page, key);
        Ok(())
    }

    pub fn update(&self, table_name: &str, key: Key, row: &Row) -> Result<(), DatabaseError> {
        let meta = self
            .find_table_meta(table_name)?
            .ok_or_else(|| DatabaseError::TableNotFound(table_name.to_string()))?;
        validate_row(row, &meta.schema)?;

        // Remove old values from index trees
        if !meta.index_trees.is_empty() {
            let old_data = self.tx.read(meta.root_page, key)?;
            if let Some(old_bytes) = &old_data {
                let old_row: Row = {
                    let archived = rkyv::access::<rkyv::Archived<Row>, Error>(old_bytes)?;
                    rkyv::deserialize::<Row, Error>(archived)?
                };
                for (col_name, idx_root) in &meta.index_trees {
                    let col_idx = meta
                        .schema
                        .columns
                        .iter()
                        .position(|c| c.name == *col_name)
                        .unwrap();
                    let col_bytes = db_value_to_bytes(&old_row.values[col_idx]);
                    self.tx.index_remove(*idx_root, &col_bytes, key)?;
                }
            }
        }

        self.tx
            .write(meta.root_page, key, rkyv::to_bytes::<Error>(row)?.to_vec());

        // Insert new values into index trees
        for (col_name, idx_root) in &meta.index_trees {
            let col_idx = meta
                .schema
                .columns
                .iter()
                .position(|c| c.name == *col_name)
                .unwrap();
            let col_bytes = db_value_to_bytes(&row.values[col_idx]);
            self.tx.index_insert(*idx_root, key, col_bytes)?;
        }

        Ok(())
    }

    /// Scan rows by an indexed column value range.
    /// Returns `(key, row)` pairs for rows whose indexed column value falls within `range`.
    pub fn scan_by_index<'b>(
        &self,
        table_name: &str,
        column_name: &str,
        range: impl RangeBounds<&'b [u8]>,
    ) -> Result<Vec<(Key, Row)>, DatabaseError> {
        let meta = self
            .find_table_meta(table_name)?
            .ok_or_else(|| DatabaseError::TableNotFound(table_name.to_string()))?;
        let idx_root = meta
            .index_trees
            .iter()
            .find(|(c, _)| c == column_name)
            .map(|(_, r)| *r)
            .ok_or_else(|| {
                DatabaseError::SchemaMismatch(format!("no index on column '{}'", column_name))
            })?;

        let keys = self.tx.index_read_range(idx_root, range)?;
        let mut result = Vec::new();
        for key in keys {
            if let Some(bytes) = self.tx.read(meta.root_page, key)? {
                let archived = rkyv::access::<rkyv::Archived<Row>, Error>(&bytes)?;
                result.push((key, rkyv::deserialize::<Row, Error>(archived)?));
            }
        }
        Ok(result)
    }

    pub fn get_indexed_columns(&self, name: &str) -> Result<Vec<String>, DatabaseError> {
        let meta = self
            .find_table_meta(name)?
            .ok_or_else(|| DatabaseError::TableNotFound(name.to_string()))?;
        Ok(meta.index_trees.iter().map(|(c, _)| c.clone()).collect())
    }

    pub fn list_tables(&self) -> Result<Vec<String>, DatabaseError> {
        let entries = self.tx.read_range(CATALOG_PAGE_NUM, ..)?;
        let mut names = Vec::new();
        for (_, bytes) in &entries {
            let archived = rkyv::access::<rkyv::Archived<TableMeta>, Error>(bytes)?;
            names.push(archived.name.to_string());
        }
        Ok(names)
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
            primary_key: None,
        }
    }

    /// Build a Row for a table with implicit `_rowid` (Null triggers auto-assign).
    fn user_row(name: &str, age: i64) -> Row {
        Row {
            values: vec![
                DbValue::Null, // _rowid auto-assign
                DbValue::Text(name.into()),
                DbValue::Integer(age),
            ],
        }
    }

    // 1. Create table + verify schema
    #[test]
    fn test_create_table() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        let meta = tx.find_table_meta("users").unwrap().unwrap();
        // 3 columns: _rowid (implicit) + name + age
        assert_eq!(meta.schema.columns.len(), 3);
        assert_eq!(meta.schema.columns[0].name, "_rowid");
        assert_eq!(meta.schema.primary_key, Some(0));
    }

    #[test]
    fn test_create_table_already_exists() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let err = tx.create_table("users", users_schema()).unwrap_err();
        assert!(matches!(err, DatabaseError::TableAlreadyExists(_)));
    }

    // 2. Insert row + get by key
    #[test]
    fn test_insert_and_get() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let key = tx.insert("users", &user_row("Alice", 30)).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let fetched = tx.get("users", key).unwrap().unwrap();
        assert_eq!(fetched.values[1], DbValue::Text("Alice".into()));
        assert_eq!(fetched.values[2], DbValue::Integer(30));
    }

    // 3. Insert with schema mismatch
    #[test]
    fn test_schema_mismatch_wrong_type() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        // _rowid + wrong type for name + age
        let bad_row = Row {
            values: vec![DbValue::Null, DbValue::Integer(123), DbValue::Integer(30)],
        };

        let tx = db.begin_transaction();
        let err = tx.insert("users", &bad_row).unwrap_err();
        assert!(matches!(err, DatabaseError::SchemaMismatch(_)));
    }

    #[test]
    fn test_schema_mismatch_wrong_column_count() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

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
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let k1 = tx.insert("users", &user_row("Alice", 30)).unwrap();
        let k2 = tx.insert("users", &user_row("Bob", 25)).unwrap();
        let _k3 = tx.insert("users", &user_row("Charlie", 35)).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let results = tx.scan("users", k1..=k2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1.values[1], DbValue::Text("Alice".into()));
        assert_eq!(results[1].1.values[1], DbValue::Text("Bob".into()));
    }

    // 5. Update and delete rows
    #[test]
    fn test_update_row() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let key = tx.insert("users", &user_row("Alice", 30)).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        tx.update(
            "users",
            key,
            &Row {
                values: vec![
                    DbValue::Integer(key as i64),
                    DbValue::Text("Alice".into()),
                    DbValue::Integer(31),
                ],
            },
        )
        .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let fetched = tx.get("users", key).unwrap().unwrap();
        assert_eq!(fetched.values[2], DbValue::Integer(31));
    }

    #[test]
    fn test_delete_row() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let key = tx.insert("users", &user_row("Alice", 30)).unwrap();
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
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.create_table(
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
                primary_key: None,
            },
        )
        .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let uk = tx.insert("users", &user_row("Alice", 30)).unwrap();
        let pk = tx
            .insert(
                "products",
                &Row {
                    values: vec![
                        DbValue::Null, // _rowid
                        DbValue::Text("Widget".into()),
                        DbValue::Float(9.99),
                    ],
                },
            )
            .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let user = tx.get("users", uk).unwrap().unwrap();
        let product = tx.get("products", pk).unwrap().unwrap();
        assert_eq!(user.values[1], DbValue::Text("Alice".into()));
        assert_eq!(product.values[1], DbValue::Text("Widget".into()));
    }

    // 7. Transaction commit/rollback behavior
    #[test]
    fn test_rollback_on_drop() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let key = tx.insert("users", &user_row("Ghost", 0)).unwrap();
        drop(tx); // rollback

        let tx = db.begin_transaction();
        assert!(tx.get("users", key).unwrap().is_none());
    }

    #[test]
    fn test_commit_persists() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let key = tx.insert("users", &user_row("Persist", 1)).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        assert!(tx.get("users", key).unwrap().is_some());
    }

    // 8. Nullable column handling
    #[test]
    fn test_nullable_column() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table(
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
                primary_key: None,
            },
        )
        .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let key = tx
            .insert(
                "events",
                &Row {
                    values: vec![
                        DbValue::Null, // _rowid
                        DbValue::Text("Launch".into()),
                        DbValue::Null,
                    ],
                },
            )
            .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let row = tx.get("events", key).unwrap().unwrap();
        assert_eq!(row.values[2], DbValue::Null);
    }

    #[test]
    fn test_non_nullable_rejects_null() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        // name (index 1) is NOT NULL — should reject
        let err = tx
            .insert(
                "users",
                &Row {
                    values: vec![DbValue::Null, DbValue::Null, DbValue::Integer(30)],
                },
            )
            .unwrap_err();
        assert!(matches!(err, DatabaseError::SchemaMismatch(_)));
    }

    // 9. Drop table
    #[test]
    fn test_drop_table() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        tx.drop_table("users").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        assert!(tx.find_table_meta("users").unwrap().is_none());
    }

    #[test]
    fn test_drop_nonexistent_table() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        let err = tx.drop_table("nope").unwrap_err();
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
        let tx = db.begin_transaction();
        tx.create_table(
            "flags",
            Schema {
                columns: vec![Column {
                    name: "active".into(),
                    column_type: ColumnType::Bool,
                    nullable: false,
                }],
                primary_key: None,
            },
        )
        .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let k1 = tx
            .insert(
                "flags",
                &Row {
                    values: vec![DbValue::Null, DbValue::Bool(true)],
                },
            )
            .unwrap();
        let k2 = tx
            .insert(
                "flags",
                &Row {
                    values: vec![DbValue::Null, DbValue::Bool(false)],
                },
            )
            .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        assert_eq!(
            tx.get("flags", k1).unwrap().unwrap().values[1],
            DbValue::Bool(true)
        );
        assert_eq!(
            tx.get("flags", k2).unwrap().unwrap().values[1],
            DbValue::Bool(false)
        );
    }

    #[test]
    fn test_drop_table_frees_pages() {
        let temp = NamedTempFile::new().unwrap();
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(temp.path())
            .unwrap();
        let pager = Pager::<4096>::new(file);
        let db = Database::create(pager).unwrap();

        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        // Insert many rows to allocate several pages
        let tx = db.begin_transaction();
        for i in 0..100 {
            tx.insert("users", &user_row(&format!("user_{}", i), i as i64))
                .unwrap();
        }
        tx.commit().unwrap();

        let pages_before_drop = db.store.get_total_page_count();

        let tx = db.begin_transaction();
        tx.drop_table("users").unwrap();
        tx.commit().unwrap();

        // Re-create and re-insert — should reuse freed pages
        let tx = db.begin_transaction();
        tx.create_table("users2", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        for i in 0..100 {
            tx.insert("users2", &user_row(&format!("user_{}", i), i as i64))
                .unwrap();
        }
        tx.commit().unwrap();

        let pages_after_reinsert = db.store.get_total_page_count();

        assert!(
            pages_after_reinsert <= pages_before_drop + 1,
            "Pages were not reused after drop_table: before_drop={}, after_reinsert={}",
            pages_before_drop,
            pages_after_reinsert,
        );
    }

    #[test]
    fn test_drop_table_with_index_frees_pages() {
        let temp = NamedTempFile::new().unwrap();
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(temp.path())
            .unwrap();
        let pager = Pager::<4096>::new(file);
        let db = Database::create(pager).unwrap();

        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.create_index("users", "name").unwrap();
        tx.create_index("users", "age").unwrap();
        tx.commit().unwrap();

        // Insert many rows to allocate pages for data + both indexes
        let tx = db.begin_transaction();
        for i in 0..100 {
            tx.insert("users", &user_row(&format!("user_{}", i), i as i64))
                .unwrap();
        }
        tx.commit().unwrap();

        let pages_before_drop = db.store.get_total_page_count();

        let tx = db.begin_transaction();
        tx.drop_table("users").unwrap();
        tx.commit().unwrap();

        // Re-create with indexes and re-insert — should reuse freed pages
        let tx = db.begin_transaction();
        tx.create_table("users2", users_schema()).unwrap();
        tx.create_index("users2", "name").unwrap();
        tx.create_index("users2", "age").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        for i in 0..100 {
            tx.insert("users2", &user_row(&format!("user_{}", i), i as i64))
                .unwrap();
        }
        tx.commit().unwrap();

        let pages_after_reinsert = db.store.get_total_page_count();

        assert!(
            pages_after_reinsert <= pages_before_drop + 1,
            "Pages were not reused after drop_table with indexes: before_drop={}, after_reinsert={}",
            pages_before_drop,
            pages_after_reinsert,
        );
    }

    // ── Index tests ─────────────────────────────────────────────────

    #[test]
    fn test_create_index() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.create_index("users", "name").unwrap();

        let meta = tx.find_table_meta("users").unwrap().unwrap();
        assert_eq!(meta.index_trees.len(), 1);
        assert_eq!(meta.index_trees[0].0, "name");
    }

    #[test]
    fn test_create_index_nonexistent_column() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        let err = tx.create_index("users", "email").unwrap_err();
        assert!(matches!(err, DatabaseError::SchemaMismatch(_)));
    }

    #[test]
    fn test_create_index_duplicate() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.create_index("users", "name").unwrap();
        let err = tx.create_index("users", "name").unwrap_err();
        assert!(matches!(err, DatabaseError::SchemaMismatch(_)));
    }

    #[test]
    fn test_drop_index() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.create_index("users", "name").unwrap();
        tx.drop_index("users", "name").unwrap();

        let meta = tx.find_table_meta("users").unwrap().unwrap();
        assert!(meta.index_trees.is_empty());
    }

    #[test]
    fn test_drop_index_nonexistent() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        let err = tx.drop_index("users", "name").unwrap_err();
        assert!(matches!(err, DatabaseError::SchemaMismatch(_)));
    }

    #[test]
    fn test_scan_by_index() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.create_index("users", "name").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        tx.insert("users", &user_row("Alice", 30)).unwrap();
        tx.insert("users", &user_row("Bob", 25)).unwrap();
        tx.insert("users", &user_row("Charlie", 35)).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let results = tx
            .scan_by_index("users", "name", &b"Bob"[..]..=&b"Charlie"[..])
            .unwrap();
        assert_eq!(results.len(), 2);
        let names: Vec<_> = results.iter().map(|(_, r)| &r.values[1]).collect();
        assert!(names.contains(&&DbValue::Text("Bob".into())));
        assert!(names.contains(&&DbValue::Text("Charlie".into())));
    }

    #[test]
    fn test_scan_by_index_no_index() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let err = tx
            .scan_by_index("users", "name", &b"A"[..]..&b"Z"[..])
            .unwrap_err();
        assert!(matches!(err, DatabaseError::SchemaMismatch(_)));
    }

    #[test]
    fn test_index_maintained_on_insert() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.create_index("users", "name").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        tx.insert("users", &user_row("Diana", 28)).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let results = tx
            .scan_by_index("users", "name", &b"Diana"[..]..=&b"Diana"[..])
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1.values[1], DbValue::Text("Diana".into()));
    }

    #[test]
    fn test_index_maintained_on_delete() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.create_index("users", "name").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let key = tx.insert("users", &user_row("Eve", 22)).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        tx.delete("users", key).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let results = tx
            .scan_by_index("users", "name", &b"Eve"[..]..=&b"Eve"[..])
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_index_maintained_on_update() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.create_index("users", "name").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let key = tx.insert("users", &user_row("Frank", 40)).unwrap();
        tx.commit().unwrap();

        // Update name from Frank to George
        let tx = db.begin_transaction();
        tx.update(
            "users",
            key,
            &Row {
                values: vec![
                    DbValue::Integer(key as i64),
                    DbValue::Text("George".into()),
                    DbValue::Integer(40),
                ],
            },
        )
        .unwrap();
        tx.commit().unwrap();

        // Frank should be gone from index
        let tx = db.begin_transaction();
        let results = tx
            .scan_by_index("users", "name", &b"Frank"[..]..=&b"Frank"[..])
            .unwrap();
        assert!(results.is_empty());

        // George should be found
        let results = tx
            .scan_by_index("users", "name", &b"George"[..]..=&b"George"[..])
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1.values[1], DbValue::Text("George".into()));
    }

    #[test]
    fn test_create_index_backfills_existing_data() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        // Insert rows before creating the index
        let tx = db.begin_transaction();
        tx.insert("users", &user_row("Alice", 30)).unwrap();
        tx.insert("users", &user_row("Bob", 25)).unwrap();
        tx.commit().unwrap();

        // Create index after data exists
        let tx = db.begin_transaction();
        tx.create_index("users", "name").unwrap();
        tx.commit().unwrap();

        // The back-filled data should be queryable
        let tx = db.begin_transaction();
        let results = tx
            .scan_by_index("users", "name", &b"Alice"[..]..=&b"Bob"[..])
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_drop_table_with_index() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.create_index("users", "name").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        tx.insert("users", &user_row("Alice", 30)).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        tx.drop_table("users").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        assert!(tx.find_table_meta("users").unwrap().is_none());
    }

    // ── Transaction conflict tests ─────────────────────────────────

    #[test]
    fn test_conflict_write_write_same_key() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let key = tx.insert("users", &user_row("Alice", 30)).unwrap();
        tx.commit().unwrap();

        // Two concurrent transactions update the same row
        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();

        tx1.update(
            "users",
            key,
            &Row {
                values: vec![
                    DbValue::Integer(key as i64),
                    DbValue::Text("Alice".into()),
                    DbValue::Integer(31),
                ],
            },
        )
        .unwrap();

        tx2.update(
            "users",
            key,
            &Row {
                values: vec![
                    DbValue::Integer(key as i64),
                    DbValue::Text("Alice".into()),
                    DbValue::Integer(32),
                ],
            },
        )
        .unwrap();

        // First to commit fails — the other active tx has conflicting writes
        tx1.commit().unwrap_err();
        // Second succeeds — conflicting tx is gone
        tx2.commit().unwrap();
    }

    #[test]
    fn test_conflict_read_write() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let key = tx.insert("users", &user_row("Alice", 30)).unwrap();
        tx.commit().unwrap();

        // tx1 reads a row, tx2 writes (updates) the same row
        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();

        tx1.get("users", key).unwrap();

        tx2.update(
            "users",
            key,
            &Row {
                values: vec![
                    DbValue::Integer(key as i64),
                    DbValue::Text("Alice".into()),
                    DbValue::Integer(31),
                ],
            },
        )
        .unwrap();

        // First to commit fails — other active tx has conflicting read/write
        tx1.commit().unwrap_err();
        tx2.commit().unwrap();
    }

    #[test]
    fn test_conflict_delete_vs_read() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let key = tx.insert("users", &user_row("Alice", 30)).unwrap();
        tx.commit().unwrap();

        // tx1 reads the row, tx2 deletes it
        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();

        tx1.get("users", key).unwrap();
        tx2.delete("users", key).unwrap();

        tx1.commit().unwrap_err();
        tx2.commit().unwrap();
    }

    #[test]
    fn test_conflict_range_scan_vs_insert() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let k1 = tx.insert("users", &user_row("Alice", 30)).unwrap();
        tx.commit().unwrap();

        // tx1 does a range scan, tx2 inserts a row whose key falls in that range
        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();

        tx1.scan("users", k1..).unwrap();

        let k2 = tx2.insert("users", &user_row("Bob", 25)).unwrap();
        // The new key should be >= k1 so it falls in the scanned range
        assert!(k2 >= k1);

        tx1.commit().unwrap_err();
        tx2.commit().unwrap();
    }

    #[test]
    fn test_conflict_index_scan_vs_insert() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.create_index("users", "name").unwrap();
        tx.commit().unwrap();

        // tx1 scans the index, tx2 inserts a row whose indexed value falls in range
        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();

        tx1.scan_by_index("users", "name", &b"A"[..]..&b"Z"[..])
            .unwrap();

        tx2.insert("users", &user_row("Bob", 25)).unwrap();

        tx1.commit().unwrap_err();
        tx2.commit().unwrap();
    }

    #[test]
    fn test_conflict_index_point_read_vs_insert() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.create_index("users", "name").unwrap();
        tx.commit().unwrap();

        // tx1 does a point lookup on index value "Alice", tx2 inserts "Alice"
        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();

        tx1.scan_by_index("users", "name", &b"Alice"[..]..=&b"Alice"[..])
            .unwrap();

        tx2.insert("users", &user_row("Alice", 30)).unwrap();

        tx1.commit().unwrap_err();
        tx2.commit().unwrap();
    }

    #[test]
    fn test_no_conflict_disjoint_rows() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let k1 = tx.insert("users", &user_row("Alice", 30)).unwrap();
        let k2 = tx.insert("users", &user_row("Bob", 25)).unwrap();
        tx.commit().unwrap();

        // tx1 reads row k1, tx2 reads row k2 — no overlap
        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();

        tx1.get("users", k1).unwrap();
        tx2.get("users", k2).unwrap();

        tx1.update(
            "users",
            k1,
            &Row {
                values: vec![
                    DbValue::Integer(k1 as i64),
                    DbValue::Text("Alice".into()),
                    DbValue::Integer(31),
                ],
            },
        )
        .unwrap();

        tx2.update(
            "users",
            k2,
            &Row {
                values: vec![
                    DbValue::Integer(k2 as i64),
                    DbValue::Text("Bob".into()),
                    DbValue::Integer(26),
                ],
            },
        )
        .unwrap();

        tx1.commit().unwrap();
        tx2.commit().unwrap(); // should succeed — no conflict
    }

    #[test]
    fn test_no_conflict_serial_transactions() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let key = tx.insert("users", &user_row("Alice", 30)).unwrap();
        tx.commit().unwrap();

        // tx1 commits before tx2 starts — no conflict possible
        let tx1 = db.begin_transaction();
        tx1.update(
            "users",
            key,
            &Row {
                values: vec![
                    DbValue::Integer(key as i64),
                    DbValue::Text("Alice".into()),
                    DbValue::Integer(31),
                ],
            },
        )
        .unwrap();
        tx1.commit().unwrap();

        let tx2 = db.begin_transaction();
        tx2.update(
            "users",
            key,
            &Row {
                values: vec![
                    DbValue::Integer(key as i64),
                    DbValue::Text("Alice".into()),
                    DbValue::Integer(32),
                ],
            },
        )
        .unwrap();
        tx2.commit().unwrap();
    }

    #[test]
    fn test_no_conflict_disjoint_index_ranges() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.create_index("users", "name").unwrap();
        tx.commit().unwrap();

        // tx1 scans names A-B, tx2 inserts name starting with Z
        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();

        tx1.scan_by_index("users", "name", &b"A"[..]..&b"C"[..])
            .unwrap();

        tx2.insert("users", &user_row("Zara", 20)).unwrap();

        tx2.commit().unwrap();
        tx1.commit().unwrap(); // should succeed — "Zara" is outside A..C
    }

    #[test]
    fn test_conflict_concurrent_inserts_same_index_value() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.create_index("users", "name").unwrap();
        tx.commit().unwrap();

        // Both transactions insert with the same indexed column value
        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();

        tx1.insert("users", &user_row("Alice", 30)).unwrap();
        tx2.insert("users", &user_row("Alice", 25)).unwrap();

        // First to commit fails — both wrote to same index value
        tx1.commit().unwrap_err();
        tx2.commit().unwrap();
    }

    #[test]
    fn test_conflict_update_vs_index_scan() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.create_index("users", "name").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let key = tx.insert("users", &user_row("Alice", 30)).unwrap();
        tx.commit().unwrap();

        // tx1 scans index range including "Bob", tx2 updates Alice → Bob
        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();

        tx1.scan_by_index("users", "name", &b"B"[..]..&b"C"[..])
            .unwrap();

        tx2.update(
            "users",
            key,
            &Row {
                values: vec![
                    DbValue::Integer(key as i64),
                    DbValue::Text("Bob".into()),
                    DbValue::Integer(30),
                ],
            },
        )
        .unwrap();

        tx1.commit().unwrap_err();
        tx2.commit().unwrap();
    }

    // ── DDL conflict tests ─────────────────────────────────────────

    #[test]
    fn test_conflict_concurrent_create_table_same_name() {
        let (db, _tmp) = open_db();

        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();

        tx1.create_table("users", users_schema()).unwrap();
        tx2.create_table("users", users_schema()).unwrap();

        tx1.commit().unwrap_err();
        tx2.commit().unwrap();
    }

    #[test]
    fn test_no_conflict_concurrent_create_table_different_names() {
        let (db, _tmp) = open_db();

        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();

        tx1.create_table("users", users_schema()).unwrap();
        tx2.create_table("products", users_schema()).unwrap();

        tx1.commit().unwrap();
        tx2.commit().unwrap();
    }

    #[test]
    fn test_conflict_concurrent_drop_table_same_name() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();

        tx1.drop_table("users").unwrap();
        tx2.drop_table("users").unwrap();

        tx1.commit().unwrap_err();
        tx2.commit().unwrap();
    }

    #[test]
    fn test_conflict_create_table_vs_drop_table_same_name() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        // tx1 drops the table, tx2 reads it (via insert)
        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();

        tx1.drop_table("users").unwrap();
        tx2.insert("users", &user_row("Alice", 30)).unwrap();

        tx1.commit().unwrap_err();
        tx2.commit().unwrap();
    }

    #[test]
    fn test_conflict_create_index_vs_insert() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        // tx1 creates an index (backfills via range read), tx2 inserts a row
        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();

        tx1.create_index("users", "name").unwrap();
        tx2.insert("users", &user_row("Alice", 30)).unwrap();

        tx1.commit().unwrap_err();
        tx2.commit().unwrap();
    }

    #[test]
    fn test_conflict_concurrent_create_index_same_column() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        // Both try to create an index on the same column
        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();

        tx1.create_index("users", "name").unwrap();
        tx2.create_index("users", "name").unwrap();

        tx1.commit().unwrap_err();
        tx2.commit().unwrap();
    }

    #[test]
    fn test_conflict_drop_index_vs_index_scan() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.create_index("users", "name").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        tx.insert("users", &user_row("Alice", 30)).unwrap();
        tx.commit().unwrap();

        // tx1 scans the index, tx2 drops it (modifying catalog metadata)
        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();

        tx1.scan_by_index("users", "name", &b"A"[..]..&b"Z"[..])
            .unwrap();
        tx2.drop_index("users", "name").unwrap();

        tx1.commit().unwrap_err();
        tx2.commit().unwrap();
    }

    #[test]
    fn test_conflict_drop_table_vs_read() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let key = tx.insert("users", &user_row("Alice", 30)).unwrap();
        tx.commit().unwrap();

        // tx1 reads a row, tx2 drops the whole table
        let tx1 = db.begin_transaction();
        let tx2 = db.begin_transaction();

        tx1.get("users", key).unwrap();
        tx2.drop_table("users").unwrap();

        tx1.commit().unwrap_err();
        tx2.commit().unwrap();
    }

    #[test]
    fn test_multiple_indexes_on_table() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.create_index("users", "name").unwrap();
        tx.create_index("users", "age").unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        tx.insert("users", &user_row("Alice", 30)).unwrap();
        tx.insert("users", &user_row("Bob", 25)).unwrap();
        tx.commit().unwrap();

        // Query by name
        let tx = db.begin_transaction();
        let by_name = tx
            .scan_by_index("users", "name", &b"Alice"[..]..=&b"Alice"[..])
            .unwrap();
        assert_eq!(by_name.len(), 1);

        // Query by age (i64 big-endian bytes for 25)
        let age_25 = 25i64.to_be_bytes().to_vec();
        let age_30 = 30i64.to_be_bytes().to_vec();
        let by_age = tx
            .scan_by_index("users", "age", age_25.as_slice()..=age_30.as_slice())
            .unwrap();
        assert_eq!(by_age.len(), 2);
    }

    #[test]
    fn test_index_access_multiple_same_key() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        let schema = Schema {
            columns: vec![
                Column {
                    name: "x".into(),
                    column_type: ColumnType::Integer,
                    nullable: false,
                },
                Column {
                    name: "y".into(),
                    column_type: ColumnType::Text,
                    nullable: false,
                },
            ],
            primary_key: None,
        };
        tx.create_table("t", schema).unwrap();
        tx.create_index("t", "x").unwrap();
        tx.insert(
            "t",
            &Row {
                values: vec![
                    DbValue::Null,
                    DbValue::Integer(1),
                    DbValue::Text("a".into()),
                ],
            },
        )
        .unwrap();
        tx.insert(
            "t",
            &Row {
                values: vec![
                    DbValue::Null,
                    DbValue::Integer(1),
                    DbValue::Text("b".into()),
                ],
            },
        )
        .unwrap();
        tx.insert(
            "t",
            &Row {
                values: vec![
                    DbValue::Null,
                    DbValue::Integer(2),
                    DbValue::Text("c".into()),
                ],
            },
        )
        .unwrap();
        tx.insert(
            "t",
            &Row {
                values: vec![
                    DbValue::Null,
                    DbValue::Integer(1),
                    DbValue::Text("d".into()),
                ],
            },
        )
        .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let key_1 = 1i64.to_be_bytes().to_vec();
        let meta = tx.find_table_meta("t").unwrap().unwrap();
        let _idx_root = meta.index_trees.iter().find(|(c, _)| c == "x").unwrap().1;
        let rows = tx
            .scan_by_index("t", "x", key_1.as_slice()..=key_1.as_slice())
            .unwrap();
        let mut values: Vec<String> = rows
            .iter()
            .map(|(_, row)| match &row.values[2] {
                DbValue::Text(s) => s.clone(),
                _ => panic!("not text"),
            })
            .collect();
        values.sort();
        assert_eq!(values, vec!["a", "b", "d"]);
    }

    // ── Primary key tests ─────────────────────────────────────────

    #[test]
    fn test_explicit_primary_key() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table(
            "items",
            Schema {
                columns: vec![
                    Column {
                        name: "id".into(),
                        column_type: ColumnType::Integer,
                        nullable: false,
                    },
                    Column {
                        name: "name".into(),
                        column_type: ColumnType::Text,
                        nullable: false,
                    },
                ],
                primary_key: Some(0),
            },
        )
        .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let key = tx
            .insert(
                "items",
                &Row {
                    values: vec![DbValue::Integer(42), DbValue::Text("widget".into())],
                },
            )
            .unwrap();
        // The B-tree key should equal the PK value
        assert_eq!(key, 42);
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let row = tx.get("items", 42).unwrap().unwrap();
        assert_eq!(row.values[0], DbValue::Integer(42));
        assert_eq!(row.values[1], DbValue::Text("widget".into()));
    }

    #[test]
    fn test_explicit_pk_no_rowid_prepended() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table(
            "t",
            Schema {
                columns: vec![
                    Column {
                        name: "id".into(),
                        column_type: ColumnType::Integer,
                        nullable: false,
                    },
                    Column {
                        name: "val".into(),
                        column_type: ColumnType::Text,
                        nullable: false,
                    },
                ],
                primary_key: Some(0),
            },
        )
        .unwrap();

        let meta = tx.find_table_meta("t").unwrap().unwrap();
        // No _rowid prepended when explicit PK is given
        assert_eq!(meta.schema.columns.len(), 2);
        assert_eq!(meta.schema.columns[0].name, "id");
        assert_eq!(meta.schema.primary_key, Some(0));
    }

    #[test]
    fn test_duplicate_primary_key_rejected() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table(
            "items",
            Schema {
                columns: vec![
                    Column {
                        name: "id".into(),
                        column_type: ColumnType::Integer,
                        nullable: false,
                    },
                    Column {
                        name: "name".into(),
                        column_type: ColumnType::Text,
                        nullable: false,
                    },
                ],
                primary_key: Some(0),
            },
        )
        .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        tx.insert(
            "items",
            &Row {
                values: vec![DbValue::Integer(1), DbValue::Text("a".into())],
            },
        )
        .unwrap();
        let err = tx
            .insert(
                "items",
                &Row {
                    values: vec![DbValue::Integer(1), DbValue::Text("b".into())],
                },
            )
            .unwrap_err();
        assert!(matches!(err, DatabaseError::DuplicateKey(1)));
    }

    #[test]
    fn test_negative_pk_rejected() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table(
            "items",
            Schema {
                columns: vec![Column {
                    name: "id".into(),
                    column_type: ColumnType::Integer,
                    nullable: false,
                }],
                primary_key: Some(0),
            },
        )
        .unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let err = tx
            .insert(
                "items",
                &Row {
                    values: vec![DbValue::Integer(-1)],
                },
            )
            .unwrap_err();
        assert!(matches!(err, DatabaseError::SchemaMismatch(_)));
    }

    #[test]
    fn test_implicit_rowid_auto_assigns() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        let tx = db.begin_transaction();
        let k1 = tx.insert("users", &user_row("Alice", 30)).unwrap();
        let k2 = tx.insert("users", &user_row("Bob", 25)).unwrap();
        // Auto-increment: k2 > k1
        assert!(k2 > k1);

        // _rowid is stored in the row
        let row = tx.get("users", k1).unwrap().unwrap();
        assert_eq!(row.values[0], DbValue::Integer(k1 as i64));
    }

    #[test]
    fn test_implicit_rowid_explicit_value() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        tx.create_table("users", users_schema()).unwrap();
        tx.commit().unwrap();

        // Provide explicit _rowid value
        let tx = db.begin_transaction();
        let key = tx
            .insert(
                "users",
                &Row {
                    values: vec![
                        DbValue::Integer(100),
                        DbValue::Text("Alice".into()),
                        DbValue::Integer(30),
                    ],
                },
            )
            .unwrap();
        assert_eq!(key, 100);

        let row = tx.get("users", 100).unwrap().unwrap();
        assert_eq!(row.values[0], DbValue::Integer(100));
        assert_eq!(row.values[1], DbValue::Text("Alice".into()));
    }

    #[test]
    fn test_pk_non_integer_rejected() {
        let (db, _tmp) = open_db();
        let tx = db.begin_transaction();
        let err = tx
            .create_table(
                "bad",
                Schema {
                    columns: vec![Column {
                        name: "id".into(),
                        column_type: ColumnType::Text,
                        nullable: false,
                    }],
                    primary_key: Some(0),
                },
            )
            .unwrap_err();
        assert!(matches!(err, DatabaseError::SchemaMismatch(_)));
    }
}
