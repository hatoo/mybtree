use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

use rkyv::rancor::{Error, fail};

use crate::{Btree, Key};

#[derive(Debug, thiserror::Error)]
pub enum TransactionError {
    #[error("Transaction conflict detected")]
    Conflict,
}

struct TransactionStoreInner {
    btree: Btree,
    next_tx_id: usize,
    active_transactions: BTreeMap<usize, Operation>,
}

pub struct TransactionStore {
    inner: Arc<Mutex<TransactionStoreInner>>,
}

pub struct Transaction {
    store: Arc<Mutex<TransactionStoreInner>>,
    tx_id: usize,
}

pub struct Operation {
    reads: BTreeSet<Key>,
    writes: BTreeMap<Key, Option<Vec<u8>>>,
}

impl TransactionStore {
    pub fn new(btree: Btree) -> Self {
        TransactionStore {
            inner: Arc::new(Mutex::new(TransactionStoreInner {
                btree,
                next_tx_id: 0,
                active_transactions: BTreeMap::new(),
            })),
        }
    }

    pub fn begin_transaction(&self) -> Transaction {
        let mut inner = self.inner.lock().unwrap();
        let tx_id = inner.next_tx_id;
        inner.next_tx_id = inner.next_tx_id.wrapping_add(1);
        inner.active_transactions.insert(
            tx_id,
            Operation {
                reads: BTreeSet::new(),
                writes: BTreeMap::new(),
            },
        );
        Transaction {
            store: Arc::clone(&self.inner),
            tx_id,
        }
    }
}

impl Transaction {
    pub fn read(&self, key: Key) -> Result<Option<Vec<u8>>, Error> {
        let mut inner = self.store.lock().unwrap();
        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            op.reads.insert(key);
            if let Some(value) = op.writes.get(&key) {
                return Ok(value.clone());
            }
        }
        let value = inner.btree.read(key, |f| f.map(|v| v.to_vec()))?;
        Ok(value)
    }

    pub fn write(&self, key: Key, value: Vec<u8>) {
        let mut inner = self.store.lock().unwrap();
        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            op.writes.insert(key, Some(value));
        }
    }

    pub fn remove(&self, key: Key) {
        let mut inner = self.store.lock().unwrap();
        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            op.writes.insert(key, None);
        }
    }

    pub fn commit(self) -> Result<(), Error> {
        let mut inner = self.store.lock().unwrap();

        let current_op = inner.active_transactions.remove(&self.tx_id).unwrap();

        for (other_tx_id, other_op) in &inner.active_transactions {
            if *other_tx_id != self.tx_id {
                for read_key in &other_op.reads {
                    if current_op.writes.contains_key(read_key) {
                        fail!(TransactionError::Conflict);
                    }
                }
                for write_key in other_op.writes.keys() {
                    if current_op.reads.contains(write_key) {
                        fail!(TransactionError::Conflict);
                    }
                    if current_op.writes.contains_key(write_key) {
                        fail!(TransactionError::Conflict);
                    }
                }
            }
        }

        for (key, value) in current_op.writes {
            if let Some(value) = value {
                inner.btree.insert(key, value)?;
            } else {
                inner.btree.remove(key)?;
            }
        }

        Ok(())
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        let mut inner = self.store.lock().unwrap();
        inner.active_transactions.remove(&self.tx_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pager;
    use std::fs;
    use tempfile::NamedTempFile;

    fn setup_transaction_store() -> (TransactionStore, NamedTempFile) {
        let temp_file = NamedTempFile::new().unwrap();
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(temp_file.path())
            .unwrap();

        let pager = Pager::new(file);
        let mut btree = Btree::new(pager);
        btree.init().unwrap();

        let store = TransactionStore::new(btree);
        (store, temp_file)
    }

    #[test]
    fn test_begin_transaction() {
        let (store, _temp) = setup_transaction_store();
        let tx = store.begin_transaction();
        assert_eq!(tx.tx_id, 0);
    }

    #[test]
    fn test_multiple_transaction_ids() {
        let (store, _temp) = setup_transaction_store();
        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();
        let tx3 = store.begin_transaction();

        assert_eq!(tx1.tx_id, 0);
        assert_eq!(tx2.tx_id, 1);
        assert_eq!(tx3.tx_id, 2);
    }

    #[test]
    fn test_write_and_read_in_transaction() {
        let (store, _temp) = setup_transaction_store();
        let tx = store.begin_transaction();

        let key = 1u64;
        let value = vec![1, 2, 3, 4, 5];

        tx.write(key, value.clone());
        let read_value = tx.read(key).unwrap();

        assert_eq!(read_value, Some(value));
    }

    #[test]
    fn test_read_nonexistent_key() {
        let (store, _temp) = setup_transaction_store();
        let tx = store.begin_transaction();

        let read_value = tx.read(999).unwrap();
        assert_eq!(read_value, None);
    }

    #[test]
    fn test_write_multiple_keys() {
        let (store, _temp) = setup_transaction_store();
        let tx = store.begin_transaction();

        tx.write(1, vec![1, 2, 3]);
        tx.write(2, vec![4, 5, 6]);
        tx.write(3, vec![7, 8, 9]);

        assert_eq!(tx.read(1).unwrap(), Some(vec![1, 2, 3]));
        assert_eq!(tx.read(2).unwrap(), Some(vec![4, 5, 6]));
        assert_eq!(tx.read(3).unwrap(), Some(vec![7, 8, 9]));
    }

    #[test]
    fn test_overwrite_value() {
        let (store, _temp) = setup_transaction_store();
        let tx = store.begin_transaction();

        let key = 1u64;
        tx.write(key, vec![1, 2, 3]);
        assert_eq!(tx.read(key).unwrap(), Some(vec![1, 2, 3]));

        tx.write(key, vec![4, 5, 6]);
        assert_eq!(tx.read(key).unwrap(), Some(vec![4, 5, 6]));
    }

    #[test]
    fn test_commit_single_transaction() {
        let (store, _temp) = setup_transaction_store();
        let tx = store.begin_transaction();

        tx.write(1, vec![1, 2, 3]);
        let result = tx.commit();
        assert!(result.is_ok());
    }

    #[test]
    fn test_concurrent_reads_no_conflict() {
        let (store, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        tx1.write(1, vec![1, 2, 3]);
        tx2.write(2, vec![4, 5, 6]);

        assert!(tx1.commit().is_ok());
        assert!(tx2.commit().is_ok());
    }

    #[test]
    fn test_write_write_conflict() {
        let (store, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        tx1.write(1, vec![1, 2, 3]);
        tx2.write(1, vec![4, 5, 6]);

        // Both transactions write to the same key, so first one to commit should fail
        // because there's an active transaction with conflicting writes
        let result = tx1.commit();
        assert!(result.is_err());
    }

    #[test]
    fn test_read_write_conflict() {
        let (store, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        tx1.read(1).unwrap();
        tx2.write(1, vec![4, 5, 6]);

        // tx2 tries to commit: it writes to key 1 that tx1 read
        // The commit check: does tx1.reads contain any of tx2.writes? No, tx1 reads 1 but tx2 writes 1
        // Actually, the check is: does tx2.writes conflict with active transactions?
        // Active transactions include tx1. tx1.reads includes 1. tx2.writes includes 1.
        // So there's a conflict: another transaction (tx1) read a key (1) that we (tx2) are writing to
        let result = tx2.commit();
        assert!(result.is_err());
    }

    #[test]
    fn test_write_read_conflict() {
        let (store, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        tx1.write(1, vec![1, 2, 3]);
        tx2.read(1).unwrap();

        // tx1 tries to commit: it writes to key 1. tx2 is active and read from key 1.
        // The check: does tx1.writes (1) conflict with tx2?
        // For read_key in tx2.reads: if tx1.writes contains it -> CONFLICT
        // tx2.reads includes 1, tx1.writes includes 1 -> CONFLICT
        let result = tx1.commit();
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_concurrent_transactions() {
        let (store, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();
        let tx3 = store.begin_transaction();

        tx1.write(1, vec![1, 2, 3]);
        tx2.write(2, vec![4, 5, 6]);
        tx3.write(3, vec![7, 8, 9]);

        assert!(tx1.commit().is_ok());
        assert!(tx2.commit().is_ok());
        assert!(tx3.commit().is_ok());
    }

    #[test]
    fn test_empty_transaction_commit() {
        let (store, _temp) = setup_transaction_store();
        let tx = store.begin_transaction();
        assert!(tx.commit().is_ok());
    }

    #[test]
    fn test_conflict_error_type() {
        let (store, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        tx1.write(1, vec![1, 2, 3]);
        tx2.write(1, vec![4, 5, 6]);

        let result = tx1.commit();
        assert!(result.is_err());
        // Verify it's a proper error with the conflict in it
        let err_str = format!("{:?}", result.unwrap_err());
        assert!(err_str.contains("Conflict") || err_str.contains("TransactionError"));
    }

    #[test]
    fn test_sequential_transactions() {
        let (store, _temp) = setup_transaction_store();

        // First transaction
        {
            let tx = store.begin_transaction();
            tx.write(1, vec![1, 2, 3]);
            assert!(tx.commit().is_ok());
        }

        // Second transaction
        {
            let tx = store.begin_transaction();
            tx.write(2, vec![4, 5, 6]);
            assert!(tx.commit().is_ok());
        }
    }

    #[test]
    fn test_remove_key() {
        let (store, _temp) = setup_transaction_store();

        // First, insert a key
        {
            let tx = store.begin_transaction();
            tx.write(1, vec![1, 2, 3]);
            assert!(tx.commit().is_ok());
        }

        // Then remove it
        {
            let tx = store.begin_transaction();
            tx.remove(1);
            assert!(tx.commit().is_ok());
        }

        // Verify it's gone
        {
            let tx = store.begin_transaction();
            assert_eq!(tx.read(1).unwrap(), None);
        }
    }

    #[test]
    fn test_remove_in_transaction() {
        let (store, _temp) = setup_transaction_store();
        let tx = store.begin_transaction();

        tx.write(1, vec![1, 2, 3]);
        assert_eq!(tx.read(1).unwrap(), Some(vec![1, 2, 3]));

        tx.remove(1);
        assert_eq!(tx.read(1).unwrap(), None);
    }

    #[test]
    fn test_data_persists_after_commit() {
        let (store, _temp) = setup_transaction_store();

        // Write in first transaction
        {
            let tx = store.begin_transaction();
            tx.write(1, vec![1, 2, 3]);
            tx.write(2, vec![4, 5, 6]);
            assert!(tx.commit().is_ok());
        }

        // Read in second transaction to verify persistence
        {
            let tx = store.begin_transaction();
            assert_eq!(tx.read(1).unwrap(), Some(vec![1, 2, 3]));
            assert_eq!(tx.read(2).unwrap(), Some(vec![4, 5, 6]));
        }
    }

    #[test]
    fn test_commit_after_conflict_fails() {
        let (store, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        tx1.write(1, vec![1, 2, 3]);
        tx2.write(1, vec![4, 5, 6]);

        // First commit should fail due to conflict
        assert!(tx1.commit().is_err());

        // Second transaction should succeed now
        assert!(tx2.commit().is_ok());
    }

    #[test]
    fn test_read_after_write_in_other_transaction() {
        let (store, _temp) = setup_transaction_store();

        // First transaction writes
        {
            let tx = store.begin_transaction();
            tx.write(1, vec![1, 2, 3]);
            assert!(tx.commit().is_ok());
        }

        // Second transaction reads and writes to different key
        {
            let tx = store.begin_transaction();
            assert_eq!(tx.read(1).unwrap(), Some(vec![1, 2, 3]));
            tx.write(2, vec![4, 5, 6]);
            assert!(tx.commit().is_ok());
        }
    }

    #[test]
    fn test_multiple_reads_same_key() {
        let (store, _temp) = setup_transaction_store();

        {
            let tx = store.begin_transaction();
            tx.write(1, vec![1, 2, 3]);
            assert!(tx.commit().is_ok());
        }

        let tx = store.begin_transaction();
        assert_eq!(tx.read(1).unwrap(), Some(vec![1, 2, 3]));
        assert_eq!(tx.read(1).unwrap(), Some(vec![1, 2, 3]));
        assert_eq!(tx.read(1).unwrap(), Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_write_after_read_same_transaction() {
        let (store, _temp) = setup_transaction_store();

        {
            let tx = store.begin_transaction();
            tx.write(1, vec![1, 2, 3]);
            assert!(tx.commit().is_ok());
        }

        let tx = store.begin_transaction();
        assert_eq!(tx.read(1).unwrap(), Some(vec![1, 2, 3]));
        tx.write(1, vec![4, 5, 6]);
        assert_eq!(tx.read(1).unwrap(), Some(vec![4, 5, 6]));
        assert!(tx.commit().is_ok());
    }

    #[test]
    fn test_complex_conflict_scenario() {
        let (store, _temp) = setup_transaction_store();

        // Setup: write initial values
        {
            let tx = store.begin_transaction();
            tx.write(1, vec![1]);
            tx.write(2, vec![2]);
            tx.write(3, vec![3]);
            assert!(tx.commit().is_ok());
        }

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();
        let tx3 = store.begin_transaction();

        // tx1 reads 1, writes 4
        tx1.read(1).unwrap();
        tx1.write(4, vec![4, 4]);

        // tx2 reads 2, writes 3
        tx2.read(2).unwrap();
        tx2.write(3, vec![3, 3]);

        // tx3 writes 5
        tx3.write(5, vec![5, 5]);

        // tx3 should succeed (writes to 5, no conflicts with any other transaction)
        assert!(tx3.commit().is_ok());

        // tx1 should succeed (reads 1, writes 4, no conflicts)
        assert!(tx1.commit().is_ok());

        // tx2 should succeed (reads 2, writes 3, no conflicts)
        assert!(tx2.commit().is_ok());
    }

    #[test]
    fn test_remove_with_conflict() {
        let (store, _temp) = setup_transaction_store();

        {
            let tx = store.begin_transaction();
            tx.write(1, vec![1, 2, 3]);
            assert!(tx.commit().is_ok());
        }

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        tx1.read(1).unwrap();
        tx2.remove(1);

        // tx2 removes key 1 that tx1 read -> conflict
        assert!(tx2.commit().is_err());
    }
}
