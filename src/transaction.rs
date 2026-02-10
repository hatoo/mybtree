use std::{
    collections::{BTreeMap, BTreeSet},
    ops::{Bound, RangeBounds},
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
    range_reads: Vec<(Bound<Key>, Bound<Key>)>,
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
                range_reads: Vec::new(),
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

    pub fn insert(&self, value: Vec<u8>) -> Result<Key, Error> {
        let mut inner = self.store.lock().unwrap();
        let mut key = inner.btree.available_key()?;

        // Also consider keys from all active transactions' writes
        for op in inner.active_transactions.values() {
            if let Some(&max_write_key) = op.writes.keys().rev().find(|&&k| op.writes[&k].is_some())
            {
                let candidate = max_write_key.checked_add(1).unwrap_or(u64::MAX);
                if candidate > key {
                    key = candidate;
                }
            }
        }

        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            op.writes.insert(key, Some(value));
        }

        Ok(key)
    }

    pub fn read_range(&self, range: impl RangeBounds<Key>) -> Result<Vec<(Key, Vec<u8>)>, Error> {
        let mut inner = self.store.lock().unwrap();

        let range_bound = (range.start_bound().cloned(), range.end_bound().cloned());

        // Collect results from the underlying btree
        let mut results: BTreeMap<Key, Vec<u8>> = BTreeMap::new();
        inner.btree.read_range(range_bound.clone(), |key, value| {
            results.insert(key, value.to_vec());
        })?;

        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            // Record individual keys as reads
            for key in results.keys() {
                op.reads.insert(*key);
            }
            // Record range read for phantom protection
            op.range_reads.push(range_bound.clone());

            // Overlay local writes
            for (key, value) in &op.writes {
                if range_bound.contains(key) {
                    if let Some(v) = value {
                        results.insert(*key, v.clone());
                    } else {
                        results.remove(key);
                    }
                }
            }
        }

        Ok(results.into_iter().collect())
    }

    pub fn remove_range(&self, range: impl RangeBounds<Key>) -> Result<(), Error> {
        let mut inner = self.store.lock().unwrap();

        let range_bound = (range.start_bound().cloned(), range.end_bound().cloned());

        // Find all keys in range from btree
        let mut keys_to_remove: Vec<Key> = Vec::new();
        inner.btree.read_range(range_bound.clone(), |key, _| {
            keys_to_remove.push(key);
        })?;

        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            // Also find keys in range from local writes
            for (key, value) in op.writes.clone() {
                if range_bound.contains(&key) && value.is_some() && !keys_to_remove.contains(&key) {
                    keys_to_remove.push(key);
                }
            }

            // Record range read for conflict detection
            op.range_reads.push(range_bound);

            // Mark all keys for removal
            for key in keys_to_remove {
                op.reads.insert(key);
                op.writes.insert(key, None);
            }
        }

        Ok(())
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
                // Check if other's range reads conflict with our writes
                for range_read in &other_op.range_reads {
                    for write_key in current_op.writes.keys() {
                        if range_read.contains(write_key) {
                            fail!(TransactionError::Conflict);
                        }
                    }
                }
                // Check if our range reads conflict with other's writes
                for range_read in &current_op.range_reads {
                    for write_key in other_op.writes.keys() {
                        if range_read.contains(write_key) {
                            fail!(TransactionError::Conflict);
                        }
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

        let pager = Pager::new(file, 256);
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

    #[test]
    fn test_read_range_basic() {
        let (store, _temp) = setup_transaction_store();

        {
            let tx = store.begin_transaction();
            tx.write(1, vec![1]);
            tx.write(3, vec![3]);
            tx.write(5, vec![5]);
            tx.write(7, vec![7]);
            tx.write(9, vec![9]);
            assert!(tx.commit().is_ok());
        }

        let tx = store.begin_transaction();
        let results = tx.read_range(2..=7).unwrap();
        assert_eq!(results, vec![(3, vec![3]), (5, vec![5]), (7, vec![7])]);
    }

    #[test]
    fn test_read_range_with_local_writes() {
        let (store, _temp) = setup_transaction_store();

        {
            let tx = store.begin_transaction();
            tx.write(1, vec![1]);
            tx.write(5, vec![5]);
            assert!(tx.commit().is_ok());
        }

        let tx = store.begin_transaction();
        tx.write(3, vec![3]);
        let results = tx.read_range(1..=5).unwrap();
        assert_eq!(results, vec![(1, vec![1]), (3, vec![3]), (5, vec![5])]);
    }

    #[test]
    fn test_read_range_with_local_removes() {
        let (store, _temp) = setup_transaction_store();

        {
            let tx = store.begin_transaction();
            tx.write(1, vec![1]);
            tx.write(3, vec![3]);
            tx.write(5, vec![5]);
            assert!(tx.commit().is_ok());
        }

        let tx = store.begin_transaction();
        tx.remove(3);
        let results = tx.read_range(1..=5).unwrap();
        assert_eq!(results, vec![(1, vec![1]), (5, vec![5])]);
    }

    #[test]
    fn test_read_range_empty() {
        let (store, _temp) = setup_transaction_store();

        let tx = store.begin_transaction();
        let results = tx.read_range(1..=10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_remove_range_basic() {
        let (store, _temp) = setup_transaction_store();

        {
            let tx = store.begin_transaction();
            tx.write(1, vec![1]);
            tx.write(3, vec![3]);
            tx.write(5, vec![5]);
            tx.write(7, vec![7]);
            assert!(tx.commit().is_ok());
        }

        {
            let tx = store.begin_transaction();
            tx.remove_range(2..=5).unwrap();
            assert!(tx.commit().is_ok());
        }

        // Verify only keys outside range remain
        {
            let tx = store.begin_transaction();
            assert_eq!(tx.read(1).unwrap(), Some(vec![1]));
            assert_eq!(tx.read(3).unwrap(), None);
            assert_eq!(tx.read(5).unwrap(), None);
            assert_eq!(tx.read(7).unwrap(), Some(vec![7]));
        }
    }

    #[test]
    fn test_remove_range_with_local_writes() {
        let (store, _temp) = setup_transaction_store();

        let tx = store.begin_transaction();
        tx.write(1, vec![1]);
        tx.write(3, vec![3]);
        tx.write(5, vec![5]);
        tx.remove_range(2..=4).unwrap();

        // Key 3 should be removed, 1 and 5 should remain
        assert_eq!(tx.read(1).unwrap(), Some(vec![1]));
        assert_eq!(tx.read(3).unwrap(), None);
        assert_eq!(tx.read(5).unwrap(), Some(vec![5]));
    }

    #[test]
    fn test_read_range_conflict_with_write() {
        let (store, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        // tx1 reads a range
        tx1.read_range(1..=10).unwrap();

        // tx2 writes to a key within that range
        tx2.write(5, vec![5]);

        // tx2 should conflict because tx1 has a range read covering key 5
        assert!(tx2.commit().is_err());
    }

    #[test]
    fn test_read_range_no_conflict_outside() {
        let (store, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        // tx1 reads a range
        tx1.read_range(1..=10).unwrap();

        // tx2 writes to a key outside that range
        tx2.write(20, vec![20]);

        // No conflict expected
        assert!(tx2.commit().is_ok());
    }

    #[test]
    fn test_remove_range_conflict() {
        let (store, _temp) = setup_transaction_store();

        {
            let tx = store.begin_transaction();
            tx.write(3, vec![3]);
            assert!(tx.commit().is_ok());
        }

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        tx1.remove_range(1..=5).unwrap();
        tx2.write(4, vec![4]);

        // tx2 writes to key 4, which is within tx1's range read -> conflict
        assert!(tx2.commit().is_err());
    }

    #[test]
    fn test_write_conflicts_with_range_read() {
        let (store, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        tx1.read_range(1..=10).unwrap();
        tx2.write(5, vec![5]);

        // tx1 commits: its range_reads cover 1..=10, tx2 writes to 5 which is in range -> conflict
        assert!(tx1.commit().is_err());
    }

    #[test]
    fn test_insert_returns_unique_key() {
        let (store, _temp) = setup_transaction_store();

        let tx = store.begin_transaction();
        let k1 = tx.insert(vec![1]).unwrap();
        let k2 = tx.insert(vec![2]).unwrap();
        let k3 = tx.insert(vec![3]).unwrap();

        assert_ne!(k1, k2);
        assert_ne!(k2, k3);
        assert_eq!(tx.read(k1).unwrap(), Some(vec![1]));
        assert_eq!(tx.read(k2).unwrap(), Some(vec![2]));
        assert_eq!(tx.read(k3).unwrap(), Some(vec![3]));
    }

    #[test]
    fn test_insert_after_existing_keys() {
        let (store, _temp) = setup_transaction_store();

        {
            let tx = store.begin_transaction();
            tx.write(10, vec![10]);
            assert!(tx.commit().is_ok());
        }

        let tx = store.begin_transaction();
        let key = tx.insert(vec![42]).unwrap();
        assert!(key > 10);
        assert_eq!(tx.read(key).unwrap(), Some(vec![42]));
        assert!(tx.commit().is_ok());
    }

    #[test]
    fn test_insert_concurrent_no_conflict() {
        let (store, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        let k1 = tx1.insert(vec![1]).unwrap();
        let k2 = tx2.insert(vec![2]).unwrap();

        // Keys should be different, so no conflict
        assert_ne!(k1, k2);
        assert!(tx1.commit().is_ok());
        assert!(tx2.commit().is_ok());
    }

    #[test]
    fn test_insert_persists_after_commit() {
        let (store, _temp) = setup_transaction_store();

        let key;
        {
            let tx = store.begin_transaction();
            key = tx.insert(vec![99]).unwrap();
            assert!(tx.commit().is_ok());
        }

        {
            let tx = store.begin_transaction();
            assert_eq!(tx.read(key).unwrap(), Some(vec![99]));
        }
    }
}
