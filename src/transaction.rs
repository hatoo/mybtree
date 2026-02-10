use std::{
    collections::{BTreeMap, BTreeSet},
    ops::{Bound, RangeBounds},
    sync::Mutex,
};

use rkyv::rancor::{Error, fail};

use crate::{Btree, Key, NodePtr, Value};

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
    inner: Mutex<TransactionStoreInner>,
}

pub struct Transaction<'a> {
    store: &'a Mutex<TransactionStoreInner>,
    tx_id: usize,
}

pub struct Operation {
    reads: BTreeSet<(NodePtr, Key)>,
    writes: BTreeMap<(NodePtr, Key), Option<Vec<u8>>>,
    range_reads: Vec<(NodePtr, Bound<Key>, Bound<Key>)>,
}

impl TransactionStore {
    pub fn new(btree: Btree) -> Self {
        TransactionStore {
            inner: Mutex::new(TransactionStoreInner {
                btree,
                next_tx_id: 0,
                active_transactions: BTreeMap::new(),
            }),
        }
    }

    pub fn begin_transaction(&self) -> Transaction<'_> {
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
            store: &self.inner,
            tx_id,
        }
    }

    pub fn init_table(&self) -> Result<NodePtr, Error> {
        let mut inner = self.inner.lock().unwrap();
        inner.btree.init_table()
    }

    pub fn drop_table(&self, root: NodePtr) -> Result<(), Error> {
        let mut inner = self.inner.lock().unwrap();
        inner.btree.free_tree(root)
    }

    pub fn get_next_page_num(&self) -> u64 {
        let inner = self.inner.lock().unwrap();
        inner.btree.pager.get_next_page_num()
    }

    pub fn init_index(&self) -> Result<NodePtr, Error> {
        let mut inner = self.inner.lock().unwrap();
        inner.btree.init_index()
    }

    pub fn drop_index_tree(&self, root: NodePtr) -> Result<(), Error> {
        let mut inner = self.inner.lock().unwrap();
        inner.btree.free_index_tree(root)
    }

    pub fn index_insert(&self, idx_root: NodePtr, key: Key, value: Vec<u8>) -> Result<bool, Error> {
        let mut inner = self.inner.lock().unwrap();
        inner.btree.index_insert(idx_root, key, value)
    }
}

impl<'a> Transaction<'a> {
    pub fn read(&self, root: NodePtr, key: Key) -> Result<Option<Vec<u8>>, Error> {
        let mut inner = self.store.lock().unwrap();
        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            op.reads.insert((root, key));
            if let Some(value) = op.writes.get(&(root, key)) {
                return Ok(value.clone());
            }
        }
        let value = inner.btree.read(root, key, |f| f.map(|v| v.to_vec()))?;
        Ok(value)
    }

    pub fn write(&self, root: NodePtr, key: Key, value: Vec<u8>) {
        let mut inner = self.store.lock().unwrap();
        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            op.writes.insert((root, key), Some(value));
        }
    }

    pub fn remove(&self, root: NodePtr, key: Key) {
        let mut inner = self.store.lock().unwrap();
        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            op.writes.insert((root, key), None);
        }
    }

    pub fn insert(&self, root: NodePtr, value: Vec<u8>) -> Result<Key, Error> {
        let mut inner = self.store.lock().unwrap();
        let mut key = inner.btree.available_key(root)?;

        // Also consider keys from all active transactions' writes for the same root
        for op in inner.active_transactions.values() {
            if let Some(&(_, max_write_key)) = op
                .writes
                .keys()
                .rev()
                .find(|&&(r, k)| r == root && op.writes[&(r, k)].is_some())
            {
                let candidate = max_write_key.checked_add(1).unwrap_or(u64::MAX);
                if candidate > key {
                    key = candidate;
                }
            }
        }

        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            op.writes.insert((root, key), Some(value));
        }

        Ok(key)
    }

    pub fn read_range(
        &self,
        root: NodePtr,
        range: impl RangeBounds<Key>,
    ) -> Result<Vec<(Key, Vec<u8>)>, Error> {
        let mut inner = self.store.lock().unwrap();

        let range_bound = (range.start_bound().cloned(), range.end_bound().cloned());

        // Collect results from the underlying btree
        let mut results: BTreeMap<Key, Vec<u8>> = BTreeMap::new();
        inner
            .btree
            .read_range(root, range_bound.clone(), |key, value| {
                results.insert(key, value.to_vec());
            })?;

        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            // Record individual keys as reads
            for key in results.keys() {
                op.reads.insert((root, *key));
            }
            // Record range read for phantom protection
            op.range_reads
                .push((root, range_bound.0.clone(), range_bound.1.clone()));

            // Overlay local writes for this root
            for ((w_root, key), value) in &op.writes {
                if *w_root == root && range_bound.contains(key) {
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

    pub fn remove_range(&self, root: NodePtr, range: impl RangeBounds<Key>) -> Result<(), Error> {
        let mut inner = self.store.lock().unwrap();

        let range_bound = (range.start_bound().cloned(), range.end_bound().cloned());

        // Find all keys in range from btree
        let mut keys_to_remove: Vec<Key> = Vec::new();
        inner
            .btree
            .read_range(root, range_bound.clone(), |key, _| {
                keys_to_remove.push(key);
            })?;

        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            // Also find keys in range from local writes
            for ((w_root, key), value) in op.writes.clone() {
                if w_root == root
                    && range_bound.contains(&key)
                    && value.is_some()
                    && !keys_to_remove.contains(&key)
                {
                    keys_to_remove.push(key);
                }
            }

            // Record range read for conflict detection
            op.range_reads.push((root, range_bound.0, range_bound.1));

            // Mark all keys for removal
            for key in keys_to_remove {
                op.reads.insert((root, key));
                op.writes.insert((root, key), None);
            }
        }

        Ok(())
    }

    // ── Index tree operations (applied directly to btree) ───────────

    pub fn index_insert(&self, idx_root: NodePtr, key: Key, value: Vec<u8>) -> Result<bool, Error> {
        let mut inner = self.store.lock().unwrap();
        inner.btree.index_insert(idx_root, key, value)
    }

    pub fn index_remove(&self, idx_root: NodePtr, value: &Value, key: Key) -> Result<bool, Error> {
        let mut inner = self.store.lock().unwrap();
        inner.btree.index_remove(idx_root, value, key)
    }

    pub fn index_read_range<R: RangeBounds<Vec<u8>>>(
        &self,
        idx_root: NodePtr,
        range: R,
    ) -> Result<Vec<Key>, Error> {
        let mut inner = self.store.lock().unwrap();
        let mut keys = Vec::new();
        inner
            .btree
            .index_read_range(idx_root, range, |_v, k| keys.push(k))?;
        Ok(keys)
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
                for (rr_root, rr_start, rr_end) in &other_op.range_reads {
                    for (w_root, w_key) in current_op.writes.keys() {
                        if w_root == rr_root && (rr_start.clone(), rr_end.clone()).contains(w_key) {
                            fail!(TransactionError::Conflict);
                        }
                    }
                }
                // Check if our range reads conflict with other's writes
                for (rr_root, rr_start, rr_end) in &current_op.range_reads {
                    for (w_root, w_key) in other_op.writes.keys() {
                        if w_root == rr_root && (rr_start.clone(), rr_end.clone()).contains(w_key) {
                            fail!(TransactionError::Conflict);
                        }
                    }
                }
            }
        }

        for ((root, key), value) in current_op.writes {
            if let Some(value) = value {
                inner.btree.insert(root, key, value)?;
            } else {
                inner.btree.remove(root, key)?;
            }
        }

        Ok(())
    }
}

impl<'a> Drop for Transaction<'a> {
    fn drop(&mut self) {
        let mut inner = self.store.lock().unwrap();
        inner.active_transactions.remove(&self.tx_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NodePtr, Pager};
    use std::fs;
    use tempfile::NamedTempFile;

    fn setup_transaction_store() -> (TransactionStore, NodePtr, NamedTempFile) {
        let temp_file = NamedTempFile::new().unwrap();
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(temp_file.path())
            .unwrap();

        let pager = Pager::new(file, 256);
        let mut btree = Btree::new(pager);
        btree.init().unwrap();
        let root = btree.init_table().unwrap();

        let store = TransactionStore::new(btree);
        (store, root, temp_file)
    }

    #[test]
    fn test_begin_transaction() {
        let (store, _root, _temp) = setup_transaction_store();
        let tx = store.begin_transaction();
        assert_eq!(tx.tx_id, 0);
    }

    #[test]
    fn test_multiple_transaction_ids() {
        let (store, _root, _temp) = setup_transaction_store();
        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();
        let tx3 = store.begin_transaction();

        assert_eq!(tx1.tx_id, 0);
        assert_eq!(tx2.tx_id, 1);
        assert_eq!(tx3.tx_id, 2);
    }

    #[test]
    fn test_write_and_read_in_transaction() {
        let (store, root, _temp) = setup_transaction_store();
        let tx = store.begin_transaction();

        let key = 1u64;
        let value = vec![1, 2, 3, 4, 5];

        tx.write(root, key, value.clone());
        let read_value = tx.read(root, key).unwrap();

        assert_eq!(read_value, Some(value));
    }

    #[test]
    fn test_read_nonexistent_key() {
        let (store, root, _temp) = setup_transaction_store();
        let tx = store.begin_transaction();

        let read_value = tx.read(root, 999).unwrap();
        assert_eq!(read_value, None);
    }

    #[test]
    fn test_write_multiple_keys() {
        let (store, root, _temp) = setup_transaction_store();
        let tx = store.begin_transaction();

        tx.write(root, 1, vec![1, 2, 3]);
        tx.write(root, 2, vec![4, 5, 6]);
        tx.write(root, 3, vec![7, 8, 9]);

        assert_eq!(tx.read(root, 1).unwrap(), Some(vec![1, 2, 3]));
        assert_eq!(tx.read(root, 2).unwrap(), Some(vec![4, 5, 6]));
        assert_eq!(tx.read(root, 3).unwrap(), Some(vec![7, 8, 9]));
    }

    #[test]
    fn test_overwrite_value() {
        let (store, root, _temp) = setup_transaction_store();
        let tx = store.begin_transaction();

        let key = 1u64;
        tx.write(root, key, vec![1, 2, 3]);
        assert_eq!(tx.read(root, key).unwrap(), Some(vec![1, 2, 3]));

        tx.write(root, key, vec![4, 5, 6]);
        assert_eq!(tx.read(root, key).unwrap(), Some(vec![4, 5, 6]));
    }

    #[test]
    fn test_commit_single_transaction() {
        let (store, root, _temp) = setup_transaction_store();
        let tx = store.begin_transaction();

        tx.write(root, 1, vec![1, 2, 3]);
        let result = tx.commit();
        assert!(result.is_ok());
    }

    #[test]
    fn test_concurrent_reads_no_conflict() {
        let (store, root, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        tx1.write(root, 1, vec![1, 2, 3]);
        tx2.write(root, 2, vec![4, 5, 6]);

        assert!(tx1.commit().is_ok());
        assert!(tx2.commit().is_ok());
    }

    #[test]
    fn test_write_write_conflict() {
        let (store, root, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        tx1.write(root, 1, vec![1, 2, 3]);
        tx2.write(root, 1, vec![4, 5, 6]);

        // Both transactions write to the same key, so first one to commit should fail
        // because there's an active transaction with conflicting writes
        let result = tx1.commit();
        assert!(result.is_err());
    }

    #[test]
    fn test_read_write_conflict() {
        let (store, root, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        tx1.read(root, 1).unwrap();
        tx2.write(root, 1, vec![4, 5, 6]);

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
        let (store, root, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        tx1.write(root, 1, vec![1, 2, 3]);
        tx2.read(root, 1).unwrap();

        // tx1 tries to commit: it writes to key 1. tx2 is active and read from key 1.
        // The check: does tx1.writes (1) conflict with tx2?
        // For read_key in tx2.reads: if tx1.writes contains it -> CONFLICT
        // tx2.reads includes 1, tx1.writes includes 1 -> CONFLICT
        let result = tx1.commit();
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_concurrent_transactions() {
        let (store, root, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();
        let tx3 = store.begin_transaction();

        tx1.write(root, 1, vec![1, 2, 3]);
        tx2.write(root, 2, vec![4, 5, 6]);
        tx3.write(root, 3, vec![7, 8, 9]);

        assert!(tx1.commit().is_ok());
        assert!(tx2.commit().is_ok());
        assert!(tx3.commit().is_ok());
    }

    #[test]
    fn test_empty_transaction_commit() {
        let (store, _root, _temp) = setup_transaction_store();
        let tx = store.begin_transaction();
        assert!(tx.commit().is_ok());
    }

    #[test]
    fn test_conflict_error_type() {
        let (store, root, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        tx1.write(root, 1, vec![1, 2, 3]);
        tx2.write(root, 1, vec![4, 5, 6]);

        let result = tx1.commit();
        assert!(result.is_err());
        // Verify it's a proper error with the conflict in it
        let err_str = format!("{:?}", result.unwrap_err());
        assert!(err_str.contains("Conflict") || err_str.contains("TransactionError"));
    }

    #[test]
    fn test_sequential_transactions() {
        let (store, root, _temp) = setup_transaction_store();

        // First transaction
        {
            let tx = store.begin_transaction();
            tx.write(root, 1, vec![1, 2, 3]);
            assert!(tx.commit().is_ok());
        }

        // Second transaction
        {
            let tx = store.begin_transaction();
            tx.write(root, 2, vec![4, 5, 6]);
            assert!(tx.commit().is_ok());
        }
    }

    #[test]
    fn test_remove_key() {
        let (store, root, _temp) = setup_transaction_store();

        // First, insert a key
        {
            let tx = store.begin_transaction();
            tx.write(root, 1, vec![1, 2, 3]);
            assert!(tx.commit().is_ok());
        }

        // Then remove it
        {
            let tx = store.begin_transaction();
            tx.remove(root, 1);
            assert!(tx.commit().is_ok());
        }

        // Verify it's gone
        {
            let tx = store.begin_transaction();
            assert_eq!(tx.read(root, 1).unwrap(), None);
        }
    }

    #[test]
    fn test_remove_in_transaction() {
        let (store, root, _temp) = setup_transaction_store();
        let tx = store.begin_transaction();

        tx.write(root, 1, vec![1, 2, 3]);
        assert_eq!(tx.read(root, 1).unwrap(), Some(vec![1, 2, 3]));

        tx.remove(root, 1);
        assert_eq!(tx.read(root, 1).unwrap(), None);
    }

    #[test]
    fn test_data_persists_after_commit() {
        let (store, root, _temp) = setup_transaction_store();

        // Write in first transaction
        {
            let tx = store.begin_transaction();
            tx.write(root, 1, vec![1, 2, 3]);
            tx.write(root, 2, vec![4, 5, 6]);
            assert!(tx.commit().is_ok());
        }

        // Read in second transaction to verify persistence
        {
            let tx = store.begin_transaction();
            assert_eq!(tx.read(root, 1).unwrap(), Some(vec![1, 2, 3]));
            assert_eq!(tx.read(root, 2).unwrap(), Some(vec![4, 5, 6]));
        }
    }

    #[test]
    fn test_commit_after_conflict_fails() {
        let (store, root, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        tx1.write(root, 1, vec![1, 2, 3]);
        tx2.write(root, 1, vec![4, 5, 6]);

        // First commit should fail due to conflict
        assert!(tx1.commit().is_err());

        // Second transaction should succeed now
        assert!(tx2.commit().is_ok());
    }

    #[test]
    fn test_read_after_write_in_other_transaction() {
        let (store, root, _temp) = setup_transaction_store();

        // First transaction writes
        {
            let tx = store.begin_transaction();
            tx.write(root, 1, vec![1, 2, 3]);
            assert!(tx.commit().is_ok());
        }

        // Second transaction reads and writes to different key
        {
            let tx = store.begin_transaction();
            assert_eq!(tx.read(root, 1).unwrap(), Some(vec![1, 2, 3]));
            tx.write(root, 2, vec![4, 5, 6]);
            assert!(tx.commit().is_ok());
        }
    }

    #[test]
    fn test_multiple_reads_same_key() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let tx = store.begin_transaction();
            tx.write(root, 1, vec![1, 2, 3]);
            assert!(tx.commit().is_ok());
        }

        let tx = store.begin_transaction();
        assert_eq!(tx.read(root, 1).unwrap(), Some(vec![1, 2, 3]));
        assert_eq!(tx.read(root, 1).unwrap(), Some(vec![1, 2, 3]));
        assert_eq!(tx.read(root, 1).unwrap(), Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_write_after_read_same_transaction() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let tx = store.begin_transaction();
            tx.write(root, 1, vec![1, 2, 3]);
            assert!(tx.commit().is_ok());
        }

        let tx = store.begin_transaction();
        assert_eq!(tx.read(root, 1).unwrap(), Some(vec![1, 2, 3]));
        tx.write(root, 1, vec![4, 5, 6]);
        assert_eq!(tx.read(root, 1).unwrap(), Some(vec![4, 5, 6]));
        assert!(tx.commit().is_ok());
    }

    #[test]
    fn test_complex_conflict_scenario() {
        let (store, root, _temp) = setup_transaction_store();

        // Setup: write initial values
        {
            let tx = store.begin_transaction();
            tx.write(root, 1, vec![1]);
            tx.write(root, 2, vec![2]);
            tx.write(root, 3, vec![3]);
            assert!(tx.commit().is_ok());
        }

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();
        let tx3 = store.begin_transaction();

        // tx1 reads 1, writes 4
        tx1.read(root, 1).unwrap();
        tx1.write(root, 4, vec![4, 4]);

        // tx2 reads 2, writes 3
        tx2.read(root, 2).unwrap();
        tx2.write(root, 3, vec![3, 3]);

        // tx3 writes 5
        tx3.write(root, 5, vec![5, 5]);

        // tx3 should succeed (writes to 5, no conflicts with any other transaction)
        assert!(tx3.commit().is_ok());

        // tx1 should succeed (reads 1, writes 4, no conflicts)
        assert!(tx1.commit().is_ok());

        // tx2 should succeed (reads 2, writes 3, no conflicts)
        assert!(tx2.commit().is_ok());
    }

    #[test]
    fn test_remove_with_conflict() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let tx = store.begin_transaction();
            tx.write(root, 1, vec![1, 2, 3]);
            assert!(tx.commit().is_ok());
        }

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        tx1.read(root, 1).unwrap();
        tx2.remove(root, 1);

        // tx2 removes key 1 that tx1 read -> conflict
        assert!(tx2.commit().is_err());
    }

    #[test]
    fn test_read_range_basic() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let tx = store.begin_transaction();
            tx.write(root, 1, vec![1]);
            tx.write(root, 3, vec![3]);
            tx.write(root, 5, vec![5]);
            tx.write(root, 7, vec![7]);
            tx.write(root, 9, vec![9]);
            assert!(tx.commit().is_ok());
        }

        let tx = store.begin_transaction();
        let results = tx.read_range(root, 2..=7).unwrap();
        assert_eq!(results, vec![(3, vec![3]), (5, vec![5]), (7, vec![7])]);
    }

    #[test]
    fn test_read_range_with_local_writes() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let tx = store.begin_transaction();
            tx.write(root, 1, vec![1]);
            tx.write(root, 5, vec![5]);
            assert!(tx.commit().is_ok());
        }

        let tx = store.begin_transaction();
        tx.write(root, 3, vec![3]);
        let results = tx.read_range(root, 1..=5).unwrap();
        assert_eq!(results, vec![(1, vec![1]), (3, vec![3]), (5, vec![5])]);
    }

    #[test]
    fn test_read_range_with_local_removes() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let tx = store.begin_transaction();
            tx.write(root, 1, vec![1]);
            tx.write(root, 3, vec![3]);
            tx.write(root, 5, vec![5]);
            assert!(tx.commit().is_ok());
        }

        let tx = store.begin_transaction();
        tx.remove(root, 3);
        let results = tx.read_range(root, 1..=5).unwrap();
        assert_eq!(results, vec![(1, vec![1]), (5, vec![5])]);
    }

    #[test]
    fn test_read_range_empty() {
        let (store, root, _temp) = setup_transaction_store();

        let tx = store.begin_transaction();
        let results = tx.read_range(root, 1..=10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_remove_range_basic() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let tx = store.begin_transaction();
            tx.write(root, 1, vec![1]);
            tx.write(root, 3, vec![3]);
            tx.write(root, 5, vec![5]);
            tx.write(root, 7, vec![7]);
            assert!(tx.commit().is_ok());
        }

        {
            let tx = store.begin_transaction();
            tx.remove_range(root, 2..=5).unwrap();
            assert!(tx.commit().is_ok());
        }

        // Verify only keys outside range remain
        {
            let tx = store.begin_transaction();
            assert_eq!(tx.read(root, 1).unwrap(), Some(vec![1]));
            assert_eq!(tx.read(root, 3).unwrap(), None);
            assert_eq!(tx.read(root, 5).unwrap(), None);
            assert_eq!(tx.read(root, 7).unwrap(), Some(vec![7]));
        }
    }

    #[test]
    fn test_remove_range_with_local_writes() {
        let (store, root, _temp) = setup_transaction_store();

        let tx = store.begin_transaction();
        tx.write(root, 1, vec![1]);
        tx.write(root, 3, vec![3]);
        tx.write(root, 5, vec![5]);
        tx.remove_range(root, 2..=4).unwrap();

        // Key 3 should be removed, 1 and 5 should remain
        assert_eq!(tx.read(root, 1).unwrap(), Some(vec![1]));
        assert_eq!(tx.read(root, 3).unwrap(), None);
        assert_eq!(tx.read(root, 5).unwrap(), Some(vec![5]));
    }

    #[test]
    fn test_read_range_conflict_with_write() {
        let (store, root, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        // tx1 reads a range
        tx1.read_range(root, 1..=10).unwrap();

        // tx2 writes to a key within that range
        tx2.write(root, 5, vec![5]);

        // tx2 should conflict because tx1 has a range read covering key 5
        assert!(tx2.commit().is_err());
    }

    #[test]
    fn test_read_range_no_conflict_outside() {
        let (store, root, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        // tx1 reads a range
        tx1.read_range(root, 1..=10).unwrap();

        // tx2 writes to a key outside that range
        tx2.write(root, 20, vec![20]);

        // No conflict expected
        assert!(tx2.commit().is_ok());
    }

    #[test]
    fn test_remove_range_conflict() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let tx = store.begin_transaction();
            tx.write(root, 3, vec![3]);
            assert!(tx.commit().is_ok());
        }

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        tx1.remove_range(root, 1..=5).unwrap();
        tx2.write(root, 4, vec![4]);

        // tx2 writes to key 4, which is within tx1's range read -> conflict
        assert!(tx2.commit().is_err());
    }

    #[test]
    fn test_write_conflicts_with_range_read() {
        let (store, root, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        tx1.read_range(root, 1..=10).unwrap();
        tx2.write(root, 5, vec![5]);

        // tx1 commits: its range_reads cover 1..=10, tx2 writes to 5 which is in range -> conflict
        assert!(tx1.commit().is_err());
    }

    #[test]
    fn test_insert_returns_unique_key() {
        let (store, root, _temp) = setup_transaction_store();

        let tx = store.begin_transaction();
        let k1 = tx.insert(root, vec![1]).unwrap();
        let k2 = tx.insert(root, vec![2]).unwrap();
        let k3 = tx.insert(root, vec![3]).unwrap();

        assert_ne!(k1, k2);
        assert_ne!(k2, k3);
        assert_eq!(tx.read(root, k1).unwrap(), Some(vec![1]));
        assert_eq!(tx.read(root, k2).unwrap(), Some(vec![2]));
        assert_eq!(tx.read(root, k3).unwrap(), Some(vec![3]));
    }

    #[test]
    fn test_insert_after_existing_keys() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let tx = store.begin_transaction();
            tx.write(root, 10, vec![10]);
            assert!(tx.commit().is_ok());
        }

        let tx = store.begin_transaction();
        let key = tx.insert(root, vec![42]).unwrap();
        assert!(key > 10);
        assert_eq!(tx.read(root, key).unwrap(), Some(vec![42]));
        assert!(tx.commit().is_ok());
    }

    #[test]
    fn test_insert_concurrent_no_conflict() {
        let (store, root, _temp) = setup_transaction_store();

        let tx1 = store.begin_transaction();
        let tx2 = store.begin_transaction();

        let k1 = tx1.insert(root, vec![1]).unwrap();
        let k2 = tx2.insert(root, vec![2]).unwrap();

        // Keys should be different, so no conflict
        assert_ne!(k1, k2);
        assert!(tx1.commit().is_ok());
        assert!(tx2.commit().is_ok());
    }

    #[test]
    fn test_insert_persists_after_commit() {
        let (store, root, _temp) = setup_transaction_store();

        let key;
        {
            let tx = store.begin_transaction();
            key = tx.insert(root, vec![99]).unwrap();
            assert!(tx.commit().is_ok());
        }

        {
            let tx = store.begin_transaction();
            assert_eq!(tx.read(root, key).unwrap(), Some(vec![99]));
        }
    }
}
