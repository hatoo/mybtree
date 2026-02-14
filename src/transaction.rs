use std::{
    collections::{BTreeMap, BTreeSet},
    ops::{Bound, RangeBounds},
    sync::Mutex,
};

use crate::{Btree, Key, NodePtr, tree::TreeError};

#[derive(Debug, thiserror::Error)]
pub enum TransactionError {
    #[error("Transaction conflict detected")]
    Conflict,
    #[error("Tree error: {0}")]
    TreeError(#[from] TreeError),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

struct TransactionStoreInner<const N: usize> {
    btree: Btree<N>,
    next_tx_id: usize,
    active_transactions: BTreeMap<usize, Operation>,
}

pub struct TransactionStore<const N: usize> {
    inner: Mutex<TransactionStoreInner<N>>,
}

pub struct Transaction<'a, const N: usize> {
    store: &'a Mutex<TransactionStoreInner<N>>,
    tx_id: usize,
}

pub struct Operation {
    reads: BTreeSet<(NodePtr, Key)>,
    writes: BTreeMap<(NodePtr, Key), Option<Vec<u8>>>,
    range_reads: Vec<(NodePtr, Bound<Key>, Bound<Key>)>,
    // Index conflict tracking
    index_reads: BTreeSet<(NodePtr, Vec<u8>)>,
    index_writes: BTreeSet<(NodePtr, Vec<u8>)>,
    index_range_reads: Vec<(NodePtr, Bound<Vec<u8>>, Bound<Vec<u8>>)>,
    // Deferred index operations: (idx_root, value_bytes, key) → true=insert, false=remove
    index_ops: BTreeMap<(NodePtr, Vec<u8>, Key), bool>,
    // Deferred structural operations
    deferred_free_trees: Vec<NodePtr>,
    deferred_free_index_trees: Vec<NodePtr>,
    deferred_init_trees: Vec<NodePtr>,
    deferred_init_indexes: Vec<NodePtr>,
}

fn any_write_in_ranges<'a, T: Ord + Clone + 'a>(
    writes: impl Iterator<Item = &'a (NodePtr, T)>,
    ranges: &[(NodePtr, Bound<T>, Bound<T>)],
) -> bool {
    for (w_root, w_key) in writes {
        for (rr_root, rr_start, rr_end) in ranges {
            if w_root == rr_root && (rr_start.clone(), rr_end.clone()).contains(w_key) {
                return true;
            }
        }
    }
    false
}

impl Operation {
    fn conflicts_with(&self, other: &Operation) -> bool {
        // Data: other's reads vs self's writes
        for read_key in &other.reads {
            if self.writes.contains_key(read_key) {
                return true;
            }
        }
        // Data: other's writes vs self's reads and writes
        for write_key in other.writes.keys() {
            if self.reads.contains(write_key) {
                return true;
            }
            if self.writes.contains_key(write_key) {
                return true;
            }
        }
        // Data: range reads vs writes (both directions)
        if any_write_in_ranges(self.writes.keys(), &other.range_reads) {
            return true;
        }
        if any_write_in_ranges(other.writes.keys(), &self.range_reads) {
            return true;
        }

        // Index: point read/write conflicts
        for read_key in &other.index_reads {
            if self.index_writes.contains(read_key) {
                return true;
            }
        }
        for read_key in &self.index_reads {
            if other.index_writes.contains(read_key) {
                return true;
            }
        }
        for write_key in &other.index_writes {
            if self.index_writes.contains(write_key) {
                return true;
            }
        }
        // Index: range reads vs writes (both directions)
        if any_write_in_ranges(self.index_writes.iter(), &other.index_range_reads) {
            return true;
        }
        if any_write_in_ranges(other.index_writes.iter(), &self.index_range_reads) {
            return true;
        }

        false
    }
}

impl<const N: usize> TransactionStore<N> {
    pub fn new(btree: Btree<N>) -> Self {
        TransactionStore {
            inner: Mutex::new(TransactionStoreInner {
                btree,
                next_tx_id: 0,
                active_transactions: BTreeMap::new(),
            }),
        }
    }

    pub fn begin_transaction(&self) -> Transaction<'_, N> {
        let mut inner = self.inner.lock().unwrap();
        let tx_id = inner.next_tx_id;
        inner.next_tx_id = inner.next_tx_id.wrapping_add(1);
        inner.active_transactions.insert(
            tx_id,
            Operation {
                reads: BTreeSet::new(),
                writes: BTreeMap::new(),
                range_reads: Vec::new(),
                index_reads: BTreeSet::new(),
                index_writes: BTreeSet::new(),
                index_range_reads: Vec::new(),
                index_ops: BTreeMap::new(),
                deferred_free_trees: Vec::new(),
                deferred_free_index_trees: Vec::new(),
                deferred_init_trees: Vec::new(),
                deferred_init_indexes: Vec::new(),
            },
        );
        Transaction {
            store: &self.inner,
            tx_id,
        }
    }

    pub fn get_total_page_count(&self) -> u64 {
        let inner = self.inner.lock().unwrap();
        inner.btree.pager.total_page_count()
    }
}

impl<'a, const N: usize> Transaction<'a, N> {
    pub fn read(&self, root: NodePtr, key: Key) -> Result<Option<Vec<u8>>, TreeError> {
        let mut inner = self.store.lock().unwrap();
        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            op.reads.insert((root, key));
            if let Some(value) = op.writes.get(&(root, key)) {
                return Ok(value.clone());
            }
        }
        let value = inner.btree.read(root, key)?.map(|v| v.to_vec());
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

    pub fn insert(&self, root: NodePtr, value: Vec<u8>) -> Result<Key, TreeError> {
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
    ) -> Result<Vec<(Key, Vec<u8>)>, TreeError> {
        let mut inner = self.store.lock().unwrap();

        let range_bound = (range.start_bound().cloned(), range.end_bound().cloned());

        // Collect results from the underlying btree
        let mut results: BTreeMap<Key, Vec<u8>> = BTreeMap::new();
        inner
            .btree
            .read_range(root, range_bound.clone(), |key, value: &[u8]| {
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

    pub fn remove_range(
        &self,
        root: NodePtr,
        range: impl RangeBounds<Key>,
    ) -> Result<(), TreeError> {
        let mut inner = self.store.lock().unwrap();

        let range_bound = (range.start_bound().cloned(), range.end_bound().cloned());

        // Find all keys in range from btree
        let mut keys_to_remove: Vec<Key> = Vec::new();
        inner
            .btree
            .read_range(root, range_bound.clone(), |key, _: &[u8]| {
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

    // ── Structural operations ──────────────────────────────────────

    pub fn init_tree(&self) -> Result<NodePtr, TreeError> {
        let mut inner = self.store.lock().unwrap();
        let page = inner.btree.init_tree()?;
        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            op.deferred_init_trees.push(page);
        }
        Ok(page)
    }

    pub fn init_index(&self) -> Result<NodePtr, TreeError> {
        let mut inner = self.store.lock().unwrap();
        let page = inner.btree.init_index()?;
        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            op.deferred_init_indexes.push(page);
        }
        Ok(page)
    }

    pub fn free_tree(&self, root: NodePtr) -> Result<(), TreeError> {
        let mut inner = self.store.lock().unwrap();
        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            op.deferred_free_trees.push(root);
        }
        Ok(())
    }

    pub fn free_index_tree(&self, root: NodePtr) -> Result<(), TreeError> {
        let mut inner = self.store.lock().unwrap();
        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            op.deferred_free_index_trees.push(root);
        }
        Ok(())
    }

    // ── Index tree operations (deferred, with local overlay) ────────

    pub fn index_insert(
        &self,
        idx_root: NodePtr,
        key: Key,
        value: Vec<u8>,
    ) -> Result<bool, TreeError> {
        let mut inner = self.store.lock().unwrap();
        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            op.index_writes.insert((idx_root, value.clone()));
            op.index_ops.insert((idx_root, value, key), true);
        }
        Ok(true)
    }

    pub fn index_remove(
        &self,
        idx_root: NodePtr,
        value: &[u8],
        key: Key,
    ) -> Result<bool, TreeError> {
        let mut inner = self.store.lock().unwrap();
        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            op.index_writes.insert((idx_root, value.to_vec()));
            op.index_ops.insert((idx_root, value.to_vec(), key), false);
        }
        Ok(true)
    }

    pub fn index_read(&self, idx_root: NodePtr, value: &[u8]) -> Result<Option<Key>, TreeError> {
        let mut inner = self.store.lock().unwrap();
        let value_bytes = value.to_vec();

        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            op.index_reads.insert((idx_root, value_bytes.clone()));

            // Check local overlay: find a locally inserted entry
            let start = (idx_root, value_bytes.clone(), 0u64);
            let end = (idx_root, value_bytes.clone(), u64::MAX);
            for ((_, _, k), is_insert) in op.index_ops.range(start..=end) {
                if *is_insert {
                    return Ok(Some(*k));
                }
            }
        }

        // Read from btree
        let btree_key = inner.btree.index_read(idx_root, value)?;

        if let Some(key) = btree_key {
            if let Some(op) = inner.active_transactions.get(&self.tx_id) {
                if op.index_ops.get(&(idx_root, value_bytes, key)) == Some(&false) {
                    return Ok(None);
                }
            }
            return Ok(Some(key));
        }

        Ok(None)
    }

    pub fn index_read_range<R: RangeBounds<Vec<u8>>>(
        &self,
        idx_root: NodePtr,
        range: R,
    ) -> Result<Vec<Key>, TreeError> {
        let mut inner = self.store.lock().unwrap();
        let range_bound = (range.start_bound().cloned(), range.end_bound().cloned());

        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            op.index_range_reads
                .push((idx_root, range_bound.0.clone(), range_bound.1.clone()));
        }

        // Read from btree: collect (value_bytes, key) pairs
        let mut btree_entries: Vec<(Vec<u8>, Key)> = Vec::new();
        inner.btree.index_read_range(
            idx_root,
            (range_bound.0.clone(), range_bound.1.clone()),
            |v: &[u8], k| {
                btree_entries.push((v.to_vec(), k));
            },
        )?;

        if let Some(op) = inner.active_transactions.get(&self.tx_id) {
            let mut result_keys = Vec::new();

            // Include btree results not locally removed
            for (v, k) in &btree_entries {
                if op.index_ops.get(&(idx_root, v.clone(), *k)) != Some(&false) {
                    result_keys.push(*k);
                }
            }

            // Add locally inserted entries whose value is in range
            for ((root, val, key), is_insert) in &op.index_ops {
                if *root == idx_root && *is_insert && range_bound.contains(val) {
                    if !result_keys.contains(key) {
                        result_keys.push(*key);
                    }
                }
            }

            Ok(result_keys)
        } else {
            Ok(btree_entries.into_iter().map(|(_, k)| k).collect())
        }
    }

    pub fn commit(self) -> Result<(), TransactionError> {
        let mut inner = self.store.lock().unwrap();

        let current_op = inner.active_transactions.remove(&self.tx_id).unwrap();

        let conflict = inner
            .active_transactions
            .values()
            .any(|other_op| current_op.conflicts_with(other_op));

        if conflict {
            for page in current_op.deferred_init_trees {
                inner.btree.pager.free_page(page)?;
            }
            for page in current_op.deferred_init_indexes {
                inner.btree.pager.free_page(page)?;
            }
            return Err(TransactionError::Conflict);
        }

        // Apply node writes
        for ((root, key), value) in current_op.writes {
            if let Some(value) = value {
                inner.btree.insert(root, key, &value)?;
            } else {
                inner.btree.remove(root, key)?;
            }
        }

        // Apply deferred index operations
        for ((idx_root, value_bytes, key), is_insert) in current_op.index_ops {
            if is_insert {
                inner.btree.index_insert(idx_root, key, &value_bytes)?;
            } else {
                inner.btree.index_remove(idx_root, &value_bytes, key)?;
            }
        }

        // Apply deferred free operations
        for root in current_op.deferred_free_trees {
            inner.btree.free_tree(root)?;
        }
        for root in current_op.deferred_free_index_trees {
            inner.btree.free_index_tree(root)?;
        }

        inner.btree.flush()?;

        Ok(())
    }
}

impl<'a, const N: usize> Drop for Transaction<'a, N> {
    fn drop(&mut self) {
        let mut inner = self.store.lock().unwrap();
        if let Some(op) = inner.active_transactions.remove(&self.tx_id) {
            for page in op.deferred_init_trees {
                let _ = inner.btree.pager.free_page(page);
            }
            for page in op.deferred_init_indexes {
                let _ = inner.btree.pager.free_page(page);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NodePtr, Pager};
    use std::fs;
    use tempfile::NamedTempFile;

    fn setup_transaction_store() -> (TransactionStore<4096>, NodePtr, NamedTempFile) {
        let temp_file = NamedTempFile::new().unwrap();
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(temp_file.path())
            .unwrap();

        let pager = Pager::<4096>::new(file);
        let mut btree = Btree::new(pager);
        btree.init().unwrap();
        let root = btree.init_tree().unwrap();

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
