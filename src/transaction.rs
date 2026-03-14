use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    ops::{Bound, DerefMut, RangeBounds},
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

pub struct LockedTransaction<'a, const N: usize> {
    btree: &'a mut Btree<N>,
    active_transactions: &'a mut Operation,
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

impl<'a, const N: usize> LockedTransaction<'a, N> {
    pub fn available_key(&mut self, root: NodePtr) -> Result<Key, TreeError> {
        let mut key = self.btree.available_key(root)?;

        if let Some(k) = self
            .active_transactions
            .writes
            .range((root, key)..=(root, Key::MAX))
            .last()
        {
            key = k.0.1 + 1;
        }

        Ok(key)
    }

    pub fn read(&mut self, root: NodePtr, key: Key) -> Result<Option<Cow<'_, [u8]>>, TreeError> {
        self.active_transactions.reads.insert((root, key));
        if let Some(value) = self.active_transactions.writes.get(&(root, key)) {
            return Ok(value.as_deref().map(Cow::Borrowed));
        }
        let value = self.btree.read(root, key)?;
        Ok(value)
    }

    pub fn read_range<F, E>(
        &mut self,
        root: NodePtr,
        range: impl RangeBounds<Key>,
        mut f: F,
    ) -> Result<(), E>
    where
        for<'local> F: FnMut(LockedTransaction<'local, N>, Key, &[u8]) -> Result<bool, E>,
        E: From<TreeError>,
    {
        let range_bound = (range.start_bound().cloned(), range.end_bound().cloned());
        let &mut LockedTransaction {
            ref mut btree,
            ref mut active_transactions,
        } = self;
        active_transactions
            .range_reads
            .push((root, range_bound.0, range_bound.1));

        let writes = active_transactions
            .writes
            .iter()
            .filter(|((r, _k), v)| *r == root && v.is_some())
            .map(|((_, k), v)| (*k, v.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut last_bound = Bound::Unbounded;
        btree.read_range(root, range_bound, |btree, k, v| {
            active_transactions.reads.insert((root, k));

            // check active writes between last_key and k for this root
            for (&write_key, value) in writes.range((last_bound, Bound::Excluded(k))) {
                if let Some(v) = value {
                    let me = LockedTransaction {
                        btree,
                        active_transactions,
                    };
                    if f(me, write_key, v)? {
                        return Ok(true);
                    }
                }
            }

            last_bound = Bound::Excluded(k);

            // check active writes
            if let Some(v) = active_transactions.writes.get(&(root, k)) {
                if let Some(v) = v {
                    let v = v.clone();
                    let me = LockedTransaction {
                        btree,
                        active_transactions,
                    };
                    f(me, k, &v)
                } else {
                    Ok(false) // deleted key, skip
                }
            } else {
                let me = LockedTransaction {
                    btree,
                    active_transactions,
                };

                f(me, k, v)
            }
        })?;
        // check active writes between last_key and k for this root
        for (&write_key, value) in writes.range((last_bound, range_bound.1)) {
            if let Some(v) = value {
                let me = LockedTransaction {
                    btree,
                    active_transactions,
                };
                if f(me, write_key, v)? {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    pub fn write(&mut self, root: NodePtr, key: Key, value: Vec<u8>) {
        self.active_transactions
            .writes
            .insert((root, key), Some(value));
    }

    pub fn remove(&mut self, root: NodePtr, key: Key) {
        self.active_transactions.writes.insert((root, key), None);
    }

    pub fn insert(&mut self, root: NodePtr, value: Vec<u8>) -> Result<Key, TreeError> {
        let mut key = self.btree.available_key(root)?;

        // Also consider keys from this transaction's writes for the same root.
        if let Some(&(_, max_write_key)) = self
            .active_transactions
            .writes
            .keys()
            .rev()
            .find(|&&(r, k)| r == root && self.active_transactions.writes[&(r, k)].is_some())
        {
            let candidate = max_write_key.saturating_add(1);
            if candidate > key {
                key = candidate;
            }
        }

        self.active_transactions
            .writes
            .insert((root, key), Some(value));

        Ok(key)
    }

    pub fn remove_range_where<F, E: From<TreeError>>(
        &mut self,
        root: NodePtr,
        range: impl RangeBounds<Key>,
        mut f: F,
    ) -> Result<(), E>
    where
        for<'local> F: FnMut(LockedTransaction<'local, N>, Key, &[u8]) -> Result<(bool, bool), E>,
    {
        let range_bound = (range.start_bound().cloned(), range.end_bound().cloned());

        self.active_transactions
            .range_reads
            .push((root, range_bound.0, range_bound.1));

        let mut next_key = Some(range_bound.0);

        while let Some(search_from) = next_key {
            let next_write_key = self
                .active_transactions
                .writes
                .range(match search_from {
                    Bound::Included(key) => (Bound::Included((root, key)), Bound::Unbounded),
                    Bound::Excluded(key) => (Bound::Excluded((root, key)), Bound::Unbounded),
                    Bound::Unbounded => (Bound::Included((root, 0)), Bound::Unbounded),
                })
                .take_while(|((write_root, _), _)| *write_root == root)
                .find_map(|(&(write_root, write_key), value)| {
                    (write_root == root && range_bound.contains(&write_key) && value.is_some())
                        .then_some(write_key)
                });

            let mut next_tree_entry = None;
            self.btree.read_range(
                root,
                (search_from, range_bound.1),
                |_btree, key, value: &[u8]| {
                    next_tree_entry = Some((key, value.to_vec()));
                    Ok::<_, TreeError>(true)
                },
            )?;

            let next_tree_key = next_tree_entry.as_ref().map(|(key, _)| *key);
            let key = match (next_write_key, next_tree_key) {
                (Some(write_key), Some(tree_key)) => write_key.min(tree_key),
                (Some(write_key), None) => write_key,
                (None, Some(tree_key)) => tree_key,
                (None, None) => break,
            };

            if next_tree_key == Some(key) {
                self.active_transactions.reads.insert((root, key));
            }

            let current_value = match self.active_transactions.writes.get(&(root, key)) {
                Some(Some(value)) => Some(value.clone()),
                Some(None) => None,
                None => next_tree_entry
                    .filter(|(tree_key, _)| *tree_key == key)
                    .map(|(_, value)| value),
            };

            let Some(value) = current_value else {
                next_key = Some(Bound::Excluded(key));
                continue;
            };

            let (should_remove, should_stop) = f(
                LockedTransaction {
                    btree: self.btree,
                    active_transactions: self.active_transactions,
                },
                key,
                &value,
            )?;

            if should_remove {
                self.active_transactions.writes.insert((root, key), None);
            }
            if should_stop {
                break;
            }

            next_key = Some(Bound::Excluded(key));
        }

        Ok(())
    }

    pub fn remove_range(
        &mut self,
        root: NodePtr,
        range: impl RangeBounds<Key>,
    ) -> Result<(), TreeError> {
        let range_bound = (range.start_bound().cloned(), range.end_bound().cloned());

        // Find all keys in range from btree
        let mut keys_to_remove: Vec<Key> = Vec::new();
        self.btree
            .read_range(root, range_bound, |_btree, key, _: &[u8]| {
                keys_to_remove.push(key);
                Ok::<_, TreeError>(false)
            })?;

        // Also find keys in range from local writes
        for ((w_root, key), value) in &self.active_transactions.writes {
            if *w_root == root
                && range_bound.contains(key)
                && value.is_some()
                && !keys_to_remove.contains(key)
            {
                keys_to_remove.push(*key);
            }
        }

        // Record range read for conflict detection
        self.active_transactions
            .range_reads
            .push((root, range_bound.0, range_bound.1));

        // Mark all keys for removal
        for key in keys_to_remove {
            self.active_transactions.reads.insert((root, key));
            self.active_transactions.writes.insert((root, key), None);
        }

        Ok(())
    }

    // ── Index tree operations (deferred, with local overlay) ────────

    pub fn index_read(
        &mut self,
        idx_root: NodePtr,
        value: &[u8],
    ) -> Result<Option<Key>, TreeError> {
        let value_bytes = value.to_vec();

        self.active_transactions
            .index_reads
            .insert((idx_root, value_bytes.clone()));

        // Check local overlay: find a locally inserted entry
        let start = (idx_root, value_bytes.clone(), 0u64);
        let end = (idx_root, value_bytes.clone(), u64::MAX);
        for ((_, _, k), is_insert) in self.active_transactions.index_ops.range(start..=end) {
            if *is_insert {
                return Ok(Some(*k));
            }
        }

        // Read from btree
        let btree_key = self.btree.index_read(idx_root, value)?;

        if let Some(key) = btree_key {
            if self
                .active_transactions
                .index_ops
                .get(&(idx_root, value_bytes, key))
                == Some(&false)
            {
                return Ok(None);
            }
            return Ok(Some(key));
        }

        Ok(None)
    }

    pub fn index_insert(&mut self, idx_root: NodePtr, key: Key, value: Vec<u8>) {
        self.active_transactions
            .index_writes
            .insert((idx_root, value.clone()));
        self.active_transactions
            .index_ops
            .insert((idx_root, value, key), true);
    }

    pub fn index_remove(&mut self, idx_root: NodePtr, value: &[u8], key: Key) {
        self.active_transactions
            .index_writes
            .insert((idx_root, value.to_vec()));
        self.active_transactions
            .index_ops
            .insert((idx_root, value.to_vec(), key), false);
    }

    pub fn index_read_range<'b, R: RangeBounds<&'b [u8]>, F, E>(
        &mut self,
        idx_root: NodePtr,
        range: R,
        mut f: F,
    ) -> Result<(), E>
    where
        for<'local> F: FnMut(LockedTransaction<'local, N>, &[u8], Key) -> Result<bool, E>,
        E: From<TreeError>,
    {
        let range_bound = (range.start_bound().cloned(), range.end_bound().cloned());
        let &mut LockedTransaction {
            ref mut btree,
            ref mut active_transactions,
        } = self;

        // Record range read for conflict detection
        active_transactions.index_range_reads.push((
            idx_root,
            range_bound.0.map(|v| v.to_vec()),
            range_bound.1.map(|v| v.to_vec()),
        ));

        // Collect local index operations for this root
        let local_ops = active_transactions
            .index_ops
            .iter()
            .filter(|((r, _, _), _)| *r == idx_root)
            .map(|((_, v, k), is_insert)| (v.clone(), *k, *is_insert))
            .collect::<Vec<_>>();

        let mut processed_entries = BTreeSet::new();
        let mut early_stop = false;

        // Iterate through btree entries
        btree.index_read_range(
            idx_root,
            range_bound,
            |btree, value, key| -> Result<bool, E> {
                // Check if this entry is locally deleted
                if let Some((_, _, false)) =
                    local_ops.iter().find(|(v, k, _)| v == value && *k == key)
                {
                    // Entry is deleted locally, skip it
                    return Ok(false);
                }

                processed_entries.insert((value.to_vec(), key));

                // Record index read for conflict detection
                active_transactions
                    .index_reads
                    .insert((idx_root, value.to_vec()));

                // Call the user callback
                let me = LockedTransaction {
                    btree,
                    active_transactions,
                };
                let should_stop = f(me, value, key)?;

                if should_stop {
                    early_stop = true;
                }

                Ok(should_stop)
            },
        )?;

        // Process locally inserted entries in range (if not stopped early)
        if !early_stop {
            for (value, key, is_insert) in local_ops {
                if is_insert && range_bound.contains(&value.as_slice()) {
                    // Check if we already processed this entry from btree
                    if !processed_entries.contains(&(value.clone(), key)) {
                        let me = LockedTransaction {
                            btree,
                            active_transactions,
                        };
                        if f(me, &value, key)? {
                            break; // Stop early
                        }
                    }
                }
            }
        }

        Ok(())
    }

    // ── Structural operations ──────────────────────────────────────

    pub fn init_tree(&mut self) -> Result<NodePtr, TreeError> {
        let page = self.btree.init_tree()?;
        self.active_transactions.deferred_init_trees.push(page);
        Ok(page)
    }

    pub fn init_index(&mut self) -> Result<NodePtr, TreeError> {
        let page = self.btree.init_index()?;
        self.active_transactions.deferred_init_indexes.push(page);
        Ok(page)
    }

    pub fn free_tree(&mut self, root: NodePtr) {
        self.active_transactions.deferred_free_trees.push(root)
    }

    pub fn free_index_tree(&mut self, root: NodePtr) {
        self.active_transactions
            .deferred_free_index_trees
            .push(root)
    }
}

impl<'a, const N: usize> Transaction<'a, N> {
    pub fn with_lock<T>(&mut self, f: impl FnOnce(LockedTransaction<'_, N>) -> T) -> T {
        let mut inner = self.store.lock().unwrap();
        let &mut TransactionStoreInner {
            ref mut btree,
            ref mut active_transactions,
            ..
        } = inner.deref_mut();
        let other_max_write_keys = active_transactions
            .iter()
            .filter(|(tx_id, _)| **tx_id != self.tx_id)
            .flat_map(|(_, op)| {
                op.writes
                    .iter()
                    .filter(|(_, value)| value.is_some())
                    .map(|((root, key), _)| (*root, *key))
            })
            .fold(BTreeMap::new(), |mut acc, (root, key)| {
                acc.entry(root)
                    .and_modify(|max_key| {
                        if key > *max_key {
                            *max_key = key;
                        }
                    })
                    .or_insert(key);
                acc
            });
        let op = active_transactions.get_mut(&self.tx_id).unwrap();
        let locked_tx = LockedTransaction {
            btree,
            active_transactions: op,
        };
        f(locked_tx)
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
    use std::{fs, ops::RangeBounds};
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

    fn with_locked<const N: usize, T>(
        tx: &mut Transaction<'_, N>,
        f: impl FnOnce(&mut LockedTransaction<'_, N>) -> T,
    ) -> T {
        tx.with_lock(|mut locked| f(&mut locked))
    }

    fn read<const N: usize>(
        tx: &mut Transaction<'_, N>,
        root: NodePtr,
        key: Key,
    ) -> Result<Option<Vec<u8>>, TreeError> {
        with_locked(tx, |tx| {
            tx.read(root, key).map(|v| v.map(|v| v.into_owned()))
        })
    }

    fn write<const N: usize>(tx: &mut Transaction<'_, N>, root: NodePtr, key: Key, value: Vec<u8>) {
        with_locked(tx, |tx| tx.write(root, key, value));
    }

    fn remove<const N: usize>(tx: &mut Transaction<'_, N>, root: NodePtr, key: Key) {
        with_locked(tx, |tx| tx.remove(root, key));
    }

    fn insert<const N: usize>(
        tx: &mut Transaction<'_, N>,
        root: NodePtr,
        value: Vec<u8>,
    ) -> Result<Key, TreeError> {
        with_locked(tx, |tx| tx.insert(root, value))
    }

    fn read_range<const N: usize, R: RangeBounds<Key>>(
        tx: &mut Transaction<'_, N>,
        root: NodePtr,
        range: R,
    ) -> Result<Vec<(Key, Vec<u8>)>, TreeError> {
        let mut results = Vec::new();
        with_locked(tx, |tx| {
            tx.read_range(root, range, |_, key, value| {
                results.push((key, value.to_vec()));
                Ok::<_, TreeError>(false)
            })
        })?;
        Ok(results)
    }

    fn remove_range<const N: usize, R: RangeBounds<Key>>(
        tx: &mut Transaction<'_, N>,
        root: NodePtr,
        range: R,
    ) -> Result<(), TreeError> {
        with_locked(tx, |tx| tx.remove_range(root, range))
    }

    fn remove_range_where<const N: usize, R: RangeBounds<Key>, F, E>(
        tx: &mut Transaction<'_, N>,
        root: NodePtr,
        range: R,
        f: F,
    ) -> Result<(), E>
    where
        F: for<'local> FnMut(LockedTransaction<'local, N>, Key, &[u8]) -> Result<(bool, bool), E>,
        E: From<TreeError>,
    {
        with_locked(tx, |tx| tx.remove_range_where(root, range, f))
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
        let mut tx = store.begin_transaction();

        let key = 1u64;
        let value = vec![1, 2, 3, 4, 5];

        write(&mut tx, root, key, value.clone());
        let read_value = read(&mut tx, root, key).unwrap();

        assert_eq!(read_value, Some(value));
    }

    #[test]
    fn test_read_nonexistent_key() {
        let (store, root, _temp) = setup_transaction_store();
        let mut tx = store.begin_transaction();

        let read_value = read(&mut tx, root, 999).unwrap();
        assert_eq!(read_value, None);
    }

    #[test]
    fn test_write_multiple_keys() {
        let (store, root, _temp) = setup_transaction_store();
        let mut tx = store.begin_transaction();

        write(&mut tx, root, 1, vec![1, 2, 3]);
        write(&mut tx, root, 2, vec![4, 5, 6]);
        write(&mut tx, root, 3, vec![7, 8, 9]);

        assert_eq!(read(&mut tx, root, 1).unwrap(), Some(vec![1, 2, 3]));
        assert_eq!(read(&mut tx, root, 2).unwrap(), Some(vec![4, 5, 6]));
        assert_eq!(read(&mut tx, root, 3).unwrap(), Some(vec![7, 8, 9]));
    }

    #[test]
    fn test_overwrite_value() {
        let (store, root, _temp) = setup_transaction_store();
        let mut tx = store.begin_transaction();

        let key = 1u64;
        write(&mut tx, root, key, vec![1, 2, 3]);
        assert_eq!(read(&mut tx, root, key).unwrap(), Some(vec![1, 2, 3]));

        write(&mut tx, root, key, vec![4, 5, 6]);
        assert_eq!(read(&mut tx, root, key).unwrap(), Some(vec![4, 5, 6]));
    }

    #[test]
    fn test_commit_single_transaction() {
        let (store, root, _temp) = setup_transaction_store();
        let mut tx = store.begin_transaction();

        write(&mut tx, root, 1, vec![1, 2, 3]);
        let result = tx.commit();
        assert!(result.is_ok());
    }

    #[test]
    fn test_concurrent_reads_no_conflict() {
        let (store, root, _temp) = setup_transaction_store();

        let mut tx1 = store.begin_transaction();
        let mut tx2 = store.begin_transaction();

        write(&mut tx1, root, 1, vec![1, 2, 3]);
        write(&mut tx2, root, 2, vec![4, 5, 6]);

        assert!(tx1.commit().is_ok());
        assert!(tx2.commit().is_ok());
    }

    #[test]
    fn test_write_write_conflict() {
        let (store, root, _temp) = setup_transaction_store();

        let mut tx1 = store.begin_transaction();
        let mut tx2 = store.begin_transaction();

        write(&mut tx1, root, 1, vec![1, 2, 3]);
        write(&mut tx2, root, 1, vec![4, 5, 6]);

        // Both transactions write to the same key, so first one to commit should fail
        // because there's an active transaction with conflicting writes
        let result = tx1.commit();
        assert!(result.is_err());
    }

    #[test]
    fn test_read_write_conflict() {
        let (store, root, _temp) = setup_transaction_store();

        let mut tx1 = store.begin_transaction();
        let mut tx2 = store.begin_transaction();

        read(&mut tx1, root, 1).unwrap();
        write(&mut tx2, root, 1, vec![4, 5, 6]);

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

        let mut tx1 = store.begin_transaction();
        let mut tx2 = store.begin_transaction();

        write(&mut tx1, root, 1, vec![1, 2, 3]);
        read(&mut tx2, root, 1).unwrap();

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

        let mut tx1 = store.begin_transaction();
        let mut tx2 = store.begin_transaction();
        let mut tx3 = store.begin_transaction();

        write(&mut tx1, root, 1, vec![1, 2, 3]);
        write(&mut tx2, root, 2, vec![4, 5, 6]);
        write(&mut tx3, root, 3, vec![7, 8, 9]);

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

        let mut tx1 = store.begin_transaction();
        let mut tx2 = store.begin_transaction();

        write(&mut tx1, root, 1, vec![1, 2, 3]);
        write(&mut tx2, root, 1, vec![4, 5, 6]);

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
            let mut tx = store.begin_transaction();
            write(&mut tx, root, 1, vec![1, 2, 3]);
            assert!(tx.commit().is_ok());
        }

        // Second transaction
        {
            let mut tx = store.begin_transaction();
            write(&mut tx, root, 2, vec![4, 5, 6]);
            assert!(tx.commit().is_ok());
        }
    }

    #[test]
    fn test_remove_key() {
        let (store, root, _temp) = setup_transaction_store();

        // First, insert a key
        {
            let mut tx = store.begin_transaction();
            write(&mut tx, root, 1, vec![1, 2, 3]);
            assert!(tx.commit().is_ok());
        }

        // Then remove it
        {
            let mut tx = store.begin_transaction();
            remove(&mut tx, root, 1);
            assert!(tx.commit().is_ok());
        }

        // Verify it's gone
        {
            let mut tx = store.begin_transaction();
            assert_eq!(read(&mut tx, root, 1).unwrap(), None);
        }
    }

    #[test]
    fn test_remove_in_transaction() {
        let (store, root, _temp) = setup_transaction_store();
        let mut tx = store.begin_transaction();

        write(&mut tx, root, 1, vec![1, 2, 3]);
        assert_eq!(read(&mut tx, root, 1).unwrap(), Some(vec![1, 2, 3]));

        remove(&mut tx, root, 1);
        assert_eq!(read(&mut tx, root, 1).unwrap(), None);
    }

    #[test]
    fn test_remove_range_where_calls_callback_in_key_order() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let mut tx = store.begin_transaction();
            write(&mut tx, root, 1, b"one".to_vec());
            write(&mut tx, root, 3, b"three".to_vec());
            write(&mut tx, root, 5, b"five".to_vec());
            tx.commit().unwrap();
        }

        let mut tx = store.begin_transaction();
        write(&mut tx, root, 2, b"two".to_vec());
        write(&mut tx, root, 4, b"four".to_vec());
        write(&mut tx, root, 3, b"THREE".to_vec());

        let mut seen = Vec::new();
        remove_range_where(&mut tx, root, 1..=5, |_, key, value| {
            seen.push((key, value.to_vec()));
            Ok::<_, TreeError>((false, false))
        })
        .unwrap();

        assert_eq!(
            seen,
            vec![
                (1, b"one".to_vec()),
                (2, b"two".to_vec()),
                (3, b"THREE".to_vec()),
                (4, b"four".to_vec()),
                (5, b"five".to_vec()),
            ]
        );
    }

    #[test]
    fn test_remove_range_where_sees_ongoing_transaction_updates() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let mut tx = store.begin_transaction();
            write(&mut tx, root, 1, b"one".to_vec());
            write(&mut tx, root, 3, b"three".to_vec());
            write(&mut tx, root, 5, b"five".to_vec());
            tx.commit().unwrap();
        }

        let mut tx = store.begin_transaction();
        write(&mut tx, root, 2, b"two".to_vec());
        write(&mut tx, root, 4, b"four".to_vec());
        remove(&mut tx, root, 5);

        let mut seen = Vec::new();
        remove_range_where(&mut tx, root, 1..=5, |mut locked, key, value| {
            seen.push((key, value.to_vec()));
            if key == 2 {
                locked.write(root, 4, b"FOUR".to_vec());
            }
            if key == 3 {
                locked.write(root, 5, b"FIVE".to_vec());
            }
            Ok::<_, TreeError>((false, false))
        })
        .unwrap();

        assert_eq!(
            seen,
            vec![
                (1, b"one".to_vec()),
                (2, b"two".to_vec()),
                (3, b"three".to_vec()),
                (4, b"FOUR".to_vec()),
                (5, b"FIVE".to_vec()),
            ]
        );
    }

    #[test]
    fn test_data_persists_after_commit() {
        let (store, root, _temp) = setup_transaction_store();

        // Write in first transaction
        {
            let mut tx = store.begin_transaction();
            write(&mut tx, root, 1, vec![1, 2, 3]);
            write(&mut tx, root, 2, vec![4, 5, 6]);
            assert!(tx.commit().is_ok());
        }

        // Read in second transaction to verify persistence
        {
            let mut tx = store.begin_transaction();
            assert_eq!(read(&mut tx, root, 1).unwrap(), Some(vec![1, 2, 3]));
            assert_eq!(read(&mut tx, root, 2).unwrap(), Some(vec![4, 5, 6]));
        }
    }

    #[test]
    fn test_commit_after_conflict_fails() {
        let (store, root, _temp) = setup_transaction_store();

        let mut tx1 = store.begin_transaction();
        let mut tx2 = store.begin_transaction();

        write(&mut tx1, root, 1, vec![1, 2, 3]);
        write(&mut tx2, root, 1, vec![4, 5, 6]);

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
            let mut tx = store.begin_transaction();
            write(&mut tx, root, 1, vec![1, 2, 3]);
            assert!(tx.commit().is_ok());
        }

        // Second transaction reads and writes to different key
        {
            let mut tx = store.begin_transaction();
            assert_eq!(read(&mut tx, root, 1).unwrap(), Some(vec![1, 2, 3]));
            write(&mut tx, root, 2, vec![4, 5, 6]);
            assert!(tx.commit().is_ok());
        }
    }

    #[test]
    fn test_multiple_reads_same_key() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let mut tx = store.begin_transaction();
            write(&mut tx, root, 1, vec![1, 2, 3]);
            assert!(tx.commit().is_ok());
        }

        let mut tx = store.begin_transaction();
        assert_eq!(read(&mut tx, root, 1).unwrap(), Some(vec![1, 2, 3]));
        assert_eq!(read(&mut tx, root, 1).unwrap(), Some(vec![1, 2, 3]));
        assert_eq!(read(&mut tx, root, 1).unwrap(), Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_write_after_read_same_transaction() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let mut tx = store.begin_transaction();
            write(&mut tx, root, 1, vec![1, 2, 3]);
            assert!(tx.commit().is_ok());
        }

        let mut tx = store.begin_transaction();
        assert_eq!(read(&mut tx, root, 1).unwrap(), Some(vec![1, 2, 3]));
        write(&mut tx, root, 1, vec![4, 5, 6]);
        assert_eq!(read(&mut tx, root, 1).unwrap(), Some(vec![4, 5, 6]));
        assert!(tx.commit().is_ok());
    }

    #[test]
    fn test_complex_conflict_scenario() {
        let (store, root, _temp) = setup_transaction_store();

        // Setup: write initial values
        {
            let mut tx = store.begin_transaction();
            write(&mut tx, root, 1, vec![1]);
            write(&mut tx, root, 2, vec![2]);
            write(&mut tx, root, 3, vec![3]);
            assert!(tx.commit().is_ok());
        }

        let mut tx1 = store.begin_transaction();
        let mut tx2 = store.begin_transaction();
        let mut tx3 = store.begin_transaction();

        // tx1 reads 1, writes 4
        read(&mut tx1, root, 1).unwrap();
        write(&mut tx1, root, 4, vec![4, 4]);

        // tx2 reads 2, writes 3
        read(&mut tx2, root, 2).unwrap();
        write(&mut tx2, root, 3, vec![3, 3]);

        // tx3 writes 5
        write(&mut tx3, root, 5, vec![5, 5]);

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
            let mut tx = store.begin_transaction();
            write(&mut tx, root, 1, vec![1, 2, 3]);
            assert!(tx.commit().is_ok());
        }

        let mut tx1 = store.begin_transaction();
        let mut tx2 = store.begin_transaction();

        read(&mut tx1, root, 1).unwrap();
        remove(&mut tx2, root, 1);

        // tx2 removes key 1 that tx1 read -> conflict
        assert!(tx2.commit().is_err());
    }

    #[test]
    fn test_read_range_basic() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let mut tx = store.begin_transaction();
            write(&mut tx, root, 1, vec![1]);
            write(&mut tx, root, 3, vec![3]);
            write(&mut tx, root, 5, vec![5]);
            write(&mut tx, root, 7, vec![7]);
            write(&mut tx, root, 9, vec![9]);
            assert!(tx.commit().is_ok());
        }

        let mut tx = store.begin_transaction();
        let results = read_range(&mut tx, root, 2..=7).unwrap();
        assert_eq!(results, vec![(3, vec![3]), (5, vec![5]), (7, vec![7])]);
    }

    #[test]
    fn test_read_range_with_local_writes() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let mut tx = store.begin_transaction();
            write(&mut tx, root, 1, vec![1]);
            write(&mut tx, root, 5, vec![5]);
            assert!(tx.commit().is_ok());
        }

        let mut tx = store.begin_transaction();
        write(&mut tx, root, 3, vec![3]);
        let results = read_range(&mut tx, root, 1..=5).unwrap();
        assert_eq!(results, vec![(1, vec![1]), (3, vec![3]), (5, vec![5])]);
    }

    #[test]
    fn test_read_range_with_local_removes() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let mut tx = store.begin_transaction();
            write(&mut tx, root, 1, vec![1]);
            write(&mut tx, root, 3, vec![3]);
            write(&mut tx, root, 5, vec![5]);
            assert!(tx.commit().is_ok());
        }

        let mut tx = store.begin_transaction();
        remove(&mut tx, root, 3);
        let results = read_range(&mut tx, root, 1..=5).unwrap();
        assert_eq!(results, vec![(1, vec![1]), (5, vec![5])]);
    }

    #[test]
    fn test_read_range_empty() {
        let (store, root, _temp) = setup_transaction_store();

        let mut tx = store.begin_transaction();
        let results = read_range(&mut tx, root, 1..=10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_remove_range_basic() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let mut tx = store.begin_transaction();
            write(&mut tx, root, 1, vec![1]);
            write(&mut tx, root, 3, vec![3]);
            write(&mut tx, root, 5, vec![5]);
            write(&mut tx, root, 7, vec![7]);
            assert!(tx.commit().is_ok());
        }

        {
            let mut tx = store.begin_transaction();
            remove_range(&mut tx, root, 2..=5).unwrap();
            assert!(tx.commit().is_ok());
        }

        // Verify only keys outside range remain
        {
            let mut tx = store.begin_transaction();
            assert_eq!(read(&mut tx, root, 1).unwrap(), Some(vec![1]));
            assert_eq!(read(&mut tx, root, 3).unwrap(), None);
            assert_eq!(read(&mut tx, root, 5).unwrap(), None);
            assert_eq!(read(&mut tx, root, 7).unwrap(), Some(vec![7]));
        }
    }

    #[test]
    fn test_remove_range_with_local_writes() {
        let (store, root, _temp) = setup_transaction_store();

        let mut tx = store.begin_transaction();
        write(&mut tx, root, 1, vec![1]);
        write(&mut tx, root, 3, vec![3]);
        write(&mut tx, root, 5, vec![5]);
        remove_range(&mut tx, root, 2..=4).unwrap();

        // Key 3 should be removed, 1 and 5 should remain
        assert_eq!(read(&mut tx, root, 1).unwrap(), Some(vec![1]));
        assert_eq!(read(&mut tx, root, 3).unwrap(), None);
        assert_eq!(read(&mut tx, root, 5).unwrap(), Some(vec![5]));
    }

    #[test]
    fn test_read_range_conflict_with_write() {
        let (store, root, _temp) = setup_transaction_store();

        let mut tx1 = store.begin_transaction();
        let mut tx2 = store.begin_transaction();

        // tx1 reads a range
        read_range(&mut tx1, root, 1..=10).unwrap();

        // tx2 writes to a key within that range
        write(&mut tx2, root, 5, vec![5]);

        // tx2 should conflict because tx1 has a range read covering key 5
        assert!(tx2.commit().is_err());
    }

    #[test]
    fn test_read_range_no_conflict_outside() {
        let (store, root, _temp) = setup_transaction_store();

        let mut tx1 = store.begin_transaction();
        let mut tx2 = store.begin_transaction();

        // tx1 reads a range
        read_range(&mut tx1, root, 1..=10).unwrap();

        // tx2 writes to a key outside that range
        write(&mut tx2, root, 20, vec![20]);

        // No conflict expected
        assert!(tx2.commit().is_ok());
    }

    #[test]
    fn test_remove_range_conflict() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let mut tx = store.begin_transaction();
            write(&mut tx, root, 3, vec![3]);
            assert!(tx.commit().is_ok());
        }

        let mut tx1 = store.begin_transaction();
        let mut tx2 = store.begin_transaction();

        remove_range(&mut tx1, root, 1..=5).unwrap();
        write(&mut tx2, root, 4, vec![4]);

        // tx2 writes to key 4, which is within tx1's range read -> conflict
        assert!(tx2.commit().is_err());
    }

    #[test]
    fn test_write_conflicts_with_range_read() {
        let (store, root, _temp) = setup_transaction_store();

        let mut tx1 = store.begin_transaction();
        let mut tx2 = store.begin_transaction();

        read_range(&mut tx1, root, 1..=10).unwrap();
        write(&mut tx2, root, 5, vec![5]);

        // tx1 commits: its range_reads cover 1..=10, tx2 writes to 5 which is in range -> conflict
        assert!(tx1.commit().is_err());
    }

    #[test]
    fn test_insert_returns_unique_key() {
        let (store, root, _temp) = setup_transaction_store();

        let mut tx = store.begin_transaction();
        let k1 = insert(&mut tx, root, vec![1]).unwrap();
        let k2 = insert(&mut tx, root, vec![2]).unwrap();
        let k3 = insert(&mut tx, root, vec![3]).unwrap();

        assert_ne!(k1, k2);
        assert_ne!(k2, k3);
        assert_eq!(read(&mut tx, root, k1).unwrap(), Some(vec![1]));
        assert_eq!(read(&mut tx, root, k2).unwrap(), Some(vec![2]));
        assert_eq!(read(&mut tx, root, k3).unwrap(), Some(vec![3]));
    }

    #[test]
    fn test_insert_after_existing_keys() {
        let (store, root, _temp) = setup_transaction_store();

        {
            let mut tx = store.begin_transaction();
            write(&mut tx, root, 10, vec![10]);
            assert!(tx.commit().is_ok());
        }

        let mut tx = store.begin_transaction();
        let key = insert(&mut tx, root, vec![42]).unwrap();
        assert!(key > 10);
        assert_eq!(read(&mut tx, root, key).unwrap(), Some(vec![42]));
        assert!(tx.commit().is_ok());
    }

    #[test]
    fn test_insert_concurrent_no_conflict() {
        let (store, root, _temp) = setup_transaction_store();

        let mut tx1 = store.begin_transaction();
        let mut tx2 = store.begin_transaction();

        let k1 = insert(&mut tx1, root, vec![1]).unwrap();
        let k2 = insert(&mut tx2, root, vec![2]).unwrap();

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
            let mut tx = store.begin_transaction();
            key = insert(&mut tx, root, vec![99]).unwrap();
            assert!(tx.commit().is_ok());
        }

        {
            let mut tx = store.begin_transaction();
            assert_eq!(read(&mut tx, root, key).unwrap(), Some(vec![99]));
        }
    }
}
