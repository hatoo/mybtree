use rkyv::Archived;
use rkyv::rancor::Error;
use std::collections::BTreeMap;
use std::ops::RangeBounds;

use crate::pager::Pager;
use crate::types::{Internal, Key, Leaf, Node, NodePtr, ROOT_PAGE_NUM, Value};
use crate::util::{is_overlap, split_internal, split_leaf};

/// Find the child page for `key` in an archived internal node.
fn find_child_page(internal: &Archived<Internal>, key: Key) -> Option<NodePtr> {
    let index = match internal.kv.binary_search_by_key(&key, |t| t.0.to_native()) {
        Ok(i) | Err(i) => i,
    };
    internal.kv.get(index).map(|t| t.1.to_native())
}

pub struct Btree {
    pub pager: Pager,
}

impl Btree {
    pub fn new(pager: Pager) -> Self {
        Btree { pager }
    }

    pub fn init(&mut self) -> Result<(), Error> {
        self.pager.next_page_num = 1;

        let root_leaf = Leaf { kv: vec![] };
        self.pager
            .write_node(ROOT_PAGE_NUM, &Node::Leaf(root_leaf))?;
        Ok(())
    }

    fn search(&mut self, key: Key) -> Result<Option<Vec<NodePtr>>, Error> {
        let mut current = ROOT_PAGE_NUM;
        let mut path = vec![];

        enum SearchResult {
            Found,
            Next(NodePtr),
            NotFound,
        }

        loop {
            path.push(current);

            let result = self
                .pager
                .read_node(current, |archived_node| match archived_node {
                    Archived::<Node>::Leaf(leaf) => {
                        match leaf.kv.binary_search_by_key(&key, |t| t.0.to_native()) {
                            Ok(_) => SearchResult::Found,
                            Err(_) => SearchResult::NotFound,
                        }
                    }
                    Archived::<Node>::Internal(internal) => match find_child_page(internal, key) {
                        Some(next) => SearchResult::Next(next),
                        None => SearchResult::NotFound,
                    },
                })?;

            match result {
                SearchResult::Found => return Ok(Some(path)),
                SearchResult::Next(next) => current = next,
                SearchResult::NotFound => return Ok(None),
            }
        }
    }

    pub fn insert(&mut self, key: Key, value: Vec<u8>) -> Result<Option<Vec<u8>>, Error> {
        let mut current = ROOT_PAGE_NUM;
        let mut path = vec![];

        enum NextNode {
            Leaf,
            Next(NodePtr),
            NeedAlloc,
        }

        loop {
            path.push(current);

            let next = self
                .pager
                .read_node(current, |archived_node| match archived_node {
                    Archived::<Node>::Leaf(_) => NextNode::Leaf,
                    Archived::<Node>::Internal(internal) => match find_child_page(internal, key) {
                        Some(next) => NextNode::Next(next),
                        None => NextNode::NeedAlloc,
                    },
                })?;

            match next {
                NextNode::Leaf => {
                    let Node::Leaf(mut leaf) = self.pager.owned_node(current)? else {
                        panic!("Expected leaf node");
                    };
                    match leaf.kv.binary_search_by_key(&key, |t| t.0) {
                        Ok(index) => {
                            let old_value_entry = leaf.kv[index].1.clone();
                            let old_bytes = self.resolve_value(&old_value_entry)?;

                            let new_value = self.make_value(value)?;
                            let grew = match (&old_value_entry, &new_value) {
                                (Value::Inline(old), Value::Inline(new_v)) => {
                                    new_v.len() > old.len()
                                }
                                (Value::Overflow { .. }, Value::Inline(_)) => true,
                                _ => false,
                            };
                            leaf.kv[index].1 = new_value;

                            if grew {
                                self.split_insert(&path, &Node::Leaf(leaf))?;
                            } else {
                                self.merge_insert(&path, &Node::Leaf(leaf))?;
                            }
                            // TODO: free old overflow pages
                            return Ok(Some(old_bytes));
                        }
                        Err(index) => {
                            let new_value = self.make_value(value)?;
                            leaf.kv.insert(index, (key, new_value));
                            self.split_insert(&path, &Node::Leaf(leaf))?;
                            return Ok(None);
                        }
                    }
                }
                NextNode::Next(next_page) => {
                    current = next_page;
                }
                NextNode::NeedAlloc => {
                    let Node::Internal(mut internal) = self.pager.owned_node(current)? else {
                        panic!("Expected internal node");
                    };

                    if internal.kv.is_empty() {
                        let new_leaf_page = self.pager.next_page_num();
                        let new_value = self.make_value(value)?;
                        let new_leaf = Leaf {
                            kv: vec![(key, new_value)],
                        };
                        self.pager
                            .write_node(new_leaf_page, &Node::Leaf(new_leaf))?;

                        internal.kv.push((key, new_leaf_page));
                        self.pager.write_node(current, &Node::Internal(internal))?;

                        return Ok(None);
                    } else {
                        let last = internal.kv.last_mut().unwrap();
                        last.0 = key;
                        let next = last.1;
                        self.pager.write_node(current, &Node::Internal(internal))?;
                        current = next;
                    }
                }
            }
        }
    }

    fn split_insert(&mut self, path: &[NodePtr], insert: &Node) -> Result<(), Error> {
        let buffer = rkyv::to_bytes(insert)?;

        let page = *path.last().unwrap();
        let parents = &path[..path.len() - 1];

        if buffer.len() <= self.pager.page_content_size() {
            self.pager.write_buffer(page, buffer)?;
            return Ok(());
        }

        let page_content_size = self.pager.page_content_size();
        let keyed_nodes: Vec<(Key, Node)> = match insert {
            Node::Leaf(leaf) => split_leaf(leaf.kv.clone(), page_content_size)?
                .into_iter()
                .map(|kv| (kv.last().unwrap().0, Node::Leaf(Leaf { kv })))
                .collect(),
            Node::Internal(internal) => split_internal(internal.kv.clone(), page_content_size)?
                .into_iter()
                .map(|kv| (kv.last().unwrap().0, Node::Internal(Internal { kv })))
                .collect(),
        };

        self.write_splits(keyed_nodes, page, parents)
    }

    /// Write split nodes to pages and update the parent.
    /// The last node in `keyed_nodes` is written to `page`; the rest get new pages.
    fn write_splits(
        &mut self,
        mut keyed_nodes: Vec<(Key, Node)>,
        page: NodePtr,
        parents: &[NodePtr],
    ) -> Result<(), Error> {
        if let Some(&parent_page) = parents.last() {
            let (_right_key, right_node) = keyed_nodes.pop().unwrap();
            self.pager.write_node(page, &right_node)?;

            let Node::Internal(mut parent_internal) = self.pager.owned_node(parent_page)? else {
                panic!("Parent is not an internal node");
            };
            let mut kv_map: BTreeMap<_, _> = parent_internal.kv.into_iter().collect();
            for (key, node) in keyed_nodes {
                let new_page = self.pager.next_page_num();
                self.pager.write_node(new_page, &node)?;
                kv_map.insert(key, new_page);
            }
            parent_internal.kv = kv_map.into_iter().collect();
            self.split_insert(parents, &Node::Internal(parent_internal))
        } else {
            let new_entries = keyed_nodes
                .into_iter()
                .map(|(key, node)| {
                    let new_page = self.pager.next_page_num();
                    self.pager.write_node(new_page, &node)?;
                    Ok((key, new_page))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            self.split_insert(
                &[ROOT_PAGE_NUM],
                &Node::Internal(Internal { kv: new_entries }),
            )
        }
    }

    pub fn remove(&mut self, key: Key) -> Result<Option<Vec<u8>>, Error> {
        if let Some(path) = self.search(key)? {
            let leaf_page = *path.last().unwrap();
            let leaf_node = self.pager.owned_node(leaf_page)?;
            let Node::Leaf(mut leaf) = leaf_node else {
                panic!("Expected leaf node");
            };
            match leaf.kv.binary_search_by_key(&key, |t| t.0) {
                Ok(index) => {
                    let old_value_entry = leaf.kv.remove(index).1;
                    self.merge_insert(&path, &Node::Leaf(leaf))?;
                    let old_bytes = self.resolve_value(&old_value_entry)?;
                    // TODO: free overflow pages
                    return Ok(Some(old_bytes));
                }
                Err(_) => {
                    panic!("Key not found in leaf node");
                }
            }
        }
        Ok(None)
    }

    pub fn remove_range(&mut self, range: impl RangeBounds<Key>) -> Result<(), Error> {
        self.remove_range_at(ROOT_PAGE_NUM, &range, 0)
    }

    fn remove_range_at(
        &mut self,
        node_ptr: NodePtr,
        range: &impl RangeBounds<Key>,
        left_key: Key,
    ) -> Result<(), Error> {
        let node = self.pager.owned_node(node_ptr)?;

        match node {
            Node::Leaf(mut leaf) => {
                leaf.kv.retain(|(k, _)| !range.contains(k));
                self.merge_insert(&[node_ptr], &Node::Leaf(leaf))?;
            }
            Node::Internal(internal) => {
                let mut left_key = left_key;
                for (key, ptr) in &internal.kv {
                    if is_overlap(&(left_key..=*key), range) {
                        self.remove_range_at(*ptr, range, left_key)?;
                    }
                    left_key = *key;
                }
            }
        }

        Ok(())
    }

    fn merge_insert(&mut self, path: &[NodePtr], insert: &Node) -> Result<(), Error> {
        let buffer = rkyv::to_bytes(insert)?;

        let page = *path.last().unwrap();

        debug_assert!(buffer.len() <= self.pager.page_content_size());

        if buffer.len() >= self.pager.page_content_size() / 2 || path.len() <= 1 {
            return self.pager.write_buffer(page, buffer);
        }

        // Handle empty leaf: remove from parent entirely
        if let Node::Leaf(leaf) = insert {
            if leaf.kv.is_empty() {
                let parents = &path[..path.len() - 1];
                let parent_page = *parents.last().unwrap();
                let Node::Internal(mut internal) = self.pager.owned_node(parent_page)? else {
                    panic!("Parent is not an internal node");
                };
                internal.kv.retain(|&(_, ptr)| ptr != page);
                return self.merge_insert(parents, &Node::Internal(internal));
            }
        }

        let parents = &path[..path.len() - 1];
        if !self.try_merge_with_left_sibling(page, insert, parents)? {
            self.pager.write_node(page, insert)?;
        }
        Ok(())
    }

    /// Attempt to merge `insert` with its left sibling in the parent.
    /// Returns `true` if the merge was performed, `false` otherwise.
    fn try_merge_with_left_sibling(
        &mut self,
        page: NodePtr,
        insert: &Node,
        parents: &[NodePtr],
    ) -> Result<bool, Error> {
        let parent_page = *parents.last().unwrap();
        let Node::Internal(mut parent_internal) = self.pager.owned_node(parent_page)? else {
            return Ok(false);
        };

        let index = parent_internal
            .kv
            .iter()
            .position(|&(_, ptr)| ptr == page)
            .unwrap();
        if index == 0 {
            return Ok(false);
        }

        let left_sibling_page = parent_internal.kv[index - 1].1;
        let left_sibling_node = self.pager.owned_node(left_sibling_page)?;

        let merged = match (left_sibling_node, insert) {
            (Node::Leaf(mut left), Node::Leaf(right)) => {
                left.kv.extend(right.kv.iter().cloned());
                Node::Leaf(left)
            }
            (Node::Internal(mut left), Node::Internal(right)) => {
                left.kv.extend(right.kv.iter().cloned());
                Node::Internal(left)
            }
            _ => return Ok(false),
        };

        let buffer = rkyv::to_bytes(&merged)?;
        if buffer.len() <= self.pager.page_content_size() {
            self.pager.write_buffer(page, buffer)?;
            parent_internal.kv.remove(index - 1);
            self.merge_insert(parents, &Node::Internal(parent_internal))?;
            // TODO: Add page to free list
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn read<T>(&mut self, key: Key, f: impl FnOnce(Option<&[u8]>) -> T) -> Result<T, Error> {
        enum ReadAction<T> {
            Done(T),
            Next(NodePtr),
            Overflow { start_page: u64, total_len: u64 },
        }

        let mut current = ROOT_PAGE_NUM;

        let mut ff = Some(f);
        let f = &mut ff;

        loop {
            let result: ReadAction<T> =
                self.pager
                    .read_node(current, |archived_node| match archived_node {
                        Archived::<Node>::Leaf(leaf) => {
                            match leaf.kv.binary_search_by_key(&key, |t| t.0.to_native()) {
                                Ok(index) => match &leaf.kv[index].1 {
                                    Archived::<Value>::Inline(data) => {
                                        ReadAction::Done(f.take().unwrap()(Some(data.as_ref())))
                                    }
                                    Archived::<Value>::Overflow {
                                        start_page,
                                        total_len,
                                    } => ReadAction::Overflow {
                                        start_page: start_page.to_native(),
                                        total_len: total_len.to_native(),
                                    },
                                },
                                Err(_) => ReadAction::Done(f.take().unwrap()(None)),
                            }
                        }
                        Archived::<Node>::Internal(internal) => {
                            match find_child_page(internal, key) {
                                Some(next) => ReadAction::Next(next),
                                None => ReadAction::Done(f.take().unwrap()(None)),
                            }
                        }
                    })?;

            match result {
                ReadAction::Done(v) => return Ok(v),
                ReadAction::Next(next) => current = next,
                ReadAction::Overflow {
                    start_page,
                    total_len,
                } => {
                    let data = self.read_overflow(start_page, total_len)?;
                    return Ok(f.take().unwrap()(Some(&data)));
                }
            }
        }
    }

    pub fn read_range<R: RangeBounds<Key>>(
        &mut self,
        range: R,
        mut f: impl FnMut(Key, &[u8]),
    ) -> Result<(), Error> {
        self.read_range_at(ROOT_PAGE_NUM, &range, &mut (f), 0)
    }

    fn read_range_at<R: RangeBounds<Key>>(
        &mut self,
        node_ptr: NodePtr,
        range: &R,
        f: &mut impl FnMut(Key, &[u8]),
        left_key: Key,
    ) -> Result<(), Error> {
        let node = self.pager.owned_node(node_ptr)?;

        match node {
            Node::Leaf(leaf) => {
                for (k, v) in leaf.kv {
                    if range.contains(&k) {
                        match v {
                            Value::Inline(data) => f(k, &data),
                            Value::Overflow {
                                start_page,
                                total_len,
                            } => {
                                let data = self.read_overflow(start_page, total_len)?;
                                f(k, &data);
                            }
                        }
                    }
                }
            }
            Node::Internal(internal) => {
                let mut left_key = left_key;
                for (key, ptr) in &internal.kv {
                    if is_overlap(&(left_key..=*key), range) {
                        self.read_range_at(*ptr, range, f, left_key)?;
                    }
                    left_key = *key;
                }
            }
        }

        Ok(())
    }

    pub fn available_key(&mut self) -> Result<Key, Error> {
        self.pager
            .read_node(ROOT_PAGE_NUM, |archived_node| match archived_node {
                Archived::<Node>::Leaf(leaf) => {
                    if let Some(t) = leaf.kv.last() {
                        t.0.to_native().checked_add(1).unwrap_or(u64::MAX)
                    } else {
                        0
                    }
                }
                Archived::<Node>::Internal(internal) => {
                    if let Some(t) = internal.kv.last() {
                        t.0.to_native().checked_add(1).unwrap_or(u64::MAX)
                    } else {
                        0
                    }
                }
            })
    }

    /// Determine whether a value is too large for inline storage and must use overflow pages.
    fn needs_overflow(&self, value_len: usize) -> bool {
        value_len > self.pager.page_content_size() / 2
    }

    /// Convert a raw value into a `Value`, using overflow pages if necessary.
    fn make_value(&mut self, value: Vec<u8>) -> Result<Value, Error> {
        if self.needs_overflow(value.len()) {
            let total_len = value.len() as u64;
            let start_page = self.write_overflow(&value)?;
            Ok(Value::Overflow {
                start_page,
                total_len,
            })
        } else {
            Ok(Value::Inline(value))
        }
    }

    /// Resolve a `Value` into owned bytes, reading overflow pages if needed.
    fn resolve_value(&mut self, value: &Value) -> Result<Vec<u8>, Error> {
        match value {
            Value::Inline(data) => Ok(data.clone()),
            Value::Overflow {
                start_page,
                total_len,
            } => self.read_overflow(*start_page, *total_len),
        }
    }

    /// Write a large value across one or more overflow pages.
    /// Each overflow page layout: [next_page: u64 LE][data bytes][padding]
    /// Returns the page number of the first overflow page.
    fn write_overflow(&mut self, data: &[u8]) -> Result<u64, Error> {
        let data_per_page = self.overflow_data_per_page();
        let num_pages = (data.len() + data_per_page - 1) / data_per_page;
        assert!(num_pages > 0);

        let pages: Vec<u64> = (0..num_pages).map(|_| self.pager.next_page_num()).collect();

        for (i, &page_num) in pages.iter().enumerate() {
            let start = i * data_per_page;
            let end = std::cmp::min(start + data_per_page, data.len());
            let chunk = &data[start..end];

            let next_page: u64 = if i + 1 < pages.len() {
                pages[i + 1]
            } else {
                u64::MAX
            };

            let mut buffer = vec![0u8; 8 + chunk.len()];
            buffer[..8].copy_from_slice(&next_page.to_le_bytes());
            buffer[8..].copy_from_slice(chunk);
            self.pager.write_raw_page(page_num, &buffer);
        }

        Ok(pages[0])
    }

    /// Read a large value stored across overflow pages.
    fn read_overflow(&mut self, start_page: u64, total_len: u64) -> Result<Vec<u8>, Error> {
        let data_per_page = self.overflow_data_per_page();
        let mut result = Vec::with_capacity(total_len as usize);
        let mut current_page = start_page;
        let mut remaining = total_len as usize;

        while remaining > 0 {
            let buffer = self.pager.read_raw_page(current_page);
            let next_page = u64::from_le_bytes(buffer[..8].try_into().unwrap());
            let chunk_len = std::cmp::min(data_per_page, remaining);
            result.extend_from_slice(&buffer[8..8 + chunk_len]);
            remaining -= chunk_len;
            current_page = next_page;
        }

        Ok(result)
    }

    /// Maximum data bytes per overflow page (page_size minus 8-byte next-page pointer).
    fn overflow_data_per_page(&self) -> usize {
        self.pager.page_size - 8
    }

    #[cfg(test)]
    pub fn debug(&mut self, root: NodePtr, min: Key, max: Key) -> Result<(), Error> {
        let node = self.pager.owned_node(root)?;

        match node {
            Node::Leaf(leaf) => {
                println!("Leaf Node (page {}):", root);
                for (k, v) in leaf.kv {
                    if !(min <= k && k <= max) {
                        panic!("Key {} out of range ({}..={})", k, min, max);
                    }
                    match &v {
                        Value::Inline(data) => {
                            println!("  Key: {}, Value Length: {} (inline)", k, data.len());
                        }
                        Value::Overflow { total_len, .. } => {
                            println!("  Key: {}, Value Length: {} (overflow)", k, total_len);
                        }
                    }
                }
            }
            Node::Internal(internal) => {
                println!("Internal Node (page {}):", root);
                for (k, ptr) in &internal.kv {
                    if !(min <= *k && *k <= max) {
                        panic!("Key {} out of range ({}..={})", k, min, max);
                    }
                    println!("  Key: {}, Child Page: {}", k, ptr);
                }
                let mut left = min;
                for (k, ptr) in &internal.kv {
                    self.debug(*ptr, left, *k)?;
                    left = *k;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use std::collections::HashMap;
    use std::ops::Bound;

    use super::*;

    /// Create an initialized btree backed by a temp file with the given page size.
    fn new_btree(page_size: usize) -> Btree {
        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file, page_size);
        let mut btree = Btree::new(pager);
        btree.init().unwrap();
        btree
    }

    /// Create a btree pre-populated with keys `0..count`, each with a 64-byte value.
    fn build_btree(count: u64) -> Btree {
        let mut btree = new_btree(4096);
        for i in 0..count {
            let mut value = vec![0u8; 64];
            value[0..8].copy_from_slice(&i.to_le_bytes());
            btree.insert(i, value).unwrap();
        }
        btree
    }

    fn read_range_keys<R: RangeBounds<Key>>(btree: &mut Btree, range: R) -> Vec<Key> {
        let mut keys = Vec::new();
        btree.read_range(range, |k, _| keys.push(k)).unwrap();
        keys
    }

    fn read_value(btree: &mut Btree, key: Key) -> Option<Vec<u8>> {
        btree.read(key, |v| v.map(|b| b.to_vec())).unwrap()
    }

    fn assert_read_eq(btree: &mut Btree, key: Key, expected: &[u8]) {
        assert!(
            btree.read(key, |v| v == Some(expected)).unwrap(),
            "Key {} value mismatch",
            key
        );
    }

    fn assert_key_exists(btree: &mut Btree, key: Key) {
        assert!(
            btree.read(key, |v| v.is_some()).unwrap(),
            "Key {} should exist",
            key
        );
    }

    fn assert_key_absent(btree: &mut Btree, key: Key) {
        assert!(
            btree.read(key, |v| v.is_none()).unwrap(),
            "Key {} should be absent",
            key
        );
    }

    fn assert_keys_exist(btree: &mut Btree, range: impl IntoIterator<Item = u64>) {
        for k in range {
            assert_key_exists(btree, k);
        }
    }

    fn assert_keys_absent(btree: &mut Btree, range: impl IntoIterator<Item = u64>) {
        for k in range {
            assert_key_absent(btree, k);
        }
    }

    #[test]
    fn test_insert() {
        let mut btree = new_btree(4096);
        btree.insert(1, b"one".to_vec()).unwrap();
    }

    #[test]
    fn test_read() {
        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file, 4096);
        let mut btree = Btree::new(pager);
        let root_leaf = Leaf {
            kv: vec![(0, Value::Inline(b"zero".to_vec()))],
        };
        btree
            .pager
            .write_node(ROOT_PAGE_NUM, &Node::Leaf(root_leaf))
            .unwrap();
        assert_read_eq(&mut btree, 0, b"zero");
    }

    #[test]
    fn test_insert_and_read() {
        let mut btree = new_btree(4096);
        btree.insert(42, b"forty-two".to_vec()).unwrap();
        assert_read_eq(&mut btree, 42, b"forty-two");
    }

    #[test]
    fn test_insert_multiple_and_read() {
        let mut btree = new_btree(64);
        let mut map = HashMap::new();

        for i in 0u64..256 {
            let value = format!("value-{}", i).as_bytes().to_vec();
            btree.insert(i, value.clone()).unwrap();
            map.insert(i, value);

            for j in 0u64..=i {
                let expected = map.get(&j).unwrap();
                assert_read_eq(&mut btree, j, expected);
            }
        }
    }

    #[test]
    fn test_remove() {
        let mut btree = new_btree(64);

        btree.insert(1, b"one".to_vec()).unwrap();
        btree.insert(2, b"two".to_vec()).unwrap();
        btree.insert(3, b"three".to_vec()).unwrap();

        assert_eq!(btree.remove(2).unwrap(), Some(b"two".to_vec()));
        assert_key_absent(&mut btree, 2);
        assert_read_eq(&mut btree, 1, b"one");
        assert_read_eq(&mut btree, 3, b"three");
        assert_eq!(btree.remove(999).unwrap(), None);
    }

    #[test]
    fn test_remove_seq() {
        const LEN: u64 = 1000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut btree = new_btree(64);

        let mut insert = (0..LEN).collect::<Vec<u64>>();
        insert.shuffle(&mut rng);

        let mut remove = insert.clone();
        remove.shuffle(&mut rng);

        for i in insert {
            btree
                .insert(i, format!("value-{}", i).as_bytes().to_vec())
                .unwrap();
        }

        for i in remove {
            assert_eq!(
                btree.remove(i).unwrap(),
                Some(format!("value-{}", i).as_bytes().to_vec()),
                "Failed to remove key {}",
                i
            );
            assert_key_absent(&mut btree, i);
        }
    }

    #[test]
    fn test_read_range_full() {
        let mut btree = build_btree(200);
        assert_eq!(read_range_keys(&mut btree, 0..=199), (0..200).collect::<Vec<_>>());
    }

    #[test]
    fn test_read_range_exclusive_start() {
        let mut btree = build_btree(200);
        let keys = read_range_keys(&mut btree, (Bound::Excluded(10), Bound::Included(20)));
        assert_eq!(keys, (11..=20).collect::<Vec<_>>());
    }

    #[test]
    fn test_read_range_unbounded_end() {
        let mut btree = build_btree(200);
        assert_eq!(read_range_keys(&mut btree, ..=5), (0..=5).collect::<Vec<_>>());
    }

    #[test]
    fn test_read_range_empty() {
        let mut btree = build_btree(200);
        assert!(read_range_keys(&mut btree, 500..=600).is_empty());
    }

    #[test]
    fn test_read_range_order() {
        let mut btree = build_btree(500);
        assert_eq!(read_range_keys(&mut btree, 123..=321), (123..=321).collect::<Vec<_>>());
    }

    #[test]
    fn test_remove_range_inclusive() {
        let mut btree = build_btree(100);
        btree.remove_range(10..=20).unwrap();

        assert_keys_absent(&mut btree, 10..=20);
        assert_keys_exist(&mut btree, 0..10);
        assert_keys_exist(&mut btree, 21..100);
    }

    #[test]
    fn test_remove_range_exclusive_start() {
        let mut btree = build_btree(100);
        btree
            .remove_range((Bound::Excluded(10), Bound::Included(20)))
            .unwrap();

        assert_key_exists(&mut btree, 10);
        assert_keys_absent(&mut btree, 11..=20);
        assert_keys_exist(&mut btree, 0..10);
        assert_keys_exist(&mut btree, 21..100);
    }

    #[test]
    fn test_remove_range_unbounded_end() {
        let mut btree = build_btree(100);
        btree.remove_range(80..).unwrap();

        assert_keys_absent(&mut btree, 80..100);
        assert_keys_exist(&mut btree, 0..80);
    }

    #[test]
    fn test_remove_range_unbounded_start() {
        let mut btree = build_btree(100);
        btree.remove_range(..=20).unwrap();

        assert_keys_absent(&mut btree, 0..=20);
        assert_keys_exist(&mut btree, 21..100);
    }

    #[test]
    fn test_remove_range_nonexistent() {
        let mut btree = build_btree(100);
        btree.remove_range(200..=300).unwrap();
        assert_keys_exist(&mut btree, 0..100);
    }

    #[test]
    fn test_remove_range_empty() {
        let mut btree = build_btree(100);
        btree.remove_range(50..50).unwrap();
        assert_keys_exist(&mut btree, 0..100);
    }

    #[test]
    fn test_remove_range_all_keys() {
        let mut btree = build_btree(100);
        btree.remove_range(0..=99).unwrap();
        assert_keys_absent(&mut btree, 0..100);
    }

    #[test]
    fn test_remove_range_multiple_calls() {
        let mut btree = build_btree(100);

        btree.remove_range(10..=20).unwrap();
        assert_keys_absent(&mut btree, 10..=20);

        btree.remove_range(50..=60).unwrap();
        assert_keys_absent(&mut btree, 10..=20);
        assert_keys_absent(&mut btree, 50..=60);
        assert_keys_exist(&mut btree, 0..10);
        assert_keys_exist(&mut btree, 21..50);
        assert_keys_exist(&mut btree, 61..100);
    }

    #[test]
    fn test_remove_range_verify_read_range() {
        let mut btree = build_btree(100);
        btree.remove_range(30..=70).unwrap();

        let keys = read_range_keys(&mut btree, 0..=99);
        let mut expected = (0..30).collect::<Vec<_>>();
        expected.extend(71..100);
        assert_eq!(keys, expected);
    }

    #[test]
    fn test_remove_range_large_dataset() {
        let mut btree = build_btree(1000);
        btree.remove_range(250..=750).unwrap();

        assert_keys_absent(&mut btree, 250..=750);
        assert_keys_exist(&mut btree, 0..250);
        assert_keys_exist(&mut btree, 751..1000);
    }

    #[test]
    fn test_big_value_insert_and_read() {
        let mut btree = new_btree(256);
        let big_value = vec![42u8; 4096];
        btree.insert(1, big_value.clone()).unwrap();
        assert_eq!(read_value(&mut btree, 1), Some(big_value));
    }

    #[test]
    fn test_big_value_with_small_values() {
        let mut btree = new_btree(256);

        btree.insert(0, b"small-before".to_vec()).unwrap();
        let big_value = vec![42u8; 4096];
        btree.insert(1, big_value.clone()).unwrap();
        btree.insert(2, b"small-after".to_vec()).unwrap();

        assert_read_eq(&mut btree, 0, b"small-before");
        assert_eq!(read_value(&mut btree, 1), Some(big_value));
        assert_read_eq(&mut btree, 2, b"small-after");
    }

    #[test]
    fn test_big_value_update() {
        let mut btree = new_btree(256);

        let big_value = vec![42u8; 4096];
        btree.insert(1, big_value.clone()).unwrap();

        let bigger_value = vec![99u8; 8192];
        let old = btree.insert(1, bigger_value.clone()).unwrap();
        assert_eq!(old, Some(big_value));
        assert_eq!(read_value(&mut btree, 1), Some(bigger_value));
    }

    #[test]
    fn test_big_value_remove() {
        let mut btree = new_btree(256);

        let big_value = vec![42u8; 4096];
        btree.insert(1, big_value.clone()).unwrap();

        assert_eq!(btree.remove(1).unwrap(), Some(big_value));
        assert_key_absent(&mut btree, 1);
    }

    #[test]
    fn test_big_value_read_range() {
        let mut btree = new_btree(256);

        btree.insert(0, b"small".to_vec()).unwrap();
        let big_value = vec![42u8; 2048];
        btree.insert(1, big_value.clone()).unwrap();
        btree.insert(2, b"also-small".to_vec()).unwrap();

        let mut results = Vec::new();
        btree
            .read_range(0..=2, |k, v| results.push((k, v.to_vec())))
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], (0, b"small".to_vec()));
        assert_eq!(results[1], (1, big_value));
        assert_eq!(results[2], (2, b"also-small".to_vec()));
    }

    #[test]
    fn test_multiple_big_values() {
        let mut btree = new_btree(256);

        let mut expected = HashMap::new();
        for i in 0u64..20 {
            let big_value = vec![i as u8; 1024 + (i as usize * 100)];
            btree.insert(i, big_value.clone()).unwrap();
            expected.insert(i, big_value);
        }

        for (key, value) in &expected {
            assert_eq!(
                read_value(&mut btree, *key).as_ref(),
                Some(value),
                "Mismatch at key {}",
                key
            );
        }
    }

    #[test]
    fn test_big_value_replace_with_small() {
        let mut btree = new_btree(256);

        let big_value = vec![42u8; 4096];
        btree.insert(1, big_value.clone()).unwrap();

        let old = btree.insert(1, b"tiny".to_vec()).unwrap();
        assert_eq!(old, Some(big_value));
        assert_read_eq(&mut btree, 1, b"tiny");
    }
}
