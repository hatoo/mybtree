use rkyv::Archived;
use rkyv::rancor::{Error, fail};
use std::collections::BTreeMap;
use std::ops::{Bound, RangeBounds};

use crate::pager::Pager;
use crate::types::{
    FREE_LIST_PAGE_NUM, IndexInternal, IndexLeaf, IndexNode, Internal, Key, Leaf, Node, NodePtr,
    Value,
};
use crate::util::{is_overlap, split_index_internal, split_index_leaf, split_internal, split_leaf};

#[derive(Debug, thiserror::Error)]
pub enum TreeError {
    #[error("unexpected node type: expected {expected}")]
    UnexpectedNodeType { expected: &'static str },
}


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
    /// Create a new `Btree` backed by the given [`Pager`].
    pub fn new(pager: Pager) -> Self {
        Btree { pager }
    }

    /// Flush all dirty cached pages to disk.
    pub fn flush(&mut self) -> Result<(), std::io::Error> {
        self.pager.flush()
    }

    /// Initialize a fresh database file, setting up the free list.
    /// The pager is reset and the first page is allocated for the free list head.
    pub fn init(&mut self) -> Result<(), Error> {
        self.pager.init()?;

        assert!(self.pager.next_page_num() == 0);

        // Initialize free list head to u64::MAX (empty)
        self.write_free_list_head(u64::MAX)?;
        Ok(())
    }

    /// Initialize a new tree at the specified page, writing an empty leaf node as its root.
    /// The page is allocated via [`alloc_page`] if needed; the caller should use the
    /// returned page number as the root for subsequent operations.
    pub fn init_tree(&mut self) -> Result<NodePtr, Error> {
        let page = self.alloc_page()?;
        let root_leaf = Leaf { kv: vec![] };
        self.pager.write_node(page, &Node::Leaf(root_leaf))?;
        Ok(page)
    }

    /// Free all pages belonging to the tree rooted at `root`, including
    /// overflow pages. The root page itself is also freed.
    pub fn free_tree(&mut self, root: NodePtr) -> Result<(), Error> {
        let node = self.pager.owned_node(root)?;
        match node {
            Node::Leaf(leaf) => {
                for (_, v) in &leaf.kv {
                    self.free_value_pages(v)?;
                }
            }
            Node::Internal(internal) => {
                for (_, ptr) in &internal.kv {
                    self.free_tree(*ptr)?;
                }
            }
        }
        self.free_page(root)
    }

    /// Insert or update a key-value pair in the tree rooted at `root`.
    /// Returns the previous value if the key already existed, or `None` for a new key.
    /// Large values are automatically stored in overflow pages.
    pub fn insert(
        &mut self,
        root: NodePtr,
        key: Key,
        value: Vec<u8>,
    ) -> Result<Option<Vec<u8>>, Error> {
        let mut current = root;
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
                        fail!(TreeError::UnexpectedNodeType { expected: "leaf" });
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
                                self.split_insert(root, &path, Node::Leaf(leaf))?;
                            } else {
                                self.merge_insert(root, &path, Node::Leaf(leaf))?;
                            }
                            self.free_value_pages(&old_value_entry)?;
                            return Ok(Some(old_bytes));
                        }
                        Err(index) => {
                            let new_value = self.make_value(value)?;
                            leaf.kv.insert(index, (key, new_value));
                            self.split_insert(root, &path, Node::Leaf(leaf))?;
                            return Ok(None);
                        }
                    }
                }
                NextNode::Next(next_page) => {
                    current = next_page;
                }
                NextNode::NeedAlloc => {
                    let Node::Internal(mut internal) = self.pager.owned_node(current)? else {
                        fail!(TreeError::UnexpectedNodeType { expected: "internal" });
                    };

                    if internal.kv.is_empty() {
                        let new_leaf_page = self.alloc_page()?;
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

    fn split_insert(
        &mut self,
        root: NodePtr,
        path: &[NodePtr],
        insert: Node,
    ) -> Result<(), Error> {
        let buffer = rkyv::to_bytes(&insert)?;

        let page = *path.last().unwrap();
        let parents = &path[..path.len() - 1];

        if buffer.len() <= self.pager.page_content_size() {
            self.pager.write_buffer(page, buffer)?;
            return Ok(());
        }

        let page_content_size = self.pager.page_content_size();
        let keyed_nodes: Vec<(Key, Node)> = match insert {
            Node::Leaf(leaf) => split_leaf(leaf.kv, page_content_size)?
                .into_iter()
                .map(|kv| (kv.last().unwrap().0, Node::Leaf(Leaf { kv })))
                .collect(),
            Node::Internal(internal) => split_internal(internal.kv, page_content_size)?
                .into_iter()
                .map(|kv| (kv.last().unwrap().0, Node::Internal(Internal { kv })))
                .collect(),
        };

        self.write_splits(root, keyed_nodes, page, parents)
    }

    /// Write split nodes to pages and update the parent.
    /// The last node in `keyed_nodes` is written to `page`; the rest get new pages.
    fn write_splits(
        &mut self,
        root: NodePtr,
        mut keyed_nodes: Vec<(Key, Node)>,
        page: NodePtr,
        parents: &[NodePtr],
    ) -> Result<(), Error> {
        if let Some(&parent_page) = parents.last() {
            let (_right_key, right_node) = keyed_nodes.pop().unwrap();
            self.pager.write_node(page, &right_node)?;

            let Node::Internal(mut parent_internal) = self.pager.owned_node(parent_page)? else {
                fail!(TreeError::UnexpectedNodeType { expected: "internal parent" });
            };
            let mut kv_map: BTreeMap<_, _> = parent_internal.kv.into_iter().collect();
            for (key, node) in keyed_nodes {
                let new_page = self.alloc_page()?;
                self.pager.write_node(new_page, &node)?;
                kv_map.insert(key, new_page);
            }
            parent_internal.kv = kv_map.into_iter().collect();
            self.split_insert(root, parents, Node::Internal(parent_internal))
        } else {
            let new_entries = keyed_nodes
                .into_iter()
                .map(|(key, node)| {
                    let new_page = self.alloc_page()?;
                    self.pager.write_node(new_page, &node)?;
                    Ok((key, new_page))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            self.split_insert(root, &[root], Node::Internal(Internal { kv: new_entries }))
        }
    }

    /// Remove the entry for `key` from the tree rooted at `root`.
    /// Returns the old value if the key was found, or `None` if it did not exist.
    pub fn remove(&mut self, root: NodePtr, key: Key) -> Result<Option<Vec<u8>>, Error> {
        let mut current = root;
        let mut path = vec![];

        loop {
            path.push(current);

            let next = self
                .pager
                .read_node(current, |archived_node| match archived_node {
                    Archived::<Node>::Leaf(leaf) => {
                        match leaf.kv.binary_search_by_key(&key, |t| t.0.to_native()) {
                            Ok(_) => None,        // found — stop traversal
                            Err(_) => Some(None), // not found
                        }
                    }
                    Archived::<Node>::Internal(internal) => Some(find_child_page(internal, key)),
                })?;

            match next {
                None => {
                    // Key found in current leaf
                    let Node::Leaf(mut leaf) = self.pager.owned_node(current)? else {
                        fail!(TreeError::UnexpectedNodeType { expected: "leaf" });
                    };
                    let index = leaf
                        .kv
                        .binary_search_by_key(&key, |t| t.0)
                        .expect("Key not found in leaf node");
                    let old_value_entry = leaf.kv.remove(index).1;
                    self.merge_insert(root, &path, Node::Leaf(leaf))?;
                    let old_bytes = self.resolve_value(&old_value_entry)?;
                    self.free_value_pages(&old_value_entry)?;
                    return Ok(Some(old_bytes));
                }
                Some(Some(next_page)) => current = next_page,
                Some(None) => return Ok(None),
            }
        }
    }

    /// Remove all entries whose keys fall within `range` from the tree rooted at `root`.
    pub fn remove_range(
        &mut self,
        root: NodePtr,
        range: impl RangeBounds<Key>,
    ) -> Result<(), Error> {
        self.remove_range_at(root, root, &range, 0)
    }

    fn remove_range_at(
        &mut self,
        root: NodePtr,
        node_ptr: NodePtr,
        range: &impl RangeBounds<Key>,
        left_key: Key,
    ) -> Result<(), Error> {
        let node = self.pager.owned_node(node_ptr)?;

        match node {
            Node::Leaf(mut leaf) => {
                let removed: Vec<_> = leaf
                    .kv
                    .iter()
                    .filter(|(k, _)| range.contains(k))
                    .map(|(_, v)| v.clone())
                    .collect();
                leaf.kv.retain(|(k, _)| !range.contains(k));
                self.merge_insert(root, &[node_ptr], Node::Leaf(leaf))?;
                for v in &removed {
                    self.free_value_pages(v)?;
                }
            }
            Node::Internal(internal) => {
                let mut left_key = left_key;
                for (key, ptr) in &internal.kv {
                    if is_overlap(&(left_key..=*key), range) {
                        self.remove_range_at(root, *ptr, range, left_key)?;
                    }
                    left_key = *key;
                }
            }
        }

        Ok(())
    }

    fn merge_insert(
        &mut self,
        root: NodePtr,
        path: &[NodePtr],
        insert: Node,
    ) -> Result<(), Error> {
        let buffer = rkyv::to_bytes(&insert)?;

        let page = *path.last().unwrap();

        debug_assert!(buffer.len() <= self.pager.page_content_size());

        if buffer.len() >= self.pager.page_content_size() / 2 || path.len() <= 1 {
            return self.pager.write_buffer(page, buffer);
        }

        // Handle empty leaf: remove from parent entirely
        if let Node::Leaf(leaf) = &insert {
            if leaf.kv.is_empty() {
                let parents = &path[..path.len() - 1];
                let parent_page = *parents.last().unwrap();
                let Node::Internal(mut internal) = self.pager.owned_node(parent_page)? else {
                    fail!(TreeError::UnexpectedNodeType { expected: "internal parent" });
                };
                internal.kv.retain(|&(_, ptr)| ptr != page);
                self.free_page(page)?;
                return self.merge_insert(root, parents, Node::Internal(internal));
            }
        }

        let parents = &path[..path.len() - 1];
        if !self.try_merge_with_left_sibling(root, page, &insert, parents)? {
            self.pager.write_buffer(page, buffer)?;
        }
        Ok(())
    }

    /// Attempt to merge `insert` with its left sibling in the parent.
    /// Returns `true` if the merge was performed, `false` otherwise.
    fn try_merge_with_left_sibling(
        &mut self,
        root: NodePtr,
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
            self.merge_insert(root, parents, Node::Internal(parent_internal))?;
            self.free_page(left_sibling_page)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Look up a single key in the tree rooted at `root`.
    /// The closure `f` receives `Some(&[u8])` if found or `None` if absent,
    /// and its return value is propagated to the caller.
    pub fn read<T>(
        &mut self,
        root: NodePtr,
        key: Key,
        f: impl FnOnce(Option<&[u8]>) -> T,
    ) -> Result<T, Error> {
        enum ReadAction<T> {
            Done(T),
            Next(NodePtr),
            Overflow { start_page: u64, total_len: u64 },
        }

        let mut current = root;

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

    /// Iterate over all entries in the tree rooted at `root` whose keys fall
    /// within `range`, calling `f(key, value)` for each entry in order.
    pub fn read_range<R: RangeBounds<Key>>(
        &mut self,
        root: NodePtr,
        range: R,
        mut f: impl FnMut(Key, &[u8]),
    ) -> Result<(), Error> {
        self.read_range_at(root, &range, &mut (f), 0)
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

    /// Return the smallest key not yet present in the tree rooted at `root`.
    /// This is derived from the maximum existing key plus one.
    pub fn available_key(&mut self, root: NodePtr) -> Result<Key, Error> {
        self.pager
            .read_node(root, |archived_node| match archived_node {
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

    /// Find the child page for a given value in an index internal node.
    /// Returns `None` if value is greater than all entries (needs alloc).
    fn find_index_child(
        &mut self,
        internal: &IndexInternal,
        value_bytes: &[u8],
    ) -> Result<Option<NodePtr>, Error> {
        for (v, ptr) in &internal.kv {
            let v_bytes = self.resolve_value(v)?;
            if v_bytes.as_slice() >= value_bytes {
                return Ok(Some(*ptr));
            }
        }
        Ok(None)
    }

    /// Write a large value across one or more overflow pages.
    /// Each overflow page layout: [next_page: u64 LE][data bytes][padding]
    /// Returns the page number of the first overflow page.
    fn write_overflow(&mut self, data: &[u8]) -> Result<u64, Error> {
        let data_per_page = self.overflow_data_per_page();
        let num_pages = (data.len() + data_per_page - 1) / data_per_page;
        assert!(num_pages > 0);

        let pages: Vec<u64> = (0..num_pages)
            .map(|_| self.alloc_page())
            .collect::<Result<_, _>>()?;

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
            self.pager.write_raw_page(page_num, &buffer)?;
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
            let buffer = self.pager.read_raw_page(current_page)?;
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
        self.pager.page_size() - 8
    }

    /// Free pages used by a `Value`, if it is an overflow value.
    fn free_value_pages(&mut self, value: &Value) -> Result<(), Error> {
        if let Value::Overflow {
            start_page,
            total_len,
        } = value
        {
            self.free_overflow_pages(*start_page, *total_len)?;
        }
        Ok(())
    }

    /// Free all overflow pages for a chain starting at `start_page`.
    fn free_overflow_pages(&mut self, start_page: u64, total_len: u64) -> Result<(), Error> {
        let data_per_page = self.overflow_data_per_page();
        let mut current_page = start_page;
        let mut remaining = total_len as usize;

        while remaining > 0 {
            let buffer = self.pager.read_raw_page(current_page)?;
            let next_page = u64::from_le_bytes(buffer[..8].try_into().unwrap());
            self.free_page(current_page)?;
            let chunk_len = std::cmp::min(data_per_page, remaining);
            remaining -= chunk_len;
            current_page = next_page;
        }
        Ok(())
    }

    // ---- Persisted free page list (linked list in file) ----

    /// Read the free list head pointer from the free list metadata page.
    fn read_free_list_head(&mut self) -> Result<u64, Error> {
        let buf = self.pager.read_raw_page(FREE_LIST_PAGE_NUM)?;
        Ok(u64::from_le_bytes(buf[..8].try_into().unwrap()))
    }

    /// Write the free list head pointer to the free list metadata page.
    fn write_free_list_head(&mut self, head: u64) -> Result<(), Error> {
        let mut buf = vec![0u8; 8];
        buf[..8].copy_from_slice(&head.to_le_bytes());
        self.pager.write_raw_page(FREE_LIST_PAGE_NUM, &buf)
    }

    /// Allocate a page, reusing a freed page if available, otherwise extending the file.
    fn alloc_page(&mut self) -> Result<u64, Error> {
        let head = self.read_free_list_head()?;
        if head == u64::MAX {
            return Ok(self.pager.next_page_num());
        }
        // Read the next pointer from the free page
        let buf = self.pager.read_raw_page(head)?;
        let next = u64::from_le_bytes(buf[..8].try_into().unwrap());
        self.write_free_list_head(next)?;
        Ok(head)
    }

    /// Return a page to the persisted free list.
    pub(crate) fn free_page(&mut self, page_num: u64) -> Result<(), Error> {
        let head = self.read_free_list_head()?;
        // Write the current head into the freed page
        let mut buf = vec![0u8; 8];
        buf[..8].copy_from_slice(&head.to_le_bytes());
        self.pager.write_raw_page(page_num, &buf)?;
        self.write_free_list_head(page_num)
    }

    // ────────────────────────────────────────────────────────────────────
    //  Index tree operations (IndexNode: value → key mapping)
    // ────────────────────────────────────────────────────────────────────

    /// Initialize a new index tree, returning its root page number.
    pub fn init_index(&mut self) -> Result<NodePtr, Error> {
        let page = self.alloc_page()?;
        let root_leaf = IndexLeaf { kv: vec![] };
        self.pager
            .write_index_node(page, &IndexNode::Leaf(root_leaf))?;
        Ok(page)
    }

    /// Free all pages belonging to an index tree rooted at `root`,
    /// including overflow pages. The root page itself is also freed.
    pub fn free_index_tree(&mut self, root: NodePtr) -> Result<(), Error> {
        let node = self.pager.owned_index_node(root)?;
        match node {
            IndexNode::Leaf(leaf) => {
                for (v, _) in &leaf.kv {
                    self.free_value_pages(v)?;
                }
            }
            IndexNode::Internal(internal) => {
                for (_, ptr) in &internal.kv {
                    self.free_index_tree(*ptr)?;
                }
            }
        }
        self.free_page(root)
    }

    /// Insert a (value, key) pair into the index tree rooted at `root`.
    /// The tree is sorted by `value` (actual bytes). Returns `true` if the entry was newly
    /// inserted, `false` if the exact (value, key) pair already existed.
    pub fn index_insert(&mut self, root: NodePtr, key: Key, value: Vec<u8>) -> Result<bool, Error> {
        let new_value = self.make_value(value)?;
        let new_bytes = self.resolve_value(&new_value)?;
        let mut current = root;
        let mut path = vec![];

        loop {
            path.push(current);
            let node = self.pager.owned_index_node(current)?;

            match node {
                IndexNode::Leaf(mut leaf) => {
                    // Linear search comparing by resolved bytes
                    let mut insert_pos = leaf.kv.len();
                    for (i, (v, k)) in leaf.kv.iter().enumerate() {
                        let v_bytes = self.resolve_value(v)?;
                        let cmp = v_bytes.as_slice().cmp(new_bytes.as_slice());
                        match cmp {
                            std::cmp::Ordering::Equal => {
                                let cmp_key = k.cmp(&key);
                                match cmp_key {
                                    std::cmp::Ordering::Equal => return Ok(false),
                                    std::cmp::Ordering::Greater => {
                                        insert_pos = i;
                                        break;
                                    }
                                    std::cmp::Ordering::Less => {}
                                }
                            }
                            std::cmp::Ordering::Greater => {
                                insert_pos = i;
                                break;
                            }
                            std::cmp::Ordering::Less => {}
                        }
                    }
                    leaf.kv.insert(insert_pos, (new_value, key));
                    self.index_split_insert(root, &path, &IndexNode::Leaf(leaf))?;
                    return Ok(true);
                }
                IndexNode::Internal(mut internal) => {
                    match self.find_index_child(&internal, &new_bytes)? {
                        Some(next) => {
                            current = next;
                        }
                        None => {
                            // value is beyond all entries
                            if internal.kv.is_empty() {
                                let new_leaf_page = self.alloc_page()?;
                                let new_leaf = IndexLeaf {
                                    kv: vec![(new_value.clone(), key)],
                                };
                                self.pager
                                    .write_index_node(new_leaf_page, &IndexNode::Leaf(new_leaf))?;
                                internal.kv.push((new_value, new_leaf_page));
                                self.pager
                                    .write_index_node(current, &IndexNode::Internal(internal))?;
                                return Ok(true);
                            } else {
                                let last = internal.kv.last_mut().unwrap();
                                last.0 = new_value.clone();
                                let next = last.1;
                                self.pager
                                    .write_index_node(current, &IndexNode::Internal(internal))?;
                                current = next;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Remove the entry matching `(value, key)` from the index tree rooted at `root`.
    /// Routes by `value` (actual bytes). Returns `true` if the entry was found and removed.
    pub fn index_remove(&mut self, root: NodePtr, value: &Value, key: Key) -> Result<bool, Error> {
        let target_bytes = self.resolve_value(value)?;
        let mut current = root;
        let mut path = vec![];

        loop {
            path.push(current);
            let node = self.pager.owned_index_node(current)?;

            match node {
                IndexNode::Leaf(mut leaf) => {
                    // Find the entry by resolved bytes + key
                    let mut found_idx = None;
                    for (i, (v, k)) in leaf.kv.iter().enumerate() {
                        let v_bytes = self.resolve_value(v)?;
                        if v_bytes == target_bytes && *k == key {
                            found_idx = Some(i);
                            break;
                        }
                    }
                    match found_idx {
                        Some(idx) => {
                            let old_value_entry = leaf.kv.remove(idx).0;
                            self.index_merge_insert(root, &path, &IndexNode::Leaf(leaf))?;
                            self.free_value_pages(&old_value_entry)?;
                            return Ok(true);
                        }
                        None => return Ok(false),
                    }
                }
                IndexNode::Internal(internal) => {
                    match self.find_index_child(&internal, &target_bytes)? {
                        Some(next) => current = next,
                        None => return Ok(false),
                    }
                }
            }
        }
    }

    /// Look up by value in the index tree rooted at `root`.
    /// The closure `f` receives `Some(key)` if found or `None` if absent.
    /// Comparison is by actual bytes. If multiple entries share the same value,
    /// returns the first match.
    pub fn index_read<T>(
        &mut self,
        root: NodePtr,
        value: &Value,
        f: impl FnOnce(Option<Key>) -> T,
    ) -> Result<T, Error> {
        let target_bytes = self.resolve_value(value)?;
        let mut current = root;
        let mut ff = Some(f);

        loop {
            let node = self.pager.owned_index_node(current)?;
            match node {
                IndexNode::Leaf(leaf) => {
                    for (v, k) in &leaf.kv {
                        let v_bytes = self.resolve_value(v)?;
                        if v_bytes == target_bytes {
                            return Ok(ff.take().unwrap()(Some(*k)));
                        }
                    }
                    return Ok(ff.take().unwrap()(None));
                }
                IndexNode::Internal(internal) => {
                    match self.find_index_child(&internal, &target_bytes)? {
                        Some(next) => current = next,
                        None => return Ok(ff.take().unwrap()(None)),
                    }
                }
            }
        }
    }

    /// Iterate over all entries in the index tree whose values (by actual bytes)
    /// fall within `range`, calling `f(value_bytes, key)` for each entry in order.
    pub fn index_read_range<R: RangeBounds<Vec<u8>>>(
        &mut self,
        root: NodePtr,
        range: R,
        mut f: impl FnMut(&[u8], Key),
    ) -> Result<(), Error> {
        self.index_read_range_at(root, &range, &mut f)
    }

    fn index_read_range_at<R: RangeBounds<Vec<u8>>>(
        &mut self,
        node_ptr: NodePtr,
        range: &R,
        f: &mut impl FnMut(&[u8], Key),
    ) -> Result<(), Error> {
        let node = self.pager.owned_index_node(node_ptr)?;

        match node {
            IndexNode::Leaf(leaf) => {
                for (v, k) in leaf.kv {
                    let v_bytes = self.resolve_value(&v)?;
                    if range.contains(&v_bytes) {
                        f(&v_bytes, k);
                    }
                }
            }
            IndexNode::Internal(internal) => {
                for i in 0..internal.kv.len() {
                    let max_bytes = self.resolve_value(&internal.kv[i].0)?;
                    let ptr = internal.kv[i].1;

                    // Skip if max_value is below range start
                    let below_start = match range.start_bound() {
                        Bound::Included(s) => max_bytes.as_slice() < s.as_slice(),
                        Bound::Excluded(s) => max_bytes.as_slice() <= s.as_slice(),
                        Bound::Unbounded => false,
                    };
                    if below_start {
                        continue;
                    }

                    // Stop if this subtree's min value exceeds range end
                    if i > 0 {
                        let prev_max_bytes = self.resolve_value(&internal.kv[i - 1].0)?;
                        let beyond_end = match range.end_bound() {
                            Bound::Included(e) => prev_max_bytes.as_slice() > e.as_slice(),
                            Bound::Excluded(e) => prev_max_bytes.as_slice() >= e.as_slice(),
                            Bound::Unbounded => false,
                        };
                        if beyond_end {
                            break;
                        }
                    }

                    self.index_read_range_at(ptr, range, f)?;
                }
            }
        }

        Ok(())
    }

    /// Return the smallest key not yet present in the index tree rooted at `root`.
    /// Since the tree is sorted by value, this traverses all leaves to find the max key.
    pub fn index_available_key(&mut self, root: NodePtr) -> Result<Key, Error> {
        let node = self.pager.owned_index_node(root)?;
        match node {
            IndexNode::Leaf(leaf) => Ok(leaf
                .kv
                .iter()
                .map(|t| t.1)
                .max()
                .map_or(0, |k| k.checked_add(1).unwrap_or(u64::MAX))),
            IndexNode::Internal(internal) => {
                let mut max_key: Key = 0;
                for (_, ptr) in &internal.kv {
                    let child_key = self.index_available_key(*ptr)?;
                    max_key = max_key.max(child_key);
                }
                Ok(max_key)
            }
        }
    }

    fn index_split_insert(
        &mut self,
        root: NodePtr,
        path: &[NodePtr],
        insert: &IndexNode,
    ) -> Result<(), Error> {
        let buffer = rkyv::to_bytes(insert)?;

        let page = *path.last().unwrap();
        let parents = &path[..path.len() - 1];

        if buffer.len() <= self.pager.page_content_size() {
            self.pager.write_buffer(page, buffer)?;
            return Ok(());
        }

        let page_content_size = self.pager.page_content_size();
        let mut keyed_nodes: Vec<(Value, IndexNode)> = match insert {
            IndexNode::Leaf(leaf) => split_index_leaf(leaf.kv.clone(), page_content_size)?
                .into_iter()
                .map(|kv| {
                    let max_value = kv.last().unwrap().0.clone();
                    (max_value, IndexNode::Leaf(IndexLeaf { kv }))
                })
                .collect(),
            IndexNode::Internal(internal) => {
                split_index_internal(internal.kv.clone(), page_content_size)?
                    .into_iter()
                    .map(|kv| {
                        let max_value = kv.last().unwrap().0.clone();
                        (max_value, IndexNode::Internal(IndexInternal { kv }))
                    })
                    .collect()
            }
        };

        // Sort split chunks by resolved max value bytes
        self.sort_keyed_nodes(&mut keyed_nodes)?;

        self.index_write_splits(root, keyed_nodes, page, parents)
    }

    /// Sort a vec of (Value, T) by resolved byte content of the Value.
    fn sort_keyed_nodes<T>(&mut self, nodes: &mut Vec<(Value, T)>) -> Result<(), Error> {
        // Resolve all values to bytes, then sort by those bytes
        let mut bytes: Vec<Vec<u8>> = Vec::with_capacity(nodes.len());
        for (v, _) in nodes.iter() {
            bytes.push(self.resolve_value(v)?);
        }
        // Build index array and sort
        let mut indices: Vec<usize> = (0..nodes.len()).collect();
        indices.sort_by(|&a, &b| bytes[a].cmp(&bytes[b]));
        // Reorder in-place using the sorted indices
        let mut sorted = Vec::with_capacity(nodes.len());
        for &i in &indices {
            // Safe because we consume the original vec below
            sorted.push(i);
        }
        let mut taken: Vec<Option<(Value, T)>> = nodes.drain(..).map(Some).collect();
        for i in sorted {
            nodes.push(taken[i].take().unwrap());
        }
        Ok(())
    }

    /// Sort an IndexInternal's kv entries by resolved byte content of the Value.
    fn sort_index_internal_kv(&mut self, kv: &mut Vec<(Value, NodePtr)>) -> Result<(), Error> {
        let mut bytes: Vec<Vec<u8>> = Vec::with_capacity(kv.len());
        for (v, _) in kv.iter() {
            bytes.push(self.resolve_value(v)?);
        }
        let mut indices: Vec<usize> = (0..kv.len()).collect();
        indices.sort_by(|&a, &b| bytes[a].cmp(&bytes[b]));
        let mut taken: Vec<Option<(Value, NodePtr)>> = kv.drain(..).map(Some).collect();
        for &i in &indices {
            kv.push(taken[i].take().unwrap());
        }
        Ok(())
    }

    fn index_write_splits(
        &mut self,
        root: NodePtr,
        mut keyed_nodes: Vec<(Value, IndexNode)>,
        page: NodePtr,
        parents: &[NodePtr],
    ) -> Result<(), Error> {
        if let Some(&parent_page) = parents.last() {
            let (_right_value, right_node) = keyed_nodes.pop().unwrap();
            self.pager.write_index_node(page, &right_node)?;

            let IndexNode::Internal(mut parent_internal) =
                self.pager.owned_index_node(parent_page)?
            else {
                fail!(TreeError::UnexpectedNodeType { expected: "internal parent" });
            };
            for (value, node) in keyed_nodes {
                let new_page = self.alloc_page()?;
                self.pager.write_index_node(new_page, &node)?;
                parent_internal.kv.push((value, new_page));
            }
            self.sort_index_internal_kv(&mut parent_internal.kv)?;
            self.index_split_insert(root, parents, &IndexNode::Internal(parent_internal))
        } else {
            let new_entries = keyed_nodes
                .into_iter()
                .map(|(value, node)| {
                    let new_page = self.alloc_page()?;
                    self.pager.write_index_node(new_page, &node)?;
                    Ok((value, new_page))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            self.index_split_insert(
                root,
                &[root],
                &IndexNode::Internal(IndexInternal { kv: new_entries }),
            )
        }
    }

    fn index_merge_insert(
        &mut self,
        root: NodePtr,
        path: &[NodePtr],
        insert: &IndexNode,
    ) -> Result<(), Error> {
        let buffer = rkyv::to_bytes(insert)?;

        let page = *path.last().unwrap();

        debug_assert!(buffer.len() <= self.pager.page_content_size());

        if buffer.len() >= self.pager.page_content_size() / 2 || path.len() <= 1 {
            return self.pager.write_buffer(page, buffer);
        }

        // Handle empty leaf: remove from parent entirely
        if let IndexNode::Leaf(leaf) = insert {
            if leaf.kv.is_empty() {
                let parents = &path[..path.len() - 1];
                let parent_page = *parents.last().unwrap();
                let IndexNode::Internal(mut internal) = self.pager.owned_index_node(parent_page)?
                else {
                    fail!(TreeError::UnexpectedNodeType { expected: "internal parent" });
                };
                internal.kv.retain(|entry| entry.1 != page);
                self.free_page(page)?;
                return self.index_merge_insert(root, parents, &IndexNode::Internal(internal));
            }
        }

        let parents = &path[..path.len() - 1];
        if !self.index_try_merge_with_left_sibling(root, page, insert, parents)? {
            self.pager.write_index_node(page, insert)?;
        }
        Ok(())
    }

    fn index_try_merge_with_left_sibling(
        &mut self,
        root: NodePtr,
        page: NodePtr,
        insert: &IndexNode,
        parents: &[NodePtr],
    ) -> Result<bool, Error> {
        let parent_page = *parents.last().unwrap();
        let IndexNode::Internal(mut parent_internal) = self.pager.owned_index_node(parent_page)?
        else {
            return Ok(false);
        };

        let index = parent_internal
            .kv
            .iter()
            .position(|entry| entry.1 == page)
            .unwrap();
        if index == 0 {
            return Ok(false);
        }

        let left_sibling_page = parent_internal.kv[index - 1].1;
        let left_sibling_node = self.pager.owned_index_node(left_sibling_page)?;

        let merged = match (left_sibling_node, insert) {
            (IndexNode::Leaf(mut left), IndexNode::Leaf(right)) => {
                left.kv.extend(right.kv.iter().cloned());
                IndexNode::Leaf(left)
            }
            (IndexNode::Internal(mut left), IndexNode::Internal(right)) => {
                left.kv.extend(right.kv.iter().cloned());
                IndexNode::Internal(left)
            }
            _ => return Ok(false),
        };

        let buffer = rkyv::to_bytes(&merged)?;
        if buffer.len() <= self.pager.page_content_size() {
            self.pager.write_buffer(page, buffer)?;
            parent_internal.kv.remove(index - 1);
            self.index_merge_insert(root, parents, &IndexNode::Internal(parent_internal))?;
            self.free_page(left_sibling_page)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Print a human-readable dump of the tree rooted at `root` for debugging.
    /// Panics if any key is outside `[min, max]`.
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

    /// Check for page leaks. Walks the tree and free list, verifying every page
    /// in [0, next_page_num) is accounted for exactly once.
    /// Panics with a detailed message if any leaked or double-used pages are found.
    #[cfg(test)]
    pub fn assert_no_page_leak(&mut self, root: NodePtr) {
        use std::collections::BTreeSet;

        let total_pages = self.pager.total_page_count();

        // Collect all pages reachable from the tree
        let mut tree_pages = BTreeSet::new();
        tree_pages.insert(root);
        tree_pages.insert(FREE_LIST_PAGE_NUM);
        self.collect_tree_pages(root, &mut tree_pages);

        // Collect all pages in the free list
        let mut free_pages = BTreeSet::new();
        let mut head = self.read_free_list_head().unwrap();
        while head != u64::MAX {
            assert!(
                free_pages.insert(head),
                "Free list cycle detected at page {}",
                head
            );
            let buf = self.pager.read_raw_page(head).unwrap();
            head = u64::from_le_bytes(buf[..8].try_into().unwrap());
        }

        // Check for overlap
        let overlap: Vec<_> = tree_pages.intersection(&free_pages).collect();
        assert!(
            overlap.is_empty(),
            "Pages in both tree and free list: {:?}",
            overlap
        );

        // Check all pages are accounted for
        let mut all_pages: BTreeSet<u64> = (0..total_pages).collect();
        for &p in &tree_pages {
            all_pages.remove(&p);
        }
        for &p in &free_pages {
            all_pages.remove(&p);
        }
        assert!(
            all_pages.is_empty(),
            "Leaked pages (not in tree or free list): {:?} (tree={}, free={}, total={})",
            all_pages,
            tree_pages.len(),
            free_pages.len(),
            total_pages
        );
    }

    /// Recursively collect all pages used by the tree rooted at `page_num`,
    /// including overflow pages.
    #[cfg(test)]
    fn collect_tree_pages(
        &mut self,
        page_num: NodePtr,
        pages: &mut std::collections::BTreeSet<u64>,
    ) {
        let node = self.pager.owned_node(page_num).unwrap();
        match node {
            Node::Leaf(leaf) => {
                for (_, v) in &leaf.kv {
                    if let Value::Overflow {
                        start_page,
                        total_len,
                    } = v
                    {
                        self.collect_overflow_pages(*start_page, *total_len, pages);
                    }
                }
            }
            Node::Internal(internal) => {
                for (_, ptr) in &internal.kv {
                    assert!(
                        pages.insert(*ptr),
                        "Page {} referenced multiple times in tree",
                        ptr
                    );
                    self.collect_tree_pages(*ptr, pages);
                }
            }
        }
    }

    /// Collect all overflow pages in a chain.
    #[cfg(test)]
    fn collect_overflow_pages(
        &mut self,
        start_page: u64,
        total_len: u64,
        pages: &mut std::collections::BTreeSet<u64>,
    ) {
        let data_per_page = self.overflow_data_per_page();
        let mut current = start_page;
        let mut remaining = total_len as usize;
        while remaining > 0 {
            assert!(
                pages.insert(current),
                "Overflow page {} referenced multiple times",
                current
            );
            let buf = self.pager.read_raw_page(current).unwrap();
            let next = u64::from_le_bytes(buf[..8].try_into().unwrap());
            let chunk = std::cmp::min(data_per_page, remaining);
            remaining -= chunk;
            current = next;
        }
    }

    /// Check for page leaks with an index tree root. Same as [`assert_no_page_leak`]
    /// but walks an `IndexNode` tree instead of a `Node` tree.
    #[cfg(test)]
    pub fn assert_no_page_leak_index(&mut self, root: NodePtr) {
        use std::collections::BTreeSet;

        let total_pages = self.pager.total_page_count();

        let mut tree_pages = BTreeSet::new();
        tree_pages.insert(root);
        tree_pages.insert(FREE_LIST_PAGE_NUM);
        self.collect_index_tree_pages(root, &mut tree_pages);

        let mut free_pages = BTreeSet::new();
        let mut head = self.read_free_list_head().unwrap();
        while head != u64::MAX {
            assert!(
                free_pages.insert(head),
                "Free list cycle detected at page {}",
                head
            );
            let buf = self.pager.read_raw_page(head).unwrap();
            head = u64::from_le_bytes(buf[..8].try_into().unwrap());
        }

        let overlap: Vec<_> = tree_pages.intersection(&free_pages).collect();
        assert!(
            overlap.is_empty(),
            "Pages in both tree and free list: {:?}",
            overlap
        );

        let mut all_pages: BTreeSet<u64> = (0..total_pages).collect();
        for &p in &tree_pages {
            all_pages.remove(&p);
        }
        for &p in &free_pages {
            all_pages.remove(&p);
        }
        assert!(
            all_pages.is_empty(),
            "Leaked pages (not in tree or free list): {:?} (tree={}, free={}, total={})",
            all_pages,
            tree_pages.len(),
            free_pages.len(),
            total_pages
        );
    }

    /// Recursively collect all pages used by the index tree rooted at `page_num`,
    /// including overflow pages.
    #[cfg(test)]
    fn collect_index_tree_pages(
        &mut self,
        page_num: NodePtr,
        pages: &mut std::collections::BTreeSet<u64>,
    ) {
        let node = self.pager.owned_index_node(page_num).unwrap();
        match node {
            IndexNode::Leaf(leaf) => {
                for (v, _) in &leaf.kv {
                    if let Value::Overflow {
                        start_page,
                        total_len,
                    } = v
                    {
                        self.collect_overflow_pages(*start_page, *total_len, pages);
                    }
                }
            }
            IndexNode::Internal(internal) => {
                for (v, ptr) in &internal.kv {
                    assert!(
                        pages.insert(*ptr),
                        "Page {} referenced multiple times in index tree",
                        ptr
                    );
                    if let Value::Overflow {
                        start_page,
                        total_len,
                    } = v
                    {
                        self.collect_overflow_pages(*start_page, *total_len, pages);
                    }
                    self.collect_index_tree_pages(*ptr, pages);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use std::collections::HashMap;
    use std::ops::Bound;

    use super::*;

    /// Create an initialized btree backed by a temp file with the given page size.
    fn new_btree(page_size: usize) -> (Btree, NodePtr) {
        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file, page_size);
        let mut btree = Btree::new(pager);
        btree.init().unwrap();
        let root = btree.init_tree().unwrap();
        (btree, root)
    }

    /// Create a btree pre-populated with keys `0..count`, each with a 64-byte value.
    fn build_btree(count: u64) -> (Btree, NodePtr) {
        let (mut btree, root) = new_btree(4096);
        for i in 0..count {
            let mut value = vec![0u8; 64];
            value[0..8].copy_from_slice(&i.to_le_bytes());
            btree.insert(root, i, value).unwrap();
        }
        (btree, root)
    }

    fn read_range_keys<R: RangeBounds<Key>>(
        btree: &mut Btree,
        root: NodePtr,
        range: R,
    ) -> Vec<Key> {
        let mut keys = Vec::new();
        btree.read_range(root, range, |k, _| keys.push(k)).unwrap();
        keys
    }

    fn read_value(btree: &mut Btree, root: NodePtr, key: Key) -> Option<Vec<u8>> {
        btree.read(root, key, |v| v.map(|b| b.to_vec())).unwrap()
    }

    fn assert_read_eq(btree: &mut Btree, root: NodePtr, key: Key, expected: &[u8]) {
        assert!(
            btree.read(root, key, |v| v == Some(expected)).unwrap(),
            "Key {} value mismatch",
            key
        );
    }

    fn assert_key_exists(btree: &mut Btree, root: NodePtr, key: Key) {
        assert!(
            btree.read(root, key, |v| v.is_some()).unwrap(),
            "Key {} should exist",
            key
        );
    }

    fn assert_key_absent(btree: &mut Btree, root: NodePtr, key: Key) {
        assert!(
            btree.read(root, key, |v| v.is_none()).unwrap(),
            "Key {} should be absent",
            key
        );
    }

    fn assert_keys_exist(btree: &mut Btree, root: NodePtr, range: impl IntoIterator<Item = u64>) {
        for k in range {
            assert_key_exists(btree, root, k);
        }
    }

    fn assert_keys_absent(btree: &mut Btree, root: NodePtr, range: impl IntoIterator<Item = u64>) {
        for k in range {
            assert_key_absent(btree, root, k);
        }
    }

    #[test]
    fn test_insert() {
        let (mut btree, root) = new_btree(4096);
        btree.insert(root, 1, b"one".to_vec()).unwrap();
    }

    #[test]
    fn test_read() {
        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file, 4096);
        let mut btree = Btree::new(pager);
        btree.init().unwrap();
        let root = btree.init_tree().unwrap();
        let root_leaf = Leaf {
            kv: vec![(0, Value::Inline(b"zero".to_vec()))],
        };
        btree
            .pager
            .write_node(root, &Node::Leaf(root_leaf))
            .unwrap();
        assert_read_eq(&mut btree, root, 0, b"zero");
    }

    #[test]
    fn test_insert_and_read() {
        let (mut btree, root) = new_btree(4096);
        btree.insert(root, 42, b"forty-two".to_vec()).unwrap();
        assert_read_eq(&mut btree, root, 42, b"forty-two");
    }

    #[test]
    fn test_insert_multiple_and_read() {
        let (mut btree, root) = new_btree(64);
        let mut map = HashMap::new();

        for i in 0u64..256 {
            let value = format!("value-{}", i).as_bytes().to_vec();
            btree.insert(root, i, value.clone()).unwrap();
            map.insert(i, value);

            for j in 0u64..=i {
                let expected = map.get(&j).unwrap();
                assert_read_eq(&mut btree, root, j, expected);
            }
        }
    }

    #[test]
    fn test_remove() {
        let (mut btree, root) = new_btree(64);

        btree.insert(root, 1, b"one".to_vec()).unwrap();
        btree.insert(root, 2, b"two".to_vec()).unwrap();
        btree.insert(root, 3, b"three".to_vec()).unwrap();

        assert_eq!(btree.remove(root, 2).unwrap(), Some(b"two".to_vec()));
        assert_key_absent(&mut btree, root, 2);
        assert_read_eq(&mut btree, root, 1, b"one");
        assert_read_eq(&mut btree, root, 3, b"three");
        assert_eq!(btree.remove(root, 999).unwrap(), None);
    }

    #[test]
    fn test_remove_seq() {
        const LEN: u64 = 1000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let (mut btree, root) = new_btree(64);

        let mut insert = (0..LEN).collect::<Vec<u64>>();
        insert.shuffle(&mut rng);

        let mut remove = insert.clone();
        remove.shuffle(&mut rng);

        for i in insert {
            btree
                .insert(root, i, format!("value-{}", i).as_bytes().to_vec())
                .unwrap();
        }

        for i in remove {
            assert_eq!(
                btree.remove(root, i).unwrap(),
                Some(format!("value-{}", i).as_bytes().to_vec()),
                "Failed to remove key {}",
                i
            );
            assert_key_absent(&mut btree, root, i);
        }
    }

    #[test]
    fn test_read_range_full() {
        let (mut btree, root) = build_btree(200);
        assert_eq!(
            read_range_keys(&mut btree, root, 0..=199),
            (0..200).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_read_range_exclusive_start() {
        let (mut btree, root) = build_btree(200);
        let keys = read_range_keys(&mut btree, root, (Bound::Excluded(10), Bound::Included(20)));
        assert_eq!(keys, (11..=20).collect::<Vec<_>>());
    }

    #[test]
    fn test_read_range_unbounded_end() {
        let (mut btree, root) = build_btree(200);
        assert_eq!(
            read_range_keys(&mut btree, root, ..=5),
            (0..=5).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_read_range_empty() {
        let (mut btree, root) = build_btree(200);
        assert!(read_range_keys(&mut btree, root, 500..=600).is_empty());
    }

    #[test]
    fn test_read_range_order() {
        let (mut btree, root) = build_btree(500);
        assert_eq!(
            read_range_keys(&mut btree, root, 123..=321),
            (123..=321).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_remove_range_inclusive() {
        let (mut btree, root) = build_btree(100);
        btree.remove_range(root, 10..=20).unwrap();

        assert_keys_absent(&mut btree, root, 10..=20);
        assert_keys_exist(&mut btree, root, 0..10);
        assert_keys_exist(&mut btree, root, 21..100);
    }

    #[test]
    fn test_remove_range_exclusive_start() {
        let (mut btree, root) = build_btree(100);
        btree
            .remove_range(root, (Bound::Excluded(10), Bound::Included(20)))
            .unwrap();

        assert_key_exists(&mut btree, root, 10);
        assert_keys_absent(&mut btree, root, 11..=20);
        assert_keys_exist(&mut btree, root, 0..10);
        assert_keys_exist(&mut btree, root, 21..100);
    }

    #[test]
    fn test_remove_range_unbounded_end() {
        let (mut btree, root) = build_btree(100);
        btree.remove_range(root, 80..).unwrap();

        assert_keys_absent(&mut btree, root, 80..100);
        assert_keys_exist(&mut btree, root, 0..80);
    }

    #[test]
    fn test_remove_range_unbounded_start() {
        let (mut btree, root) = build_btree(100);
        btree.remove_range(root, ..=20).unwrap();

        assert_keys_absent(&mut btree, root, 0..=20);
        assert_keys_exist(&mut btree, root, 21..100);
    }

    #[test]
    fn test_remove_range_nonexistent() {
        let (mut btree, root) = build_btree(100);
        btree.remove_range(root, 200..=300).unwrap();
        assert_keys_exist(&mut btree, root, 0..100);
    }

    #[test]
    fn test_remove_range_empty() {
        let (mut btree, root) = build_btree(100);
        btree.remove_range(root, 50..50).unwrap();
        assert_keys_exist(&mut btree, root, 0..100);
    }

    #[test]
    fn test_remove_range_all_keys() {
        let (mut btree, root) = build_btree(100);
        btree.remove_range(root, 0..=99).unwrap();
        assert_keys_absent(&mut btree, root, 0..100);
    }

    #[test]
    fn test_remove_range_multiple_calls() {
        let (mut btree, root) = build_btree(100);

        btree.remove_range(root, 10..=20).unwrap();
        assert_keys_absent(&mut btree, root, 10..=20);

        btree.remove_range(root, 50..=60).unwrap();
        assert_keys_absent(&mut btree, root, 10..=20);
        assert_keys_absent(&mut btree, root, 50..=60);
        assert_keys_exist(&mut btree, root, 0..10);
        assert_keys_exist(&mut btree, root, 21..50);
        assert_keys_exist(&mut btree, root, 61..100);
    }

    #[test]
    fn test_remove_range_verify_read_range() {
        let (mut btree, root) = build_btree(100);
        btree.remove_range(root, 30..=70).unwrap();

        let keys = read_range_keys(&mut btree, root, 0..=99);
        let mut expected = (0..30).collect::<Vec<_>>();
        expected.extend(71..100);
        assert_eq!(keys, expected);
    }

    #[test]
    fn test_remove_range_large_dataset() {
        let (mut btree, root) = build_btree(1000);
        btree.remove_range(root, 250..=750).unwrap();

        assert_keys_absent(&mut btree, root, 250..=750);
        assert_keys_exist(&mut btree, root, 0..250);
        assert_keys_exist(&mut btree, root, 751..1000);
    }

    #[test]
    fn test_big_value_insert_and_read() {
        let (mut btree, root) = new_btree(256);
        let big_value = vec![42u8; 4096];
        btree.insert(root, 1, big_value.clone()).unwrap();
        assert_eq!(read_value(&mut btree, root, 1), Some(big_value));
    }

    #[test]
    fn test_big_value_with_small_values() {
        let (mut btree, root) = new_btree(256);

        btree.insert(root, 0, b"small-before".to_vec()).unwrap();
        let big_value = vec![42u8; 4096];
        btree.insert(root, 1, big_value.clone()).unwrap();
        btree.insert(root, 2, b"small-after".to_vec()).unwrap();

        assert_read_eq(&mut btree, root, 0, b"small-before");
        assert_eq!(read_value(&mut btree, root, 1), Some(big_value));
        assert_read_eq(&mut btree, root, 2, b"small-after");
    }

    #[test]
    fn test_big_value_update() {
        let (mut btree, root) = new_btree(256);

        let big_value = vec![42u8; 4096];
        btree.insert(root, 1, big_value.clone()).unwrap();

        let bigger_value = vec![99u8; 8192];
        let old = btree.insert(root, 1, bigger_value.clone()).unwrap();
        assert_eq!(old, Some(big_value));
        assert_eq!(read_value(&mut btree, root, 1), Some(bigger_value));
    }

    #[test]
    fn test_big_value_remove() {
        let (mut btree, root) = new_btree(256);

        let big_value = vec![42u8; 4096];
        btree.insert(root, 1, big_value.clone()).unwrap();

        assert_eq!(btree.remove(root, 1).unwrap(), Some(big_value));
        assert_key_absent(&mut btree, root, 1);
    }

    #[test]
    fn test_big_value_read_range() {
        let (mut btree, root) = new_btree(256);

        btree.insert(root, 0, b"small".to_vec()).unwrap();
        let big_value = vec![42u8; 2048];
        btree.insert(root, 1, big_value.clone()).unwrap();
        btree.insert(root, 2, b"also-small".to_vec()).unwrap();

        let mut results = Vec::new();
        btree
            .read_range(root, 0..=2, |k, v| results.push((k, v.to_vec())))
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], (0, b"small".to_vec()));
        assert_eq!(results[1], (1, big_value));
        assert_eq!(results[2], (2, b"also-small".to_vec()));
    }

    #[test]
    fn test_multiple_big_values() {
        let (mut btree, root) = new_btree(256);

        let mut expected = HashMap::new();
        for i in 0u64..20 {
            let big_value = vec![i as u8; 1024 + (i as usize * 100)];
            btree.insert(root, i, big_value.clone()).unwrap();
            expected.insert(i, big_value);
        }

        for (key, value) in &expected {
            assert_eq!(
                read_value(&mut btree, root, *key).as_ref(),
                Some(value),
                "Mismatch at key {}",
                key
            );
        }
    }

    #[test]
    fn test_big_value_replace_with_small() {
        let (mut btree, root) = new_btree(256);

        let big_value = vec![42u8; 4096];
        btree.insert(root, 1, big_value.clone()).unwrap();

        let old = btree.insert(root, 1, b"tiny".to_vec()).unwrap();
        assert_eq!(old, Some(big_value));
        assert_read_eq(&mut btree, root, 1, b"tiny");
    }

    #[test]
    fn test_free_pages_reused_after_remove() {
        let (mut btree, root) = new_btree(4096);
        for i in 0u64..100 {
            btree.insert(root, i, vec![0u8; 64]).unwrap();
        }
        let pages_after_insert = btree.pager.total_page_count();

        // Remove all keys — freed pages should accumulate
        for i in 0u64..100 {
            btree.remove(root, i).unwrap();
        }
        // Free list head should not be empty
        assert_ne!(
            btree.read_free_list_head().unwrap(),
            u64::MAX,
            "Expected free pages after removal"
        );

        // Re-insert — file should not grow (pages reused)
        for i in 0u64..100 {
            btree.insert(root, i, vec![0u8; 64]).unwrap();
        }
        assert!(
            btree.pager.total_page_count() <= pages_after_insert + 1,
            "File grew unexpectedly: {} > {}",
            btree.pager.total_page_count(),
            pages_after_insert
        );
    }

    #[test]
    fn test_free_overflow_pages_on_remove() {
        let (mut btree, root) = new_btree(256);
        let big_value = vec![42u8; 4096];
        btree.insert(root, 1, big_value).unwrap();
        let pages_before = btree.pager.total_page_count();

        btree.remove(root, 1).unwrap();
        assert_ne!(
            btree.read_free_list_head().unwrap(),
            u64::MAX,
            "Overflow pages should be freed on remove"
        );

        // Re-insert a big value — should reuse freed overflow pages
        let big_value2 = vec![99u8; 4096];
        btree.insert(root, 2, big_value2).unwrap();
        assert!(
            btree.pager.total_page_count() <= pages_before + 1,
            "Overflow pages were not reused"
        );
    }

    #[test]
    fn test_free_overflow_pages_on_update() {
        let (mut btree, root) = new_btree(256);
        let big_value = vec![42u8; 4096];
        btree.insert(root, 1, big_value).unwrap();
        let pages_before = btree.pager.total_page_count();

        // Replace with small value — overflow pages should be freed
        btree.insert(root, 1, b"small".to_vec()).unwrap();
        assert_ne!(
            btree.read_free_list_head().unwrap(),
            u64::MAX,
            "Old overflow pages should be freed on update"
        );

        // Allocating a big value again should reuse pages
        let big_value2 = vec![99u8; 4096];
        btree.insert(root, 2, big_value2).unwrap();
        assert!(
            btree.pager.total_page_count() <= pages_before + 1,
            "Old overflow pages were not reused on update"
        );
    }

    #[test]
    fn test_free_list_persisted() {
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let path = temp_file.path().to_owned();

        // Create a btree, insert and remove keys, then drop it
        let root;
        {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            let pager = Pager::new(file, 4096);
            let mut btree = Btree::new(pager);
            btree.init().unwrap();
            root = btree.init_tree().unwrap();

            for i in 0u64..50 {
                btree.insert(root, i, vec![0u8; 64]).unwrap();
            }
            for i in 0u64..50 {
                btree.remove(root, i).unwrap();
            }
            assert_ne!(btree.read_free_list_head().unwrap(), u64::MAX);
        }

        // Reopen the file — free list should still be available
        {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            let pager = Pager::new(file, 4096);
            let mut btree = Btree::new(pager);
            let pages_before = btree.pager.total_page_count();

            assert_ne!(
                btree.read_free_list_head().unwrap(),
                u64::MAX,
                "Free list lost after reopen"
            );

            // New inserts should reuse freed pages
            for i in 0u64..50 {
                btree.insert(root, i, vec![0u8; 64]).unwrap();
            }
            assert!(
                btree.pager.total_page_count() <= pages_before,
                "File grew after reopen despite free pages"
            );
        }
    }

    #[test]
    fn test_no_leak_insert_only() {
        let (mut btree, root) = new_btree(64);
        for i in 0u64..256 {
            btree
                .insert(root, i, format!("v-{}", i).into_bytes())
                .unwrap();
        }
        btree.assert_no_page_leak(root);
    }

    #[test]
    fn test_no_leak_insert_and_remove_all() {
        let (mut btree, root) = new_btree(64);
        for i in 0u64..200 {
            btree.insert(root, i, vec![0u8; 16]).unwrap();
        }
        for i in 0u64..200 {
            btree.remove(root, i).unwrap();
        }
        btree.assert_no_page_leak(root);
    }

    #[test]
    fn test_no_leak_insert_remove_shuffle() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let (mut btree, root) = new_btree(64);

        let mut keys: Vec<u64> = (0..500).collect();
        keys.shuffle(&mut rng);
        for &k in &keys {
            btree
                .insert(root, k, format!("val-{}", k).into_bytes())
                .unwrap();
        }

        keys.shuffle(&mut rng);
        for &k in &keys {
            btree.remove(root, k).unwrap();
        }
        btree.assert_no_page_leak(root);
    }

    #[test]
    fn test_no_leak_remove_range() {
        let (mut btree, root) = build_btree(200);
        btree.remove_range(root, 50..=150).unwrap();
        btree.assert_no_page_leak(root);
    }

    #[test]
    fn test_no_leak_remove_range_all() {
        let (mut btree, root) = build_btree(200);
        btree.remove_range(root, 0..=199).unwrap();
        btree.assert_no_page_leak(root);
    }

    #[test]
    fn test_no_leak_overflow_insert_remove() {
        let (mut btree, root) = new_btree(256);
        for i in 0u64..10 {
            let big = vec![i as u8; 1024 + i as usize * 100];
            btree.insert(root, i, big).unwrap();
        }
        btree.assert_no_page_leak(root);

        for i in 0u64..10 {
            btree.remove(root, i).unwrap();
        }
        btree.assert_no_page_leak(root);
    }

    #[test]
    fn test_no_leak_overflow_update() {
        let (mut btree, root) = new_btree(256);
        btree.insert(root, 1, vec![42u8; 4096]).unwrap();
        btree.assert_no_page_leak(root);

        // Replace overflow with small
        btree.insert(root, 1, b"small".to_vec()).unwrap();
        btree.assert_no_page_leak(root);

        // Replace small with overflow
        btree.insert(root, 1, vec![99u8; 4096]).unwrap();
        btree.assert_no_page_leak(root);

        // Replace overflow with different overflow
        btree.insert(root, 1, vec![77u8; 8192]).unwrap();
        btree.assert_no_page_leak(root);
    }

    #[test]
    fn test_no_leak_mixed_operations() {
        let (mut btree, root) = new_btree(128);

        for round in 0..5 {
            let base = round * 100;
            for i in 0u64..100 {
                btree.insert(root, base + i, vec![0u8; 32]).unwrap();
            }
            let start = base + 20;
            let end = base + 80;
            btree.remove_range(root, start..=end).unwrap();
            btree.assert_no_page_leak(root);
        }

        // Remove everything
        btree.remove_range(root, ..).unwrap();
        btree.assert_no_page_leak(root);
    }

    // ────────────────────────────────────────────────────────────────────
    //  Index tree tests
    // ────────────────────────────────────────────────────────────────────

    /// Create an initialized btree with an index tree root.
    fn new_index_btree(page_size: usize) -> (Btree, NodePtr) {
        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file, page_size);
        let mut btree = Btree::new(pager);
        btree.init().unwrap();
        let root = btree.init_index().unwrap();
        (btree, root)
    }

    /// Helper: read a key from the index tree by value.
    fn index_read_key(btree: &mut Btree, root: NodePtr, value: &[u8]) -> Option<Key> {
        let v = Value::Inline(value.to_vec());
        btree.index_read(root, &v, |k| k).unwrap()
    }

    #[test]
    fn test_index_insert_and_read() {
        let (mut btree, root) = new_index_btree(4096);
        assert!(btree.index_insert(root, 42, b"hello".to_vec()).unwrap());
        assert_eq!(index_read_key(&mut btree, root, b"hello"), Some(42));
    }

    #[test]
    fn test_index_insert_duplicate_returns_false() {
        let (mut btree, root) = new_index_btree(4096);
        assert!(btree.index_insert(root, 1, b"val".to_vec()).unwrap());
        assert!(!btree.index_insert(root, 1, b"val".to_vec()).unwrap());
    }

    #[test]
    fn test_index_same_value_different_keys() {
        let (mut btree, root) = new_index_btree(4096);
        assert!(btree.index_insert(root, 1, b"dup".to_vec()).unwrap());
        assert!(btree.index_insert(root, 2, b"dup".to_vec()).unwrap());
        assert!(btree.index_insert(root, 3, b"dup".to_vec()).unwrap());

        // index_read returns the first match
        let key = index_read_key(&mut btree, root, b"dup");
        assert!(key == Some(1) || key == Some(2) || key == Some(3));
    }

    #[test]
    fn test_index_read_absent() {
        let (mut btree, root) = new_index_btree(4096);
        btree.index_insert(root, 1, b"exists".to_vec()).unwrap();
        assert_eq!(index_read_key(&mut btree, root, b"missing"), None);
    }

    #[test]
    fn test_index_remove() {
        let (mut btree, root) = new_index_btree(4096);
        btree.index_insert(root, 1, b"alpha".to_vec()).unwrap();
        btree.index_insert(root, 2, b"beta".to_vec()).unwrap();

        let v = Value::Inline(b"alpha".to_vec());
        assert!(btree.index_remove(root, &v, 1).unwrap());
        assert_eq!(index_read_key(&mut btree, root, b"alpha"), None);
        assert_eq!(index_read_key(&mut btree, root, b"beta"), Some(2));
    }

    #[test]
    fn test_index_remove_absent() {
        let (mut btree, root) = new_index_btree(4096);
        btree.index_insert(root, 1, b"data".to_vec()).unwrap();

        let v = Value::Inline(b"nonexistent".to_vec());
        assert!(!btree.index_remove(root, &v, 1).unwrap());
    }

    #[test]
    fn test_index_remove_wrong_key_same_value() {
        let (mut btree, root) = new_index_btree(4096);
        btree.index_insert(root, 1, b"shared".to_vec()).unwrap();
        btree.index_insert(root, 2, b"shared".to_vec()).unwrap();

        // Remove key=99 which doesn't exist for this value
        let v = Value::Inline(b"shared".to_vec());
        assert!(!btree.index_remove(root, &v, 99).unwrap());

        // Both originals still present
        let key = index_read_key(&mut btree, root, b"shared");
        assert!(key.is_some());
    }

    #[test]
    fn test_index_multiple_values_sorted() {
        let (mut btree, root) = new_index_btree(256);
        let values: Vec<(&[u8], Key)> = vec![
            (b"cherry", 3),
            (b"apple", 1),
            (b"banana", 2),
            (b"date", 4),
            (b"elderberry", 5),
        ];
        for (v, k) in &values {
            btree.index_insert(root, *k, v.to_vec()).unwrap();
        }

        // All values should be findable
        for (v, k) in &values {
            assert_eq!(index_read_key(&mut btree, root, v), Some(*k));
        }
    }

    #[test]
    fn test_index_insert_many_and_read() {
        let (mut btree, root) = new_index_btree(4096);
        let mut map = HashMap::new();

        for i in 0u64..256 {
            let value = format!("value-{:04}", i).as_bytes().to_vec();
            btree.index_insert(root, i, value.clone()).unwrap();
            map.insert(value, i);
        }

        for (value, key) in &map {
            let found = index_read_key(&mut btree, root, value);
            assert_eq!(found, Some(*key), "Failed to find value {:?}", value);
        }
    }

    #[test]
    fn test_index_insert_remove_seq() {
        const LEN: u64 = 200;
        let mut rng = rand::rngs::StdRng::seed_from_u64(77);
        let (mut btree, root) = new_index_btree(256);

        let mut entries: Vec<(u64, Vec<u8>)> = (0..LEN)
            .map(|i| (i, format!("val-{:04}", i).into_bytes()))
            .collect();
        entries.shuffle(&mut rng);

        for (k, v) in &entries {
            assert!(btree.index_insert(root, *k, v.clone()).unwrap());
        }

        // Verify all present
        for (k, v) in &entries {
            assert_eq!(index_read_key(&mut btree, root, v), Some(*k));
        }

        entries.shuffle(&mut rng);
        for (k, v) in &entries {
            let val = Value::Inline(v.clone());
            assert!(btree.index_remove(root, &val, *k).unwrap());
            assert_eq!(index_read_key(&mut btree, root, v), None);
        }
    }

    #[test]
    fn test_index_read_range() {
        let (mut btree, root) = new_index_btree(4096);

        btree.index_insert(root, 1, b"aaa".to_vec()).unwrap();
        btree.index_insert(root, 2, b"bbb".to_vec()).unwrap();
        btree.index_insert(root, 3, b"ccc".to_vec()).unwrap();
        btree.index_insert(root, 4, b"ddd".to_vec()).unwrap();
        btree.index_insert(root, 5, b"eee".to_vec()).unwrap();

        let mut results = Vec::new();
        btree
            .index_read_range(root, b"bbb".to_vec()..=b"ddd".to_vec(), |v, k| {
                results.push((v.to_vec(), k))
            })
            .unwrap();

        assert_eq!(results.len(), 3);
        assert!(results.iter().any(|(v, k)| v == b"bbb" && *k == 2));
        assert!(results.iter().any(|(v, k)| v == b"ccc" && *k == 3));
        assert!(results.iter().any(|(v, k)| v == b"ddd" && *k == 4));
    }

    #[test]
    fn test_index_read_range_unbounded() {
        let (mut btree, root) = new_index_btree(4096);
        for i in 0u64..10 {
            btree
                .index_insert(root, i, format!("{:02}", i).into_bytes())
                .unwrap();
        }

        let mut results = Vec::new();
        btree
            .index_read_range(root, .., |v, k| results.push((v.to_vec(), k)))
            .unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_index_read_range_empty() {
        let (mut btree, root) = new_index_btree(4096);
        btree.index_insert(root, 1, b"aaa".to_vec()).unwrap();

        let mut results = Vec::new();
        btree
            .index_read_range(root, b"zzz".to_vec()..=b"zzzz".to_vec(), |v, k| {
                results.push((v.to_vec(), k))
            })
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_index_available_key() {
        let (mut btree, root) = new_index_btree(4096);
        assert_eq!(btree.index_available_key(root).unwrap(), 0);

        btree.index_insert(root, 5, b"aaa".to_vec()).unwrap();
        assert_eq!(btree.index_available_key(root).unwrap(), 6);

        btree.index_insert(root, 10, b"bbb".to_vec()).unwrap();
        assert_eq!(btree.index_available_key(root).unwrap(), 11);

        btree.index_insert(root, 7, b"ccc".to_vec()).unwrap();
        assert_eq!(btree.index_available_key(root).unwrap(), 11);
    }

    #[test]
    fn test_index_free_tree() {
        let (mut btree, root) = new_index_btree(256);
        for i in 0u64..100 {
            btree
                .index_insert(root, i, format!("v-{}", i).into_bytes())
                .unwrap();
        }
        let pages_before = btree.pager.total_page_count();
        btree.free_index_tree(root).unwrap();

        // All tree pages should now be in the free list
        assert_ne!(btree.read_free_list_head().unwrap(), u64::MAX);

        // Reuse: create a new index tree; pages should be reused
        let root2 = btree.init_index().unwrap();
        for i in 0u64..100 {
            btree
                .index_insert(root2, i, format!("v-{}", i).into_bytes())
                .unwrap();
        }
        assert!(
            btree.pager.total_page_count() <= pages_before + 1,
            "Pages not reused after free_index_tree"
        );
    }

    #[test]
    fn test_index_big_values() {
        let (mut btree, root) = new_index_btree(256);
        let big_value = vec![42u8; 4096];
        assert!(btree.index_insert(root, 1, big_value.clone()).unwrap());

        let v = Value::Inline(big_value.clone());
        let found = btree.index_read(root, &v, |k| k).unwrap();
        assert_eq!(found, Some(1));
    }

    #[test]
    fn test_index_big_value_remove() {
        let (mut btree, root) = new_index_btree(256);
        let big_value = vec![42u8; 4096];
        btree.index_insert(root, 1, big_value.clone()).unwrap();

        let v = Value::Inline(big_value);
        assert!(btree.index_remove(root, &v, 1).unwrap());
    }

    #[test]
    fn test_index_no_leak_insert_only() {
        let (mut btree, root) = new_index_btree(256);
        for i in 0u64..256 {
            btree
                .index_insert(root, i, format!("v-{:04}", i).into_bytes())
                .unwrap();
        }
        btree.assert_no_page_leak_index(root);
    }

    #[test]
    fn test_index_no_leak_insert_and_remove_all() {
        let (mut btree, root) = new_index_btree(256);
        let entries: Vec<(u64, Vec<u8>)> = (0u64..200)
            .map(|i| (i, format!("v-{:04}", i).into_bytes()))
            .collect();

        for (k, v) in &entries {
            btree.index_insert(root, *k, v.clone()).unwrap();
        }
        for (k, v) in &entries {
            let val = Value::Inline(v.clone());
            btree.index_remove(root, &val, *k).unwrap();
        }
        btree.assert_no_page_leak_index(root);
    }

    #[test]
    fn test_index_no_leak_shuffle() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let (mut btree, root) = new_index_btree(256);

        let mut entries: Vec<(u64, Vec<u8>)> = (0u64..300)
            .map(|i| (i, format!("val-{:04}", i).into_bytes()))
            .collect();
        entries.shuffle(&mut rng);
        for (k, v) in &entries {
            btree.index_insert(root, *k, v.clone()).unwrap();
        }

        entries.shuffle(&mut rng);
        for (k, v) in &entries {
            let val = Value::Inline(v.clone());
            btree.index_remove(root, &val, *k).unwrap();
        }
        btree.assert_no_page_leak_index(root);
    }

    #[test]
    fn test_index_no_leak_big_values() {
        let (mut btree, root) = new_index_btree(4096);
        let entries: Vec<(u64, Vec<u8>)> = (0u64..10)
            .map(|i| (i, vec![i as u8; 8192 + i as usize * 100]))
            .collect();

        for (k, v) in &entries {
            btree.index_insert(root, *k, v.clone()).unwrap();
        }
        btree.assert_no_page_leak_index(root);

        for (k, v) in &entries {
            let val = Value::Inline(v.clone());
            btree.index_remove(root, &val, *k).unwrap();
        }
        btree.assert_no_page_leak_index(root);
    }

    #[test]
    fn test_index_sorted_order() {
        let (mut btree, root) = new_index_btree(256);

        // Insert values in reverse order
        let values: Vec<Vec<u8>> = (0u64..50)
            .rev()
            .map(|i| format!("{:04}", i).into_bytes())
            .collect();
        for (i, v) in values.iter().enumerate() {
            btree.index_insert(root, i as u64, v.clone()).unwrap();
        }

        // Range read should return values in sorted byte order
        let mut result = Vec::new();
        btree
            .index_read_range(root, .., |v, _k| result.push(v.to_vec()))
            .unwrap();

        let mut sorted = result.clone();
        sorted.sort();
        assert_eq!(
            result, sorted,
            "Index tree entries should be in sorted value order"
        );
    }

    #[test]
    fn test_index_read_range_many() {
        let (mut btree, root) = new_index_btree(256);

        for i in 0u64..200 {
            btree
                .index_insert(root, i, format!("{:04}", i).into_bytes())
                .unwrap();
        }

        let mut results = Vec::new();
        btree
            .index_read_range(root, b"0050".to_vec()..b"0150".to_vec(), |v, _k| {
                results.push(v.to_vec())
            })
            .unwrap();

        // Values "0050".."0150" (exclusive end) = 100 entries
        assert_eq!(results.len(), 100);
        for v in &results {
            assert!(v.as_slice() >= b"0050".as_slice() && v.as_slice() < b"0150".as_slice());
        }
    }
}
