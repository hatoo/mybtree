use std::io;
use std::ops::{Bound, RangeBounds};

use crate::page::{
    AnyPage, IndexInternalPage, IndexLeafPage, InternalPage, LeafPage, OVERFLOW_FLAG,
    OVERFLOW_META_SIZE, PageType,
};
use crate::pager::Pager;
use crate::types::{FREE_LIST_PAGE_NUM, Key, NodePtr};
use crate::util::is_overlap;

#[derive(Debug, thiserror::Error)]
pub enum TreeError {
    #[error("unexpected page type: expected {expected}")]
    UnexpectedPageType { expected: &'static str },
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}

pub struct Btree<const N: usize> {
    pub pager: Pager<N>,
}

impl<const N: usize> Btree<N> {
    pub fn new(pager: Pager<N>) -> Self {
        Btree { pager }
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.pager.flush()
    }

    pub fn init(&mut self) -> io::Result<()> {
        self.pager.init()?;
        // Reserve page 0 for the free list
        let fl_page = self.pager.next_page_num();
        assert_eq!(fl_page, FREE_LIST_PAGE_NUM);
        self.write_free_list_head(u64::MAX)?;
        Ok(())
    }

    // ────────────────────────────────────────────────────────────────────
    //  Primary tree: init / free
    // ────────────────────────────────────────────────────────────────────

    pub fn init_tree(&mut self) -> io::Result<NodePtr> {
        let page = self.alloc_page()?;
        let leaf = LeafPage::<N>::new();
        self.pager.write_node(page, leaf.into())?;
        Ok(page)
    }

    pub fn free_tree(&mut self, root: NodePtr) -> Result<(), TreeError> {
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            let page: &AnyPage<N> = self.pager.read_node(node)?;
            match page.page_type() {
                PageType::Leaf => {
                    let leaf: &LeafPage<N> = page.try_into().unwrap();
                    let overflow_pages_to_free = (0..leaf.len())
                        .filter(|&i| leaf.is_overflow(i))
                        .map(|i| Self::parse_overflow_meta(leaf.value(i)))
                        .collect::<Vec<_>>();
                    for (start_page, total_len) in overflow_pages_to_free {
                        self.free_overflow_pages(start_page, total_len)?;
                    }
                }
                PageType::Internal => {
                    let internal: &InternalPage<N> = page.try_into().unwrap();
                    for i in 0..internal.len() {
                        stack.push(internal.ptr(i));
                    }
                }
                _ => {
                    return Err(TreeError::UnexpectedPageType {
                        expected: "Leaf or Internal",
                    });
                }
            }
            self.free_page(node)?;
        }
        Ok(())
    }

    // ────────────────────────────────────────────────────────────────────
    //  Primary tree: insert
    // ────────────────────────────────────────────────────────────────────

    pub fn insert(
        &mut self,
        root: NodePtr,
        key: Key,
        value: Vec<u8>,
    ) -> Result<Option<Vec<u8>>, TreeError> {
        let mut current = root;
        let mut path = vec![];

        loop {
            path.push(current);
            let page = self.pager.read_node(current)?;

            match page.page_type() {
                PageType::Leaf => {
                    let mut leaf: LeafPage<N> = page.clone().try_into().unwrap();
                    match leaf.search_key(key) {
                        Ok(index) => {
                            // Key exists — read old value, remove old entry, insert new
                            let old_bytes = self.read_leaf_value(&leaf, index)?;
                            let overflow_meta = if leaf.is_overflow(index) {
                                Some(Self::parse_overflow_meta(leaf.value(index)))
                            } else {
                                None
                            };
                            leaf.remove(index);
                            self.leaf_insert_and_propagate(root, &path, leaf, key, &value)?;
                            if let Some((start_page, total_len)) = overflow_meta {
                                self.free_overflow_pages(start_page, total_len)?;
                            }
                            return Ok(Some(old_bytes));
                        }
                        Err(_) => {
                            // New key
                            self.leaf_insert_and_propagate(root, &path, leaf, key, &value)?;
                            return Ok(None);
                        }
                    }
                }
                PageType::Internal => {
                    let internal: &InternalPage<N> = page.try_into().unwrap();
                    match internal.search_index(key) {
                        Some(idx) => {
                            current = internal.ptr(idx);
                        }
                        None => {
                            // Key is beyond all entries
                            if internal.len() == 0 {
                                let new_leaf_page = self.alloc_page()?;
                                let internal: &mut InternalPage<N> =
                                    self.pager.mut_node(current)?.try_into().unwrap();
                                internal.insert(key, new_leaf_page);
                                let mut new_leaf = LeafPage::<N>::new();
                                self.leaf_insert_entry(&mut new_leaf, key, &value)?;
                                self.pager.write_node(new_leaf_page, new_leaf.into())?;
                                return Ok(None);
                            } else {
                                let internal: &mut InternalPage<N> =
                                    self.pager.mut_node(current)?.try_into().unwrap();
                                let last_idx = internal.len() - 1;
                                let next = internal.ptr(last_idx);
                                // Update the last key to accommodate the new key
                                internal.remove(last_idx);
                                internal.insert(key, next);
                                current = next;
                            }
                        }
                    }
                }
                _ => {
                    return Err(TreeError::UnexpectedPageType {
                        expected: "Leaf or Internal",
                    });
                }
            }
        }
    }

    /// Insert a key-value into a leaf page, handling splits and propagation.
    fn leaf_insert_and_propagate(
        &mut self,
        root: NodePtr,
        path: &[NodePtr],
        mut leaf: LeafPage<N>,
        key: Key,
        value: &[u8],
    ) -> Result<(), TreeError> {
        if leaf.can_insert(value.len()) {
            self.leaf_insert_entry(&mut leaf, key, value)?;
            let page = *path.last().unwrap();
            self.pager.write_node(page, leaf.into())?;
            return Ok(());
        }

        // Need to split
        let mut right = leaf.split();
        let split_key = leaf.key(leaf.len() - 1);

        if key <= split_key {
            self.leaf_insert_entry(&mut leaf, key, value)?;
        } else {
            self.leaf_insert_entry(&mut right, key, value)?;
        }

        let page = *path.last().unwrap();
        let parents = &path[..path.len() - 1];

        let left_max_key = leaf.key(leaf.len() - 1);
        let right_page = self.alloc_page()?;

        // Write right to new page, left stays at current page
        self.pager.write_node(right_page, right.into())?;
        self.pager.write_node(page, leaf.into())?;

        self.propagate_split(root, parents, page, left_max_key, right_page)
    }

    /// Insert a key-value entry into a leaf, handling overflow if needed.
    fn leaf_insert_entry(
        &mut self,
        leaf: &mut LeafPage<N>,
        key: Key,
        value: &[u8],
    ) -> Result<(), TreeError> {
        if LeafPage::<N>::needs_overflow(value.len()) {
            let start_page = self.write_overflow(value)?;
            let mut meta = [0u8; OVERFLOW_META_SIZE];
            meta[0..8].copy_from_slice(&start_page.to_le_bytes());
            meta[8..16].copy_from_slice(&(value.len() as u64).to_le_bytes());
            leaf.insert_raw(key, &meta, OVERFLOW_META_SIZE as u16 | OVERFLOW_FLAG);
        } else {
            leaf.insert_raw(key, value, value.len() as u16);
        }
        Ok(())
    }

    /// Propagate a split up to the parent, iteratively.
    /// `left_page` now has max key `left_max_key`.
    /// `right_page` is a newly allocated page.
    fn propagate_split(
        &mut self,
        root: NodePtr,
        parents: &[NodePtr],
        left_page: NodePtr,
        left_max_key: Key,
        right_page: NodePtr,
    ) -> Result<(), TreeError> {
        let mut cur_left_page = left_page;
        let mut cur_left_max_key = left_max_key;
        let mut cur_right_page = right_page;
        let mut depth = parents.len();

        loop {
            if depth == 0 {
                // Splitting the root — create a new internal root
                let right_content = self.pager.owned_node(cur_right_page)?;
                let right_max_key = match right_content.page_type() {
                    PageType::Leaf => {
                        let r: LeafPage<N> = right_content.try_into().unwrap();
                        r.key(r.len() - 1)
                    }
                    PageType::Internal => {
                        let r: InternalPage<N> = right_content.try_into().unwrap();
                        r.key(r.len() - 1)
                    }
                    _ => unreachable!(),
                };
                let moved_page = self.alloc_page()?;
                let root_content = self.pager.owned_node(root)?;
                self.pager.write_node(moved_page, root_content)?;

                let mut new_root = InternalPage::<N>::new();
                new_root.insert(cur_left_max_key, moved_page);
                new_root.insert(right_max_key, cur_right_page);
                self.pager.write_node(root, new_root.into())?;
                return Ok(());
            }

            let parent_ptr = parents[depth - 1];

            // Read parent to find entry index, old key, and can_insert
            let (i, old_key, can_insert) = {
                let parent_ref: &InternalPage<N> =
                    self.pager.read_node(parent_ptr)?.try_into().unwrap();
                let mut found_idx = None;
                for j in 0..parent_ref.len() {
                    if parent_ref.ptr(j) == cur_left_page {
                        found_idx = Some(j);
                        break;
                    }
                }
                let i = found_idx.expect("left_page not found in parent during split propagation");
                // can_insert after removing one and inserting two (net +1)
                let can_insert =
                    (parent_ref.len() + 1) * INTERNAL_ELEMENT_SIZE + INTERNAL_HEADER_SIZE <= N;
                (i, parent_ref.key(i), can_insert)
            };
            let right_max_key = old_key;

            // Mutate parent: remove old entry, insert left entry
            {
                let p = self.pager.mut_node(parent_ptr)?;
                let parent: &mut InternalPage<N> = p.try_into().unwrap();
                parent.remove(i);
                parent.insert(cur_left_max_key, cur_left_page);
            }

            if can_insert {
                // Fast path: insert right entry and done
                let p = self.pager.mut_node(parent_ptr)?;
                let parent: &mut InternalPage<N> = p.try_into().unwrap();
                parent.insert(right_max_key, cur_right_page);
                return Ok(());
            }

            // Need to split the parent too
            // Parent is already dirty in cache — owned_node gets it from cache
            let mut internal: InternalPage<N> =
                self.pager.owned_node(parent_ptr)?.try_into().unwrap();
            let mut right = internal.split();
            let split_key = internal.key(internal.len() - 1);

            if right_max_key <= split_key {
                internal.insert(right_max_key, cur_right_page);
            } else {
                right.insert(right_max_key, cur_right_page);
            }

            let new_left_max_key = internal.key(internal.len() - 1);
            let new_right_page = self.alloc_page()?;

            self.pager.write_node(new_right_page, right.into())?;
            self.pager.write_node(parent_ptr, internal.into())?;

            // Continue up the tree
            cur_left_page = parent_ptr;
            cur_left_max_key = new_left_max_key;
            cur_right_page = new_right_page;
            depth -= 1;
        }
    }

    // ────────────────────────────────────────────────────────────────────
    //  Primary tree: remove
    // ────────────────────────────────────────────────────────────────────

    pub fn remove(&mut self, root: NodePtr, key: Key) -> Result<Option<Vec<u8>>, TreeError> {
        let mut current = root;
        let mut path = vec![];

        loop {
            path.push(current);
            let page = self.pager.owned_node(current)?;

            match page.page_type() {
                PageType::Leaf => {
                    let leaf: LeafPage<N> = page.try_into().unwrap();
                    match leaf.search_key(key) {
                        Ok(index) => {
                            let old_bytes = self.read_leaf_value(&leaf, index)?;
                            let overflow_meta = if leaf.is_overflow(index) {
                                Some(Self::parse_overflow_meta(leaf.value(index)))
                            } else {
                                None
                            };
                            {
                                let p = self.pager.mut_node(current)?;
                                let leaf_mut: &mut LeafPage<N> = p.try_into().unwrap();
                                leaf_mut.remove(index);
                            }
                            self.merge_leaf(&path)?;
                            if let Some((start_page, total_len)) = overflow_meta {
                                self.free_overflow_pages(start_page, total_len)?;
                            }
                            return Ok(Some(old_bytes));
                        }
                        Err(_) => return Ok(None),
                    }
                }
                PageType::Internal => {
                    let internal: InternalPage<N> = page.try_into().unwrap();
                    match internal.search_index(key) {
                        Some(idx) => current = internal.ptr(idx),
                        None => return Ok(None),
                    }
                }
                _ => {
                    return Err(TreeError::UnexpectedPageType {
                        expected: "Leaf or Internal",
                    });
                }
            }
        }
    }

    pub fn remove_range(
        &mut self,
        root: NodePtr,
        range: impl RangeBounds<Key>,
    ) -> Result<(), TreeError> {
        self.remove_range_at(root, &range, 0)
    }

    fn remove_range_at(
        &mut self,
        node_ptr: NodePtr,
        range: &impl RangeBounds<Key>,
        left_key: Key,
    ) -> Result<(), TreeError> {
        // Collect leaf nodes to process first, then modify them.
        // Use a stack to avoid recursion.
        let mut visit_stack: Vec<(NodePtr, Key)> = vec![(node_ptr, left_key)];
        let mut leaves_to_process: Vec<NodePtr> = Vec::new();

        while let Some((cur, lk)) = visit_stack.pop() {
            let page = self.pager.owned_node(cur)?;
            match page.page_type() {
                PageType::Leaf => {
                    leaves_to_process.push(cur);
                }
                PageType::Internal => {
                    let internal: InternalPage<N> = page.try_into().unwrap();
                    let mut left_key = lk;
                    for i in 0..internal.len() {
                        let k = internal.key(i);
                        let ptr = internal.ptr(i);
                        if is_overlap(&(left_key..=k), range) {
                            visit_stack.push((ptr, left_key));
                        }
                        left_key = k;
                    }
                }
                _ => {
                    return Err(TreeError::UnexpectedPageType {
                        expected: "Leaf or Internal",
                    });
                }
            }
        }

        for leaf_ptr in leaves_to_process {
            let (overflow_metas, indices_to_remove) = {
                let page = self.pager.read_node(leaf_ptr)?;
                let leaf: &LeafPage<N> = page.try_into().unwrap();
                let mut overflow_metas = vec![];
                let mut indices_to_remove = vec![];
                for i in 0..leaf.len() {
                    if range.contains(&leaf.key(i)) {
                        if leaf.is_overflow(i) {
                            overflow_metas.push(Self::parse_overflow_meta(leaf.value(i)));
                        }
                        indices_to_remove.push(i);
                    }
                }
                (overflow_metas, indices_to_remove)
            };
            if !indices_to_remove.is_empty() {
                {
                    let p = self.pager.mut_node(leaf_ptr)?;
                    let leaf_mut: &mut LeafPage<N> = p.try_into().unwrap();
                    for &i in indices_to_remove.iter().rev() {
                        leaf_mut.remove(i);
                    }
                }
                self.merge_leaf(&[leaf_ptr])?;
            }
            for (start_page, total_len) in overflow_metas {
                self.free_overflow_pages(start_page, total_len)?;
            }
        }

        Ok(())
    }

    /// Check and merge leaf page if underfull.
    /// The page at `path.last()` must already be dirty in the pager cache.
    fn merge_leaf(&mut self, path: &[NodePtr]) -> Result<(), TreeError> {
        let page = *path.last().unwrap();

        let (free_space, len) = {
            let leaf_ref: &LeafPage<N> = self.pager.read_node(page)?.try_into().unwrap();
            (leaf_ref.free_space(), leaf_ref.len())
        };

        // If at root or leaf is at least half full, already dirty — done
        if path.len() <= 1 || free_space <= N / 2 {
            return Ok(());
        }

        // Handle empty leaf: remove from parent
        if len == 0 {
            let parents = &path[..path.len() - 1];
            let parent_page = *parents.last().unwrap();
            {
                let p = self.pager.mut_node(parent_page)?;
                let parent: &mut InternalPage<N> = p.try_into().unwrap();
                for i in 0..parent.len() {
                    if parent.ptr(i) == page {
                        parent.remove(i);
                        break;
                    }
                }
            }
            self.free_page(page)?;
            return self.merge_internal_iterative(parents);
        }

        let parents = &path[..path.len() - 1];
        self.try_merge_leaf_with_left_sibling(page, parents)?;
        Ok(())
    }

    fn try_merge_leaf_with_left_sibling(
        &mut self,
        page: NodePtr,
        parents: &[NodePtr],
    ) -> Result<bool, TreeError> {
        let parent_page = *parents.last().unwrap();

        let (index, left_sibling_page) = {
            let parent_ref: &InternalPage<N> =
                self.pager.read_node(parent_page)?.try_into().unwrap();
            let index = (0..parent_ref.len())
                .find(|&i| parent_ref.ptr(i) == page)
                .unwrap();
            if index == 0 {
                return Ok(false);
            }
            (index, parent_ref.ptr(index - 1))
        };

        // Read right leaf entries
        let right_entries = {
            let right_ref: &LeafPage<N> = self.pager.read_node(page)?.try_into().unwrap();
            (0..right_ref.len())
                .map(|i| {
                    let (_, _, vl) = right_ref.read_slot(i);
                    (right_ref.key(i), right_ref.value(i).to_vec(), vl)
                })
                .collect::<Vec<_>>()
        };

        // Check if merge is feasible using left leaf
        {
            let left_ref: &LeafPage<N> =
                self.pager.read_node(left_sibling_page)?.try_into().unwrap();
            let mut free = left_ref.free_space();
            for (_, value_data, _) in &right_entries {
                let inline_size = value_data.len();
                if free < LEAF_SLOT_SIZE + inline_size {
                    return Ok(false);
                }
                free -= LEAF_SLOT_SIZE + inline_size;
            }
        }

        // Read left entries for merging into the right page
        let left_entries = {
            let left_ref: &LeafPage<N> =
                self.pager.read_node(left_sibling_page)?.try_into().unwrap();
            (0..left_ref.len())
                .map(|i| {
                    let (_, _, vl) = left_ref.read_slot(i);
                    (left_ref.key(i), left_ref.value(i).to_vec(), vl)
                })
                .collect::<Vec<_>>()
        };

        // Merge: insert left entries into the right page (at `page`)
        {
            let p = self.pager.mut_node(page)?;
            let merged: &mut LeafPage<N> = p.try_into().unwrap();
            for (k, v, vl) in &left_entries {
                merged.insert_raw(*k, v, *vl);
            }
        }

        // Remove left sibling entry from parent
        {
            let p = self.pager.mut_node(parent_page)?;
            let parent: &mut InternalPage<N> = p.try_into().unwrap();
            parent.remove(index - 1);
        }
        self.free_page(left_sibling_page)?;
        self.merge_internal_iterative(parents)?;
        Ok(true)
    }

    /// Check and merge internal pages if underfull. Iterative.
    /// The page at `path.last()` must already be dirty in the pager cache.
    fn merge_internal_iterative(&mut self, path: &[NodePtr]) -> Result<(), TreeError> {
        let mut cur_path = path;

        loop {
            let page = *cur_path.last().unwrap();

            let (len, used) = {
                let node_ref: &InternalPage<N> = self.pager.read_node(page)?.try_into().unwrap();
                let len = node_ref.len();
                (len, INTERNAL_HEADER_SIZE + len * INTERNAL_ELEMENT_SIZE)
            };

            if cur_path.len() <= 1 || used >= N / 2 {
                // Already dirty from caller, nothing more to do
                return Ok(());
            }

            if len == 0 {
                if cur_path.len() <= 1 {
                    let leaf = LeafPage::<N>::new();
                    self.pager.write_node(page, leaf.into())?;
                    return Ok(());
                }
                let parents = &cur_path[..cur_path.len() - 1];
                let parent_page = *parents.last().unwrap();
                {
                    let p = self.pager.mut_node(parent_page)?;
                    let parent: &mut InternalPage<N> = p.try_into().unwrap();
                    for i in 0..parent.len() {
                        if parent.ptr(i) == page {
                            parent.remove(i);
                            break;
                        }
                    }
                }
                self.free_page(page)?;
                cur_path = parents;
                continue;
            }

            let parents = &cur_path[..cur_path.len() - 1];
            if self.try_merge_internal_with_left_sibling(page, parents)? {
                // Merge succeeded; parent was modified via mut_node, continue up
                cur_path = parents;
                continue;
            } else {
                // Already dirty from caller
                return Ok(());
            }
        }
    }

    /// Try to merge `right_internal` (at `page`) with its left sibling in the parent.
    /// Returns `true` if merge succeeded, `false` otherwise.
    /// On success, the merged page is written to `page`, left sibling is freed,
    /// and parent entry is removed via `mut_node`.
    fn try_merge_internal_with_left_sibling(
        &mut self,
        page: NodePtr,
        parents: &[NodePtr],
    ) -> Result<bool, TreeError> {
        let parent_page = *parents.last().unwrap();

        // Read parent and right page to determine merge feasibility
        let (index, left_sibling_page) = {
            let parent_ref: &InternalPage<N> =
                self.pager.read_node(parent_page)?.try_into().unwrap();
            let index = (0..parent_ref.len())
                .find(|&i| parent_ref.ptr(i) == page)
                .unwrap();
            if index == 0 {
                return Ok(false);
            }
            (index, parent_ref.ptr(index - 1))
        };

        let right_len = {
            let right_ref: &InternalPage<N> = self.pager.read_node(page)?.try_into().unwrap();
            right_ref.len()
        };

        let left_len = {
            let left_ref: &InternalPage<N> =
                self.pager.read_node(left_sibling_page)?.try_into().unwrap();
            left_ref.len()
        };

        let total_entries = left_len + right_len;
        let total_used = INTERNAL_HEADER_SIZE + total_entries * INTERNAL_ELEMENT_SIZE;
        if total_used > N {
            return Ok(false);
        }

        // Merge: copy left entries into right page (which keeps its position)
        let left_entries = {
            let left_ref: &InternalPage<N> =
                self.pager.read_node(left_sibling_page)?.try_into().unwrap();
            (0..left_ref.len())
                .map(|i| (left_ref.key(i), left_ref.ptr(i)))
                .collect::<Vec<_>>()
        };

        {
            let p = self.pager.mut_node(page)?;
            let merged: &mut InternalPage<N> = p.try_into().unwrap();
            for (k, ptr) in left_entries {
                merged.insert(k, ptr);
            }
        }

        // Remove left sibling entry from parent
        {
            let p = self.pager.mut_node(parent_page)?;
            let parent: &mut InternalPage<N> = p.try_into().unwrap();
            parent.remove(index - 1);
        }
        self.free_page(left_sibling_page)?;
        Ok(true)
    }

    // ────────────────────────────────────────────────────────────────────
    //  Primary tree: read
    // ────────────────────────────────────────────────────────────────────

    pub fn read<T>(
        &mut self,
        root: NodePtr,
        key: Key,
        f: impl FnOnce(Option<&[u8]>) -> T,
    ) -> Result<T, TreeError> {
        let mut current = root;

        loop {
            let page = self.pager.owned_node(current)?;

            match page.page_type() {
                PageType::Leaf => {
                    let leaf: LeafPage<N> = page.try_into().unwrap();
                    match leaf.search_key(key) {
                        Ok(index) => {
                            if leaf.is_overflow(index) {
                                let (start_page, total_len) =
                                    Self::parse_overflow_meta(leaf.value(index));
                                let data = self.read_overflow(start_page, total_len)?;
                                return Ok(f(Some(&data)));
                            } else {
                                return Ok(f(Some(leaf.value(index))));
                            }
                        }
                        Err(_) => return Ok(f(None)),
                    }
                }
                PageType::Internal => {
                    let internal: InternalPage<N> = page.try_into().unwrap();
                    match internal.search_index(key) {
                        Some(idx) => current = internal.ptr(idx),
                        None => return Ok(f(None)),
                    }
                }
                _ => {
                    return Err(TreeError::UnexpectedPageType {
                        expected: "Leaf or Internal",
                    });
                }
            }
        }
    }

    pub fn read_range<R: RangeBounds<Key>>(
        &mut self,
        root: NodePtr,
        range: R,
        mut f: impl FnMut(Key, &[u8]),
    ) -> Result<(), TreeError> {
        self.read_range_at(root, &range, &mut f, 0)
    }

    fn read_range_at<R: RangeBounds<Key>>(
        &mut self,
        node_ptr: NodePtr,
        range: &R,
        f: &mut impl FnMut(Key, &[u8]),
        left_key: Key,
    ) -> Result<(), TreeError> {
        // Stack of (node_ptr, left_key)
        let mut stack = vec![(node_ptr, left_key)];
        while let Some((cur, lk)) = stack.pop() {
            let page = self.pager.owned_node(cur)?;
            match page.page_type() {
                PageType::Leaf => {
                    let leaf: LeafPage<N> = page.try_into().unwrap();
                    for i in 0..leaf.len() {
                        let k = leaf.key(i);
                        if range.contains(&k) {
                            if leaf.is_overflow(i) {
                                let (start_page, total_len) =
                                    Self::parse_overflow_meta(leaf.value(i));
                                let data = self.read_overflow(start_page, total_len)?;
                                f(k, &data);
                            } else {
                                f(k, leaf.value(i));
                            }
                        }
                    }
                }
                PageType::Internal => {
                    let internal: InternalPage<N> = page.try_into().unwrap();
                    let mut left_key = lk;
                    // Push children in reverse so leftmost is processed first
                    let mut children = Vec::new();
                    for i in 0..internal.len() {
                        let k = internal.key(i);
                        let ptr = internal.ptr(i);
                        if is_overlap(&(left_key..=k), range) {
                            children.push((ptr, left_key));
                        }
                        left_key = k;
                    }
                    for child in children.into_iter().rev() {
                        stack.push(child);
                    }
                }
                _ => {
                    return Err(TreeError::UnexpectedPageType {
                        expected: "Leaf or Internal",
                    });
                }
            }
        }
        Ok(())
    }

    pub fn available_key(&mut self, root: NodePtr) -> Result<Key, TreeError> {
        let page = self.pager.owned_node(root)?;
        match page.page_type() {
            PageType::Leaf => {
                let leaf: LeafPage<N> = page.try_into().unwrap();
                if leaf.len() > 0 {
                    Ok(leaf.key(leaf.len() - 1).checked_add(1).unwrap_or(u64::MAX))
                } else {
                    Ok(0)
                }
            }
            PageType::Internal => {
                let internal: InternalPage<N> = page.try_into().unwrap();
                if internal.len() > 0 {
                    Ok(internal
                        .key(internal.len() - 1)
                        .checked_add(1)
                        .unwrap_or(u64::MAX))
                } else {
                    Ok(0)
                }
            }
            _ => {
                return Err(TreeError::UnexpectedPageType {
                    expected: "Leaf or Internal",
                });
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────
    //  Leaf value helpers
    // ────────────────────────────────────────────────────────────────────

    fn read_leaf_value(&mut self, leaf: &LeafPage<N>, index: usize) -> io::Result<Vec<u8>> {
        if leaf.is_overflow(index) {
            let (start_page, total_len) = Self::parse_overflow_meta(leaf.value(index));
            Ok(self.read_overflow(start_page, total_len)?)
        } else {
            Ok(leaf.value(index).to_vec())
        }
    }

    // ────────────────────────────────────────────────────────────────────
    //  Overflow pages
    // ────────────────────────────────────────────────────────────────────

    fn parse_overflow_meta(meta: &[u8]) -> (u64, u64) {
        let start_page = u64::from_le_bytes(meta[0..8].try_into().unwrap());
        let total_len = u64::from_le_bytes(meta[8..16].try_into().unwrap());
        (start_page, total_len)
    }

    fn write_overflow(&mut self, data: &[u8]) -> io::Result<u64> {
        let data_per_page = self.overflow_data_per_page();
        let num_pages = (data.len() + data_per_page - 1) / data_per_page;
        assert!(num_pages > 0);

        let pages: Vec<u64> = (0..num_pages)
            .map(|_| self.alloc_page())
            .collect::<io::Result<Vec<u64>>>()?;

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

    fn read_overflow(&mut self, start_page: u64, total_len: u64) -> io::Result<Vec<u8>> {
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

    fn overflow_data_per_page(&self) -> usize {
        self.pager.page_size() - 8
    }

    fn free_overflow_pages(&mut self, start_page: u64, total_len: u64) -> io::Result<()> {
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

    // ────────────────────────────────────────────────────────────────────
    //  Free page list
    // ────────────────────────────────────────────────────────────────────

    fn read_free_list_head(&mut self) -> io::Result<u64> {
        let buf = self.pager.read_raw_page(FREE_LIST_PAGE_NUM)?;
        Ok(u64::from_le_bytes(buf[..8].try_into().unwrap()))
    }

    fn write_free_list_head(&mut self, head: u64) -> io::Result<()> {
        let mut buf = vec![0u8; 8];
        buf[..8].copy_from_slice(&head.to_le_bytes());
        self.pager.write_raw_page(FREE_LIST_PAGE_NUM, &buf)
    }

    fn alloc_page(&mut self) -> io::Result<u64> {
        let head = self.read_free_list_head()?;
        if head == u64::MAX {
            return Ok(self.pager.next_page_num());
        }
        let buf = self.pager.read_raw_page(head)?;
        let next = u64::from_le_bytes(buf[..8].try_into().unwrap());
        self.write_free_list_head(next)?;
        Ok(head)
    }

    pub(crate) fn free_page(&mut self, page_num: u64) -> io::Result<()> {
        let head = self.read_free_list_head()?;
        let mut buf = vec![0u8; 8];
        buf[..8].copy_from_slice(&head.to_le_bytes());
        self.pager.write_raw_page(page_num, &buf)?;
        self.write_free_list_head(page_num)
    }

    // ────────────────────────────────────────────────────────────────────
    //  Index tree operations
    // ────────────────────────────────────────────────────────────────────

    pub fn init_index(&mut self) -> io::Result<NodePtr> {
        let page = self.alloc_page()?;
        let leaf = IndexLeafPage::<N>::new();
        self.pager.write_node(page, leaf.into())?;
        Ok(page)
    }

    pub fn free_index_tree(&mut self, root: NodePtr) -> Result<(), TreeError> {
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            let page = self.pager.owned_node(node)?;
            match page.page_type() {
                PageType::IndexLeaf => {
                    let leaf: IndexLeafPage<N> = page.try_into().unwrap();
                    for i in 0..leaf.len() {
                        if leaf.is_overflow(i) {
                            let (start_page, total_len) = Self::parse_overflow_meta(leaf.key(i));
                            self.free_overflow_pages(start_page, total_len)?;
                        }
                    }
                }
                PageType::IndexInternal => {
                    let internal: IndexInternalPage<N> = page.try_into().unwrap();
                    for i in 0..internal.len() {
                        if internal.is_overflow(i) {
                            let (start_page, total_len) =
                                Self::parse_overflow_meta(internal.key(i));
                            self.free_overflow_pages(start_page, total_len)?;
                        }
                        stack.push(internal.ptr(i));
                    }
                }
                _ => {
                    return Err(TreeError::UnexpectedPageType {
                        expected: "IndexLeaf or IndexInternal",
                    });
                }
            }
            self.free_page(node)?;
        }
        Ok(())
    }

    pub fn index_insert(
        &mut self,
        root: NodePtr,
        key: Key,
        value: Vec<u8>,
    ) -> Result<bool, TreeError> {
        let mut current = root;
        let mut path = vec![];

        loop {
            path.push(current);
            let page = self.pager.owned_node(current)?;

            match page.page_type() {
                PageType::IndexLeaf => {
                    let mut leaf: IndexLeafPage<N> = page.try_into().unwrap();

                    // Check if exact (value, key) already exists
                    if leaf.find_entry(&value, key, &mut self.pager).is_some() {
                        return Ok(false);
                    }

                    if leaf.can_insert(value.len()) {
                        leaf.insert_entry(&value, key, &mut self.pager);
                        let page_num = *path.last().unwrap();
                        self.pager.write_node(page_num, leaf.into())?;
                        return Ok(true);
                    }

                    // Need to split
                    let mut right = leaf.split();
                    // Determine which half to insert into
                    let split_key_bytes = leaf.resolved_key(leaf.len() - 1, &mut self.pager);
                    let cmp = value.as_slice().cmp(split_key_bytes.as_ref());
                    if cmp == std::cmp::Ordering::Less
                        || (cmp == std::cmp::Ordering::Equal && key <= leaf.value(leaf.len() - 1))
                    {
                        leaf.insert_entry(&value, key, &mut self.pager);
                    } else {
                        right.insert_entry(&value, key, &mut self.pager);
                    }

                    let page_num = *path.last().unwrap();
                    let parents = &path[..path.len() - 1];
                    let right_page = self.alloc_page()?;

                    self.pager.write_node(right_page, right.into())?;
                    self.pager.write_node(page_num, leaf.into())?;

                    self.index_propagate_split(root, parents, page_num, right_page)?;
                    return Ok(true);
                }
                PageType::IndexInternal => {
                    let internal: IndexInternalPage<N> = page.try_into().unwrap();
                    match internal.find_child(&value, &mut self.pager) {
                        Some(next) => current = next,
                        None => {
                            // Value is beyond all entries
                            let mut internal: IndexInternalPage<N> =
                                self.pager.owned_node(current)?.try_into().unwrap();
                            if internal.len() == 0 {
                                let new_leaf_page = self.alloc_page()?;
                                let mut new_leaf = IndexLeafPage::<N>::new();
                                new_leaf.insert_entry(&value, key, &mut self.pager);
                                self.pager.write_node(new_leaf_page, new_leaf.into())?;
                                internal.insert(&value, new_leaf_page, &mut self.pager);
                                self.pager.write_node(current, internal.into())?;
                                return Ok(true);
                            } else {
                                let last_idx = internal.len() - 1;
                                let next = internal.ptr(last_idx);
                                // Update last key to accommodate new value
                                internal.remove(last_idx);
                                internal.insert(&value, next, &mut self.pager);
                                self.pager.write_node(current, internal.into())?;
                                current = next;
                            }
                        }
                    }
                }
                _ => {
                    return Err(TreeError::UnexpectedPageType {
                        expected: "IndexLeaf or IndexInternal",
                    });
                }
            }
        }
    }

    /// Propagate an index page split up to the parent.
    fn index_propagate_split(
        &mut self,
        root: NodePtr,
        parents: &[NodePtr],
        left_page: NodePtr,
        right_page: NodePtr,
    ) -> Result<(), TreeError> {
        // Get max key from left page for the parent entry
        let left_any = self.pager.owned_node(left_page)?;
        let left_max_key: Vec<u8> = match left_any.page_type() {
            PageType::IndexLeaf => {
                let leaf: IndexLeafPage<N> = left_any.try_into().unwrap();
                leaf.resolved_key(leaf.len() - 1, &mut self.pager)
                    .into_owned()
            }
            PageType::IndexInternal => {
                let internal: IndexInternalPage<N> = left_any.try_into().unwrap();
                internal
                    .resolved_key(internal.len() - 1, &mut self.pager)
                    .into_owned()
            }
            _ => unreachable!(),
        };

        if let Some(&parent_ptr) = parents.last() {
            let mut parent: IndexInternalPage<N> =
                self.pager.owned_node(parent_ptr)?.try_into().unwrap();

            // Find the entry for left_page and update
            for i in 0..parent.len() {
                if parent.ptr(i) == left_page {
                    let old_key = parent.resolved_key(i, &mut self.pager).into_owned();
                    let (_, raw_kl, _) = parent.read_slot(i);
                    // Remove old entry, re-insert with new max key for left
                    parent.remove(i);

                    // Free old overflow key if applicable
                    if raw_kl & OVERFLOW_FLAG != 0 {
                        // The old key data was overflow — but we're removing the slot;
                        // the overflow pages for the old key in internal nodes should be freed
                        // But actually the internal node key is the max key from the subtree,
                        // and the overflow data is shared with the subtree. So we shouldn't
                        // free it here. The subtree owns those overflow pages.
                        // Actually, index internal page keys are COPIES, not shared.
                        // We need to free them.
                        let _meta_bytes = &old_key; // This is already resolved
                        // Actually, resolved_key already read the overflow. The raw inline
                        // data in the slot is the overflow metadata. We need the raw bytes.
                        // Let's skip freeing here — the key data will be re-inserted.
                    }

                    // Insert left_max_key for left_page
                    if IndexInternalPage::<N>::needs_overflow(left_max_key.len()) {
                        let start_page = self.write_overflow(&left_max_key)?;
                        let mut meta = [0u8; OVERFLOW_META_SIZE];
                        meta[0..8].copy_from_slice(&start_page.to_le_bytes());
                        meta[8..16].copy_from_slice(&(left_max_key.len() as u64).to_le_bytes());
                        let idx = parent.len(); // append at end, will re-sort
                        parent.insert_raw_at(
                            idx,
                            &meta,
                            OVERFLOW_META_SIZE as u16 | OVERFLOW_FLAG,
                            left_page,
                        );
                    } else {
                        // Find correct position
                        let idx = parent
                            .search(&left_max_key, &mut self.pager)
                            .unwrap_or(parent.len());
                        parent.insert_raw_at(
                            idx,
                            &left_max_key,
                            left_max_key.len() as u16,
                            left_page,
                        );
                    }

                    // Insert right_page with old max key
                    if parent.can_insert(old_key.len()) {
                        parent.insert(&old_key, right_page, &mut self.pager);
                        self.pager.write_node(parent_ptr, parent.into())?;
                    } else {
                        // Need to split parent too
                        let mut right_parent = parent.split();
                        // Determine which half gets the new entry
                        let split_key = parent
                            .resolved_key(parent.len() - 1, &mut self.pager)
                            .into_owned();
                        if old_key.as_slice() <= split_key.as_slice() {
                            parent.insert(&old_key, right_page, &mut self.pager);
                        } else {
                            right_parent.insert(&old_key, right_page, &mut self.pager);
                        }

                        let rp = self.alloc_page()?;
                        self.pager.write_node(rp, right_parent.into())?;
                        self.pager.write_node(parent_ptr, parent.into())?;

                        let gparents = &parents[..parents.len() - 1];
                        self.index_propagate_split(root, gparents, parent_ptr, rp)?;
                    }

                    return Ok(());
                }
            }
            panic!("left_page not found in parent during index split propagation");
        } else {
            // Splitting the root
            let right_any = self.pager.owned_node(right_page)?;
            let right_max_key: Vec<u8> = match right_any.page_type() {
                PageType::IndexLeaf => {
                    let leaf: IndexLeafPage<N> = right_any.try_into().unwrap();
                    leaf.resolved_key(leaf.len() - 1, &mut self.pager)
                        .into_owned()
                }
                PageType::IndexInternal => {
                    let internal: IndexInternalPage<N> = right_any.try_into().unwrap();
                    internal
                        .resolved_key(internal.len() - 1, &mut self.pager)
                        .into_owned()
                }
                _ => unreachable!(),
            };

            // Move root content to a new page
            let moved_page = self.alloc_page()?;
            let root_content = self.pager.owned_node(root)?;
            self.pager.write_node(moved_page, root_content)?;

            let mut new_root = IndexInternalPage::<N>::new();
            new_root.insert(&left_max_key, moved_page, &mut self.pager);
            new_root.insert(&right_max_key, right_page, &mut self.pager);
            self.pager.write_node(root, new_root.into())?;
            Ok(())
        }
    }

    pub fn index_remove(
        &mut self,
        root: NodePtr,
        value: &[u8],
        key: Key,
    ) -> Result<bool, TreeError> {
        let mut current = root;
        let mut path = vec![];

        loop {
            path.push(current);
            let page = self.pager.owned_node(current)?;

            match page.page_type() {
                PageType::IndexLeaf => {
                    let leaf: IndexLeafPage<N> = page.try_into().unwrap();
                    if let Some(idx) = leaf.find_entry(value, key, &mut self.pager) {
                        let overflow_meta = if leaf.is_overflow(idx) {
                            Some(Self::parse_overflow_meta(leaf.key(idx)))
                        } else {
                            None
                        };
                        {
                            let p = self.pager.mut_node(current)?;
                            let leaf_mut: &mut IndexLeafPage<N> = p.try_into().unwrap();
                            leaf_mut.remove(idx);
                        }
                        self.index_merge_leaf(&path)?;
                        if let Some((start_page, total_len)) = overflow_meta {
                            self.free_overflow_pages(start_page, total_len)?;
                        }
                        return Ok(true);
                    }
                    return Ok(false);
                }
                PageType::IndexInternal => {
                    let internal: IndexInternalPage<N> = page.try_into().unwrap();
                    match internal.find_child(value, &mut self.pager) {
                        Some(next) => current = next,
                        None => return Ok(false),
                    }
                }
                _ => {
                    return Err(TreeError::UnexpectedPageType {
                        expected: "IndexLeaf or IndexInternal",
                    });
                }
            }
        }
    }

    /// Check and merge index leaf page if underfull.
    /// The page at `path.last()` must already be dirty in the pager cache.
    fn index_merge_leaf(&mut self, path: &[NodePtr]) -> Result<(), TreeError> {
        let page = *path.last().unwrap();

        let (free_space, len) = {
            let leaf_ref: &IndexLeafPage<N> = self.pager.read_node(page)?.try_into().unwrap();
            (leaf_ref.free_space(), leaf_ref.len())
        };

        if path.len() <= 1 || free_space <= N / 2 {
            // Already dirty from caller
            return Ok(());
        }

        if len == 0 {
            let parents = &path[..path.len() - 1];
            let parent_page = *parents.last().unwrap();
            // Find entry, extract overflow info, and remove from parent
            let overflow_meta = {
                let p = self.pager.mut_node(parent_page)?;
                let parent: &mut IndexInternalPage<N> = p.try_into().unwrap();
                let mut meta = None;
                for i in 0..parent.len() {
                    if parent.ptr(i) == page {
                        if parent.is_overflow(i) {
                            meta = Some(Self::parse_overflow_meta(parent.key(i)));
                        }
                        parent.remove(i);
                        break;
                    }
                }
                meta
            };
            if let Some((sp, tl)) = overflow_meta {
                self.free_overflow_pages(sp, tl)?;
            }
            self.free_page(page)?;
            return self.index_merge_internal(parents);
        }

        let parents = &path[..path.len() - 1];
        self.index_try_merge_leaf_with_left(page, parents)?;
        Ok(())
    }

    fn index_try_merge_leaf_with_left(
        &mut self,
        page: NodePtr,
        parents: &[NodePtr],
    ) -> Result<bool, TreeError> {
        let parent_page = *parents.last().unwrap();

        let (index, left_sibling_page) = {
            let parent_ref: &IndexInternalPage<N> =
                self.pager.read_node(parent_page)?.try_into().unwrap();
            let index = (0..parent_ref.len())
                .find(|&i| parent_ref.ptr(i) == page)
                .unwrap();
            if index == 0 {
                return Ok(false);
            }
            (index, parent_ref.ptr(index - 1))
        };

        // Read right leaf entries
        let right_entries = {
            let right_ref: &IndexLeafPage<N> = self.pager.read_node(page)?.try_into().unwrap();
            (0..right_ref.len())
                .map(|i| {
                    let (_, kl, v) = right_ref.read_slot(i);
                    (right_ref.key(i).to_vec(), kl, v)
                })
                .collect::<Vec<_>>()
        };

        // Check feasibility
        {
            let left_ref: &IndexLeafPage<N> =
                self.pager.read_node(left_sibling_page)?.try_into().unwrap();
            let mut free = left_ref.free_space();
            for (key_data, _, _) in &right_entries {
                let inline_size = key_data.len();
                if free < INDEX_SLOT_SIZE + inline_size {
                    return Ok(false);
                }
                free -= INDEX_SLOT_SIZE + inline_size;
            }
        }

        // Read left entries for merging
        let left_entries = {
            let left_ref: &IndexLeafPage<N> =
                self.pager.read_node(left_sibling_page)?.try_into().unwrap();
            (0..left_ref.len())
                .map(|i| {
                    let (_, kl, v) = left_ref.read_slot(i);
                    (left_ref.key(i).to_vec(), kl, v)
                })
                .collect::<Vec<_>>()
        };

        // Merge left entries into right page (left entries go at the beginning)
        {
            let p = self.pager.mut_node(page)?;
            let merged: &mut IndexLeafPage<N> = p.try_into().unwrap();
            for (i, (key_data, kl, v)) in left_entries.iter().enumerate() {
                merged.insert_raw_at(i, key_data, *kl, *v);
            }
        }

        // Extract overflow info and remove parent entry
        let overflow_meta = {
            let p = self.pager.mut_node(parent_page)?;
            let parent: &mut IndexInternalPage<N> = p.try_into().unwrap();
            let meta = if parent.is_overflow(index - 1) {
                Some(Self::parse_overflow_meta(parent.key(index - 1)))
            } else {
                None
            };
            parent.remove(index - 1);
            meta
        };
        if let Some((sp, tl)) = overflow_meta {
            self.free_overflow_pages(sp, tl)?;
        }
        self.free_page(left_sibling_page)?;
        self.index_merge_internal(parents)?;
        Ok(true)
    }

    /// Check and merge index internal pages if underfull. Iterative.
    /// The page at `path.last()` must already be dirty in the pager cache.
    fn index_merge_internal(&mut self, path: &[NodePtr]) -> Result<(), TreeError> {
        let mut cur_path = path;

        loop {
            let page = *cur_path.last().unwrap();

            let (len, free_space) = {
                let node_ref: &IndexInternalPage<N> =
                    self.pager.read_node(page)?.try_into().unwrap();
                (node_ref.len(), node_ref.free_space())
            };

            if cur_path.len() <= 1 || free_space <= N / 2 {
                // Already dirty from caller
                return Ok(());
            }

            if len == 0 {
                if cur_path.len() <= 1 {
                    let leaf = IndexLeafPage::<N>::new();
                    self.pager.write_node(page, leaf.into())?;
                    return Ok(());
                }
                let parents = &cur_path[..cur_path.len() - 1];
                let parent_page = *parents.last().unwrap();
                // Find entry, extract overflow info, and remove from parent
                let overflow_meta = {
                    let p = self.pager.mut_node(parent_page)?;
                    let parent: &mut IndexInternalPage<N> = p.try_into().unwrap();
                    let mut meta = None;
                    for i in 0..parent.len() {
                        if parent.ptr(i) == page {
                            if parent.is_overflow(i) {
                                meta = Some(Self::parse_overflow_meta(parent.key(i)));
                            }
                            parent.remove(i);
                            break;
                        }
                    }
                    meta
                };
                if let Some((sp, tl)) = overflow_meta {
                    self.free_overflow_pages(sp, tl)?;
                }
                self.free_page(page)?;
                cur_path = parents;
                continue;
            }

            let parents = &cur_path[..cur_path.len() - 1];
            if self.index_try_merge_internal_with_left(page, parents)? {
                cur_path = parents;
                continue;
            } else {
                // Already dirty from caller
                return Ok(());
            }
        }
    }

    /// Returns `true` if merge succeeded, `false` otherwise.
    fn index_try_merge_internal_with_left(
        &mut self,
        page: NodePtr,
        parents: &[NodePtr],
    ) -> Result<bool, TreeError> {
        let parent_page = *parents.last().unwrap();

        let (index, left_sibling_page) = {
            let parent_ref: &IndexInternalPage<N> =
                self.pager.read_node(parent_page)?.try_into().unwrap();
            let index = (0..parent_ref.len())
                .find(|&i| parent_ref.ptr(i) == page)
                .unwrap();
            if index == 0 {
                return Ok(false);
            }
            (index, parent_ref.ptr(index - 1))
        };

        // Read right entries
        let right_entries = {
            let right_ref: &IndexInternalPage<N> = self.pager.read_node(page)?.try_into().unwrap();
            (0..right_ref.len())
                .map(|i| {
                    let (_, kl, v) = right_ref.read_slot(i);
                    (right_ref.key(i).to_vec(), kl, v)
                })
                .collect::<Vec<_>>()
        };

        // Check if merge is feasible
        {
            let left_ref: &IndexInternalPage<N> =
                self.pager.read_node(left_sibling_page)?.try_into().unwrap();
            let mut free = left_ref.free_space();
            for (key_data, _, _) in &right_entries {
                let inline_size = key_data.len();
                if free < INDEX_SLOT_SIZE + inline_size {
                    return Ok(false);
                }
                free -= INDEX_SLOT_SIZE + inline_size;
            }
        }

        // Read left entries for merging into the right page
        let left_entries = {
            let left_ref: &IndexInternalPage<N> =
                self.pager.read_node(left_sibling_page)?.try_into().unwrap();
            (0..left_ref.len())
                .map(|i| {
                    let (_, kl, v) = left_ref.read_slot(i);
                    (left_ref.key(i).to_vec(), kl, v)
                })
                .collect::<Vec<_>>()
        };

        // Merge left entries into right page (left entries go at the beginning)
        {
            let p = self.pager.mut_node(page)?;
            let merged: &mut IndexInternalPage<N> = p.try_into().unwrap();
            for (i, (key_data, kl, v)) in left_entries.iter().enumerate() {
                merged.insert_raw_at(i, key_data, *kl, *v);
            }
        }

        // Extract overflow info and remove entry from parent
        let overflow_meta = {
            let p = self.pager.mut_node(parent_page)?;
            let parent: &mut IndexInternalPage<N> = p.try_into().unwrap();
            let meta = if parent.is_overflow(index - 1) {
                Some(Self::parse_overflow_meta(parent.key(index - 1)))
            } else {
                None
            };
            parent.remove(index - 1);
            meta
        };
        if let Some((sp, tl)) = overflow_meta {
            self.free_overflow_pages(sp, tl)?;
        }
        self.free_page(left_sibling_page)?;
        Ok(true)
    }

    pub fn index_read<T>(
        &mut self,
        root: NodePtr,
        value: &[u8],
        f: impl FnOnce(Option<Key>) -> T,
    ) -> Result<T, TreeError> {
        let mut current = root;
        let mut ff = Some(f);

        loop {
            let page = self.pager.owned_node(current)?;
            match page.page_type() {
                PageType::IndexLeaf => {
                    let leaf: IndexLeafPage<N> = page.try_into().unwrap();
                    if let Some(key) = leaf.get(value, &mut self.pager) {
                        return Ok(ff.take().unwrap()(Some(key)));
                    }
                    return Ok(ff.take().unwrap()(None));
                }
                PageType::IndexInternal => {
                    let internal: IndexInternalPage<N> = page.try_into().unwrap();
                    match internal.find_child(value, &mut self.pager) {
                        Some(next) => current = next,
                        None => return Ok(ff.take().unwrap()(None)),
                    }
                }
                _ => {
                    return Err(TreeError::UnexpectedPageType {
                        expected: "IndexLeaf or IndexInternal",
                    });
                }
            }
        }
    }

    pub fn index_read_range<R: RangeBounds<Vec<u8>>>(
        &mut self,
        root: NodePtr,
        range: R,
        mut f: impl FnMut(&[u8], Key),
    ) -> Result<(), TreeError> {
        self.index_read_range_at(root, &range, &mut f)
    }

    fn index_read_range_at<R: RangeBounds<Vec<u8>>>(
        &mut self,
        node_ptr: NodePtr,
        range: &R,
        f: &mut impl FnMut(&[u8], Key),
    ) -> Result<(), TreeError> {
        let page = self.pager.owned_node(node_ptr)?;

        match page.page_type() {
            PageType::IndexLeaf => {
                let leaf: IndexLeafPage<N> = page.try_into().unwrap();
                for i in 0..leaf.len() {
                    let key_bytes = leaf.resolved_key(i, &mut self.pager).into_owned();
                    if range.contains(&key_bytes) {
                        f(&key_bytes, leaf.value(i));
                    }
                }
            }
            PageType::IndexInternal => {
                let internal: IndexInternalPage<N> = page.try_into().unwrap();
                for i in 0..internal.len() {
                    let max_bytes = internal.resolved_key(i, &mut self.pager).into_owned();
                    let ptr = internal.ptr(i);

                    let below_start = match range.start_bound() {
                        Bound::Included(s) => max_bytes.as_slice() < s.as_slice(),
                        Bound::Excluded(s) => max_bytes.as_slice() <= s.as_slice(),
                        Bound::Unbounded => false,
                    };
                    if below_start {
                        continue;
                    }

                    if i > 0 {
                        let prev_max = internal.resolved_key(i - 1, &mut self.pager).into_owned();
                        let beyond_end = match range.end_bound() {
                            Bound::Included(e) => prev_max.as_slice() > e.as_slice(),
                            Bound::Excluded(e) => prev_max.as_slice() >= e.as_slice(),
                            Bound::Unbounded => false,
                        };
                        if beyond_end {
                            break;
                        }
                    }

                    self.index_read_range_at(ptr, range, f)?;
                }
            }
            _ => {
                return Err(TreeError::UnexpectedPageType {
                    expected: "IndexLeaf or IndexInternal",
                });
            }
        }

        Ok(())
    }

    pub fn index_available_key(&mut self, root: NodePtr) -> Result<Key, TreeError> {
        let page = self.pager.owned_node(root)?;
        match page.page_type() {
            PageType::IndexLeaf => {
                let leaf: IndexLeafPage<N> = page.try_into().unwrap();
                Ok((0..leaf.len())
                    .map(|i| leaf.value(i))
                    .max()
                    .map_or(0, |k| k.checked_add(1).unwrap_or(u64::MAX)))
            }
            PageType::IndexInternal => {
                let internal: IndexInternalPage<N> = page.try_into().unwrap();
                let mut max_key: Key = 0;
                for i in 0..internal.len() {
                    let child_key = self.index_available_key(internal.ptr(i))?;
                    max_key = max_key.max(child_key);
                }
                Ok(max_key)
            }
            _ => {
                return Err(TreeError::UnexpectedPageType {
                    expected: "IndexLeaf or IndexInternal",
                });
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────
    //  Test helpers
    // ────────────────────────────────────────────────────────────────────

    #[cfg(test)]
    pub fn debug(&mut self, root: NodePtr, min: Key, max: Key) -> Result<(), TreeError> {
        let page = self.pager.owned_node(root)?;

        match page.page_type() {
            PageType::Leaf => {
                let leaf: LeafPage<N> = page.try_into().unwrap();
                println!("Leaf Node (page {}):", root);
                for i in 0..leaf.len() {
                    let k = leaf.key(i);
                    if !(min <= k && k <= max) {
                        panic!("Key {} out of range ({}..={})", k, min, max);
                    }
                    if leaf.is_overflow(i) {
                        let (_, total_len) = Self::parse_overflow_meta(leaf.value(i));
                        println!("  Key: {}, Value Length: {} (overflow)", k, total_len);
                    } else {
                        println!(
                            "  Key: {}, Value Length: {} (inline)",
                            k,
                            leaf.value(i).len()
                        );
                    }
                }
            }
            PageType::Internal => {
                let internal: InternalPage<N> = page.try_into().unwrap();
                println!("Internal Node (page {}):", root);
                for i in 0..internal.len() {
                    let k = internal.key(i);
                    let ptr = internal.ptr(i);
                    if !(min <= k && k <= max) {
                        panic!("Key {} out of range ({}..={})", k, min, max);
                    }
                    println!("  Key: {}, Child Page: {}", k, ptr);
                }
                let mut left = min;
                for i in 0..internal.len() {
                    self.debug(internal.ptr(i), left, internal.key(i))?;
                    left = internal.key(i);
                }
            }
            _ => {}
        }

        Ok(())
    }

    #[cfg(test)]
    fn assert_no_page_leak_impl(&mut self, root: NodePtr) {
        use std::collections::BTreeSet;

        let total_pages = self.pager.total_page_count();
        let mut tree_pages = BTreeSet::new();
        tree_pages.insert(root);
        tree_pages.insert(FREE_LIST_PAGE_NUM);
        self.collect_all_pages(root, &mut tree_pages);

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

    #[cfg(test)]
    pub fn assert_no_page_leak(&mut self, root: NodePtr) {
        self.assert_no_page_leak_impl(root);
    }

    #[cfg(test)]
    pub fn assert_no_page_leak_index(&mut self, root: NodePtr) {
        self.assert_no_page_leak_impl(root);
    }

    #[cfg(test)]
    fn collect_all_pages(&mut self, root: NodePtr, pages: &mut std::collections::BTreeSet<u64>) {
        let mut stack = vec![root];
        while let Some(page_num) = stack.pop() {
            let page = self.pager.owned_node(page_num).unwrap();
            match page.page_type() {
                PageType::Leaf => {
                    let leaf: LeafPage<N> = page.try_into().unwrap();
                    for i in 0..leaf.len() {
                        if leaf.is_overflow(i) {
                            let (start_page, total_len) = Self::parse_overflow_meta(leaf.value(i));
                            self.collect_overflow_pages(start_page, total_len, pages);
                        }
                    }
                }
                PageType::Internal => {
                    let internal: InternalPage<N> = page.try_into().unwrap();
                    for i in 0..internal.len() {
                        assert!(
                            pages.insert(internal.ptr(i)),
                            "Page {} referenced multiple times in tree",
                            internal.ptr(i)
                        );
                        stack.push(internal.ptr(i));
                    }
                }
                PageType::IndexLeaf => {
                    let leaf: IndexLeafPage<N> = page.try_into().unwrap();
                    for i in 0..leaf.len() {
                        if leaf.is_overflow(i) {
                            let (start_page, total_len) = Self::parse_overflow_meta(leaf.key(i));
                            self.collect_overflow_pages(start_page, total_len, pages);
                        }
                    }
                }
                PageType::IndexInternal => {
                    let internal: IndexInternalPage<N> = page.try_into().unwrap();
                    for i in 0..internal.len() {
                        assert!(
                            pages.insert(internal.ptr(i)),
                            "Page {} referenced multiple times in index tree",
                            internal.ptr(i)
                        );
                        if internal.is_overflow(i) {
                            let (start_page, total_len) =
                                Self::parse_overflow_meta(internal.key(i));
                            self.collect_overflow_pages(start_page, total_len, pages);
                        }
                        stack.push(internal.ptr(i));
                    }
                }
            }
        }
    }

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
}

// Constants for page layout (matching page.rs internal definitions)
const INTERNAL_HEADER_SIZE: usize = 6;
const INTERNAL_ELEMENT_SIZE: usize = std::mem::size_of::<Key>() + std::mem::size_of::<NodePtr>();
const LEAF_SLOT_SIZE: usize = std::mem::size_of::<Key>() + 2 + 2;
const INDEX_SLOT_SIZE: usize = 2 + 2 + 8;

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use std::collections::HashMap;
    use std::ops::Bound;

    use super::*;

    fn new_btree<const N: usize>() -> (Btree<N>, NodePtr) {
        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file);
        let mut btree = Btree::new(pager);
        btree.init().unwrap();
        let root = btree.init_tree().unwrap();
        (btree, root)
    }

    fn build_btree(count: u64) -> (Btree<4096>, NodePtr) {
        let (mut btree, root) = new_btree::<4096>();
        for i in 0..count {
            let mut value = vec![0u8; 64];
            value[0..8].copy_from_slice(&i.to_le_bytes());
            btree.insert(root, i, value).unwrap();
        }
        (btree, root)
    }

    fn read_range_keys<const N: usize, R: RangeBounds<Key>>(
        btree: &mut Btree<N>,
        root: NodePtr,
        range: R,
    ) -> Vec<Key> {
        let mut keys = Vec::new();
        btree.read_range(root, range, |k, _| keys.push(k)).unwrap();
        keys
    }

    fn read_value<const N: usize>(
        btree: &mut Btree<N>,
        root: NodePtr,
        key: Key,
    ) -> Option<Vec<u8>> {
        btree.read(root, key, |v| v.map(|b| b.to_vec())).unwrap()
    }

    fn assert_read_eq<const N: usize>(
        btree: &mut Btree<N>,
        root: NodePtr,
        key: Key,
        expected: &[u8],
    ) {
        assert!(
            btree.read(root, key, |v| v == Some(expected)).unwrap(),
            "Key {} value mismatch",
            key
        );
    }

    fn assert_key_exists<const N: usize>(btree: &mut Btree<N>, root: NodePtr, key: Key) {
        assert!(
            btree.read(root, key, |v| v.is_some()).unwrap(),
            "Key {} should exist",
            key
        );
    }

    fn assert_key_absent<const N: usize>(btree: &mut Btree<N>, root: NodePtr, key: Key) {
        assert!(
            btree.read(root, key, |v| v.is_none()).unwrap(),
            "Key {} should be absent",
            key
        );
    }

    fn assert_keys_exist(
        btree: &mut Btree<4096>,
        root: NodePtr,
        range: impl IntoIterator<Item = u64>,
    ) {
        for k in range {
            assert_key_exists(btree, root, k);
        }
    }

    fn assert_keys_absent(
        btree: &mut Btree<4096>,
        root: NodePtr,
        range: impl IntoIterator<Item = u64>,
    ) {
        for k in range {
            assert_key_absent(btree, root, k);
        }
    }

    #[test]
    fn test_insert() {
        let (mut btree, root) = new_btree::<4096>();
        btree.insert(root, 1, b"one".to_vec()).unwrap();
    }

    #[test]
    fn test_read() {
        let (mut btree, root) = new_btree::<4096>();
        btree.insert(root, 0, b"zero".to_vec()).unwrap();
        assert_read_eq(&mut btree, root, 0, b"zero");
    }

    #[test]
    fn test_insert_and_read() {
        let (mut btree, root) = new_btree::<4096>();
        btree.insert(root, 42, b"forty-two".to_vec()).unwrap();
        assert_read_eq(&mut btree, root, 42, b"forty-two");
    }

    #[test]
    fn test_insert_multiple_and_read() {
        let (mut btree, root) = new_btree::<64>();
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
        let (mut btree, root) = new_btree::<64>();

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
        let (mut btree, root) = new_btree::<64>();

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
        let (mut btree, root) = new_btree::<256>();
        let big_value = vec![42u8; 4096];
        btree.insert(root, 1, big_value.clone()).unwrap();
        assert_eq!(read_value(&mut btree, root, 1), Some(big_value));
    }

    #[test]
    fn test_big_value_with_small_values() {
        let (mut btree, root) = new_btree::<256>();

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
        let (mut btree, root) = new_btree::<256>();

        let big_value = vec![42u8; 4096];
        btree.insert(root, 1, big_value.clone()).unwrap();

        let bigger_value = vec![99u8; 8192];
        let old = btree.insert(root, 1, bigger_value.clone()).unwrap();
        assert_eq!(old, Some(big_value));
        assert_eq!(read_value(&mut btree, root, 1), Some(bigger_value));
    }

    #[test]
    fn test_big_value_remove() {
        let (mut btree, root) = new_btree::<256>();

        let big_value = vec![42u8; 4096];
        btree.insert(root, 1, big_value.clone()).unwrap();

        assert_eq!(btree.remove(root, 1).unwrap(), Some(big_value));
        assert_key_absent(&mut btree, root, 1);
    }

    #[test]
    fn test_big_value_read_range() {
        let (mut btree, root) = new_btree::<256>();

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
        let (mut btree, root) = new_btree::<256>();

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
        let (mut btree, root) = new_btree::<256>();

        let big_value = vec![42u8; 4096];
        btree.insert(root, 1, big_value.clone()).unwrap();

        let old = btree.insert(root, 1, b"tiny".to_vec()).unwrap();
        assert_eq!(old, Some(big_value));
        assert_read_eq(&mut btree, root, 1, b"tiny");
    }

    #[test]
    fn test_free_pages_reused_after_remove() {
        let (mut btree, root) = new_btree::<4096>();
        for i in 0u64..100 {
            btree.insert(root, i, vec![0u8; 64]).unwrap();
        }
        let pages_after_insert = btree.pager.total_page_count();

        for i in 0u64..100 {
            btree.remove(root, i).unwrap();
        }
        assert_ne!(
            btree.read_free_list_head().unwrap(),
            u64::MAX,
            "Expected free pages after removal"
        );

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
        let (mut btree, root) = new_btree::<256>();
        let big_value = vec![42u8; 4096];
        btree.insert(root, 1, big_value).unwrap();
        let pages_before = btree.pager.total_page_count();

        btree.remove(root, 1).unwrap();
        assert_ne!(
            btree.read_free_list_head().unwrap(),
            u64::MAX,
            "Overflow pages should be freed on remove"
        );

        let big_value2 = vec![99u8; 4096];
        btree.insert(root, 2, big_value2).unwrap();
        assert!(
            btree.pager.total_page_count() <= pages_before + 1,
            "Overflow pages were not reused"
        );
    }

    #[test]
    fn test_free_overflow_pages_on_update() {
        let (mut btree, root) = new_btree::<256>();
        let big_value = vec![42u8; 4096];
        btree.insert(root, 1, big_value).unwrap();
        let pages_before = btree.pager.total_page_count();

        btree.insert(root, 1, b"small".to_vec()).unwrap();
        assert_ne!(
            btree.read_free_list_head().unwrap(),
            u64::MAX,
            "Old overflow pages should be freed on update"
        );

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

        let root;
        {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            let pager = Pager::<4096>::new(file);
            let mut btree = Btree::new(pager);
            btree.init().unwrap();
            root = btree.init_tree().unwrap();

            for i in 0u64..200 {
                btree.insert(root, i, vec![0u8; 64]).unwrap();
            }
            for i in 0u64..200 {
                btree.remove(root, i).unwrap();
            }
            assert_ne!(btree.read_free_list_head().unwrap(), u64::MAX);
        }

        {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            let pager = Pager::<4096>::new(file);
            let mut btree = Btree::new(pager);
            let pages_before = btree.pager.total_page_count();

            assert_ne!(
                btree.read_free_list_head().unwrap(),
                u64::MAX,
                "Free list lost after reopen"
            );

            for i in 0u64..200 {
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
        let (mut btree, root) = new_btree::<64>();
        for i in 0u64..256 {
            btree
                .insert(root, i, format!("v-{}", i).into_bytes())
                .unwrap();
        }
        btree.assert_no_page_leak(root);
    }

    #[test]
    fn test_no_leak_insert_and_remove_all() {
        let (mut btree, root) = new_btree::<64>();
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
        let (mut btree, root) = new_btree::<128>();

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
        let (mut btree, root) = new_btree::<256>();
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
        let (mut btree, root) = new_btree::<256>();
        btree.insert(root, 1, vec![42u8; 4096]).unwrap();
        btree.assert_no_page_leak(root);

        btree.insert(root, 1, b"small".to_vec()).unwrap();
        btree.assert_no_page_leak(root);

        btree.insert(root, 1, vec![99u8; 4096]).unwrap();
        btree.assert_no_page_leak(root);

        btree.insert(root, 1, vec![77u8; 8192]).unwrap();
        btree.assert_no_page_leak(root);
    }

    #[test]
    fn test_no_leak_mixed_operations() {
        let (mut btree, root) = new_btree::<128>();

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

        btree.remove_range(root, ..).unwrap();
        btree.assert_no_page_leak(root);
    }

    // ────────────────────────────────────────────────────────────────────
    //  Index tree tests
    // ────────────────────────────────────────────────────────────────────

    fn new_index_btree<const N: usize>() -> (Btree<N>, NodePtr) {
        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file);
        let mut btree = Btree::new(pager);
        btree.init().unwrap();
        let root = btree.init_index().unwrap();
        (btree, root)
    }

    fn index_read_key<const N: usize>(
        btree: &mut Btree<N>,
        root: NodePtr,
        value: &[u8],
    ) -> Option<Key> {
        btree.index_read(root, value, |k| k).unwrap()
    }

    #[test]
    fn test_index_insert_and_read() {
        let (mut btree, root) = new_index_btree::<4096>();
        assert!(btree.index_insert(root, 42, b"hello".to_vec()).unwrap());
        assert_eq!(index_read_key(&mut btree, root, b"hello"), Some(42));
    }

    #[test]
    fn test_index_insert_duplicate_returns_false() {
        let (mut btree, root) = new_index_btree::<4096>();
        assert!(btree.index_insert(root, 1, b"val".to_vec()).unwrap());
        assert!(!btree.index_insert(root, 1, b"val".to_vec()).unwrap());
    }

    #[test]
    fn test_index_same_value_different_keys() {
        let (mut btree, root) = new_index_btree::<4096>();
        assert!(btree.index_insert(root, 1, b"dup".to_vec()).unwrap());
        assert!(btree.index_insert(root, 2, b"dup".to_vec()).unwrap());
        assert!(btree.index_insert(root, 3, b"dup".to_vec()).unwrap());

        let key = index_read_key(&mut btree, root, b"dup");
        assert!(key == Some(1) || key == Some(2) || key == Some(3));
    }

    #[test]
    fn test_index_read_absent() {
        let (mut btree, root) = new_index_btree::<4096>();
        btree.index_insert(root, 1, b"exists".to_vec()).unwrap();
        assert_eq!(index_read_key(&mut btree, root, b"missing"), None);
    }

    #[test]
    fn test_index_remove() {
        let (mut btree, root) = new_index_btree::<4096>();
        btree.index_insert(root, 1, b"alpha".to_vec()).unwrap();
        btree.index_insert(root, 2, b"beta".to_vec()).unwrap();

        assert!(btree.index_remove(root, b"alpha", 1).unwrap());
        assert_eq!(index_read_key(&mut btree, root, b"alpha"), None);
        assert_eq!(index_read_key(&mut btree, root, b"beta"), Some(2));
    }

    #[test]
    fn test_index_remove_absent() {
        let (mut btree, root) = new_index_btree::<4096>();
        btree.index_insert(root, 1, b"data".to_vec()).unwrap();

        assert!(!btree.index_remove(root, b"nonexistent", 1).unwrap());
    }

    #[test]
    fn test_index_remove_wrong_key_same_value() {
        let (mut btree, root) = new_index_btree::<4096>();
        btree.index_insert(root, 1, b"shared".to_vec()).unwrap();
        btree.index_insert(root, 2, b"shared".to_vec()).unwrap();

        assert!(!btree.index_remove(root, b"shared", 99).unwrap());

        let key = index_read_key(&mut btree, root, b"shared");
        assert!(key.is_some());
    }

    #[test]
    fn test_index_multiple_values_sorted() {
        let (mut btree, root) = new_index_btree::<256>();
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

        for (v, k) in &values {
            assert_eq!(index_read_key(&mut btree, root, v), Some(*k));
        }
    }

    #[test]
    fn test_index_insert_many_and_read() {
        let (mut btree, root) = new_index_btree::<4096>();
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
        let (mut btree, root) = new_index_btree::<256>();

        let mut entries: Vec<(u64, Vec<u8>)> = (0..LEN)
            .map(|i| (i, format!("val-{:04}", i).into_bytes()))
            .collect();
        entries.shuffle(&mut rng);

        for (k, v) in &entries {
            assert!(btree.index_insert(root, *k, v.clone()).unwrap());
        }

        for (k, v) in &entries {
            assert_eq!(index_read_key(&mut btree, root, v), Some(*k));
        }

        entries.shuffle(&mut rng);
        for (k, v) in &entries {
            assert!(btree.index_remove(root, v, *k).unwrap());
            assert_eq!(index_read_key(&mut btree, root, v), None);
        }
    }

    #[test]
    fn test_index_read_range() {
        let (mut btree, root) = new_index_btree::<4096>();

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
        let (mut btree, root) = new_index_btree::<4096>();
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
        let (mut btree, root) = new_index_btree::<4096>();
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
        let (mut btree, root) = new_index_btree::<4096>();
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
        let (mut btree, root) = new_index_btree::<256>();
        for i in 0u64..100 {
            btree
                .index_insert(root, i, format!("v-{}", i).into_bytes())
                .unwrap();
        }
        let pages_before = btree.pager.total_page_count();
        btree.free_index_tree(root).unwrap();

        assert_ne!(btree.read_free_list_head().unwrap(), u64::MAX);

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
        let (mut btree, root) = new_index_btree::<256>();
        let big_value = vec![42u8; 4096];
        assert!(btree.index_insert(root, 1, big_value.clone()).unwrap());

        let found = btree.index_read(root, &big_value, |k| k).unwrap();
        assert_eq!(found, Some(1));
    }

    #[test]
    fn test_index_big_value_remove() {
        let (mut btree, root) = new_index_btree::<256>();
        let big_value = vec![42u8; 4096];
        btree.index_insert(root, 1, big_value.clone()).unwrap();

        assert!(btree.index_remove(root, &big_value, 1).unwrap());
    }

    #[test]
    fn test_index_no_leak_insert_only() {
        let (mut btree, root) = new_index_btree::<256>();
        for i in 0u64..256 {
            btree
                .index_insert(root, i, format!("v-{:04}", i).into_bytes())
                .unwrap();
        }
        btree.assert_no_page_leak_index(root);
    }

    #[test]
    fn test_index_no_leak_insert_and_remove_all() {
        let (mut btree, root) = new_index_btree::<256>();
        let entries: Vec<(u64, Vec<u8>)> = (0u64..200)
            .map(|i| (i, format!("v-{:04}", i).into_bytes()))
            .collect();

        for (k, v) in &entries {
            btree.index_insert(root, *k, v.clone()).unwrap();
        }
        for (k, v) in &entries {
            btree.index_remove(root, v, *k).unwrap();
        }
        btree.assert_no_page_leak_index(root);
    }

    #[test]
    fn test_index_no_leak_shuffle() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let (mut btree, root) = new_index_btree::<256>();

        let mut entries: Vec<(u64, Vec<u8>)> = (0u64..300)
            .map(|i| (i, format!("val-{:04}", i).into_bytes()))
            .collect();
        entries.shuffle(&mut rng);
        for (k, v) in &entries {
            btree.index_insert(root, *k, v.clone()).unwrap();
        }

        entries.shuffle(&mut rng);
        for (k, v) in &entries {
            btree.index_remove(root, v, *k).unwrap();
        }
        btree.assert_no_page_leak_index(root);
    }

    #[test]
    fn test_index_no_leak_big_values() {
        let (mut btree, root) = new_index_btree::<4096>();
        let entries: Vec<(u64, Vec<u8>)> = (0u64..10)
            .map(|i| (i, vec![i as u8; 8192 + i as usize * 100]))
            .collect();

        for (k, v) in &entries {
            btree.index_insert(root, *k, v.clone()).unwrap();
        }
        btree.assert_no_page_leak_index(root);

        for (k, v) in &entries {
            btree.index_remove(root, v, *k).unwrap();
        }
        btree.assert_no_page_leak_index(root);
    }

    #[test]
    fn test_index_sorted_order() {
        let (mut btree, root) = new_index_btree::<256>();

        let values: Vec<Vec<u8>> = (0u64..50)
            .rev()
            .map(|i| format!("{:04}", i).into_bytes())
            .collect();
        for (i, v) in values.iter().enumerate() {
            btree.index_insert(root, i as u64, v.clone()).unwrap();
        }

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
        let (mut btree, root) = new_index_btree::<256>();

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

        assert_eq!(results.len(), 100);
        for v in &results {
            assert!(v.as_slice() >= b"0050".as_slice() && v.as_slice() < b"0150".as_slice());
        }
    }
}
