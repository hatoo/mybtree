use core::panic;
use rkyv::rancor::Error;
use std::collections::BTreeMap;
use std::ops::RangeBounds;

use crate::pager::Pager;
use crate::types::{
    Internal, Key, Leaf, Node, NodePtr, PAGE_CONTENT_SIZE, PAGE_SIZE, ROOT_PAGE_NUM,
};
use crate::util::{is_overlap, split_internal, split_leaf};

pub struct Btree {
    pub pager: Pager,
}

impl Btree {
    pub fn new(pager: Pager) -> Self {
        Btree { pager }
    }

    pub fn init(&mut self) -> Result<(), Error> {
        self.pager.next_page_num = 1;
        self.pager.file.set_len(PAGE_SIZE as u64).unwrap();

        let root_leaf = Leaf { kv: vec![] };
        self.pager
            .write_node(ROOT_PAGE_NUM, &Node::Leaf(root_leaf))?;
        Ok(())
    }

    fn search(&mut self, key: Key) -> Result<Option<Vec<NodePtr>>, Error> {
        let mut current = ROOT_PAGE_NUM;
        let mut path = vec![];

        enum NextNode {
            Leaf,
            Next(NodePtr),
            NotFound,
        }

        loop {
            path.push(current);

            let next = self
                .pager
                .read_node(current, |archived_node| match archived_node {
                    rkyv::Archived::<Node>::Leaf(leaf) => {
                        match leaf.kv.binary_search_by_key(&key, |t| t.0.to_native()) {
                            Ok(_) => NextNode::Leaf,
                            Err(_) => NextNode::NotFound,
                        }
                    }
                    rkyv::Archived::<Node>::Internal(internal) => {
                        match internal.kv.binary_search_by_key(&key, |t| t.0.to_native()) {
                            Ok(index) | Err(index) => {
                                if let Some(next_page) =
                                    internal.kv.get(index).map(|t| t.1.to_native())
                                {
                                    NextNode::Next(next_page)
                                } else {
                                    NextNode::NotFound
                                }
                            }
                        }
                    }
                })?;

            match next {
                NextNode::Leaf => {
                    return Ok(Some(path));
                }
                NextNode::Next(next_page) => {
                    current = next_page;
                }
                NextNode::NotFound => {
                    return Ok(None);
                }
            }
        }
    }

    pub fn insert(&mut self, key: Key, mut value: Vec<u8>) -> Result<Option<Vec<u8>>, Error> {
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
                    rkyv::Archived::<Node>::Leaf(_) => NextNode::Leaf,
                    rkyv::Archived::<Node>::Internal(internal) => {
                        match internal.kv.binary_search_by_key(&key, |t| t.0.to_native()) {
                            Ok(index) | Err(index) => {
                                if let Some(next_page) =
                                    internal.kv.get(index).map(|t| t.1.to_native())
                                {
                                    NextNode::Next(next_page)
                                } else {
                                    NextNode::NeedAlloc
                                }
                            }
                        }
                    }
                })?;

            match next {
                NextNode::Leaf => {
                    let Node::Leaf(mut leaf) = self.pager.owned_node(current)? else {
                        panic!("Expected leaf node");
                    };
                    match leaf.kv.binary_search_by_key(&key, |t| t.0) {
                        Ok(index) => {
                            std::mem::swap(&mut leaf.kv[index].1, &mut value);
                            let old_value = value;
                            if leaf.kv[index].1.len() > old_value.len() {
                                self.split_insert(&path, &Node::Leaf(leaf))?;
                            } else {
                                self.merge_insert(&path, &Node::Leaf(leaf))?;
                            }
                            return Ok(Some(old_value));
                        }
                        Err(index) => {
                            leaf.kv.insert(index, (key, value));
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
                        let new_leaf = Leaf {
                            kv: vec![(key, value)],
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

        if buffer.len() <= PAGE_CONTENT_SIZE {
            self.pager.write_buffer(page, buffer)?;
            return Ok(());
        }

        match insert {
            Node::Leaf(leaf) => {
                let mut splits = split_leaf(leaf.kv.clone())?;

                if let Some(&parent_page) = parents.last() {
                    let parent_node = self.pager.owned_node(parent_page)?;
                    let Node::Internal(mut internal) = parent_node else {
                        panic!("Parent is not an internal node");
                    };
                    let right = splits.pop().unwrap();
                    self.pager
                        .write_node(page, &Node::Leaf(Leaf { kv: right }))?;
                    let mut left_pages = Vec::new();
                    for split in splits {
                        let new_page = self.pager.next_page_num();
                        let left_key = split.last().unwrap().0;
                        self.pager
                            .write_node(new_page, &Node::Leaf(Leaf { kv: split }))?;
                        left_pages.push((left_key, new_page));
                    }
                    let mut kv = internal.kv.iter().cloned().collect::<BTreeMap<_, _>>();
                    kv.extend(left_pages.into_iter());
                    let kv = kv.into_iter().collect::<Vec<_>>();
                    internal.kv = kv;
                    self.split_insert(parents, &Node::Internal(internal))?;
                } else {
                    let mut new_pages = Vec::new();
                    for split in splits {
                        let new_page = self.pager.next_page_num();
                        let key = split.last().unwrap().0;
                        self.pager
                            .write_node(new_page, &Node::Leaf(Leaf { kv: split }))?;
                        new_pages.push((key, new_page));
                    }

                    self.split_insert(
                        &[ROOT_PAGE_NUM],
                        &Node::Internal(Internal { kv: new_pages }),
                    )?;
                }
            }
            Node::Internal(internal) => {
                let mut splits = split_internal(internal.kv.clone())?;

                if let Some(&parent_page) = parents.last() {
                    let parent_node = self.pager.owned_node(parent_page)?;
                    let Node::Internal(mut internal) = parent_node else {
                        panic!("Parent is not an internal node");
                    };
                    let right = splits.pop().unwrap();
                    self.pager
                        .write_node(page, &Node::Internal(Internal { kv: right }))?;
                    let mut left_pages = Vec::new();

                    for split in splits {
                        let new_left_page = self.pager.next_page_num();
                        let left_key = split.last().unwrap().0;
                        self.pager
                            .write_node(new_left_page, &Node::Internal(Internal { kv: split }))?;
                        left_pages.push((left_key, new_left_page));
                    }
                    let mut kv = internal.kv.iter().cloned().collect::<BTreeMap<_, _>>();
                    kv.extend(left_pages.into_iter());
                    let kv = kv.into_iter().collect::<Vec<_>>();
                    internal.kv = kv;
                    self.split_insert(parents, &Node::Internal(internal))?;
                } else {
                    let new_pages = splits
                        .into_iter()
                        .map(|split| {
                            let new_page = self.pager.next_page_num();
                            let key = split.last().unwrap().0;
                            self.pager
                                .write_node(new_page, &Node::Internal(Internal { kv: split }))?;
                            Ok((key, new_page))
                        })
                        .collect::<Result<Vec<_>, Error>>()?;

                    let new_root_internal = Internal { kv: new_pages };

                    self.pager
                        .write_node(ROOT_PAGE_NUM, &Node::Internal(new_root_internal))?;
                }
            }
        }
        Ok(())
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
                    let old_value = leaf.kv.remove(index).1;
                    self.merge_insert(&path, &Node::Leaf(leaf))?;
                    return Ok(Some(old_value));
                }
                Err(_) => {
                    panic!("Key not found in leaf node");
                }
            }
        }
        Ok(None)
    }

    fn merge_insert(&mut self, path: &[NodePtr], insert: &Node) -> Result<(), Error> {
        let buffer = rkyv::to_bytes(insert)?;

        let page = *path.last().unwrap();

        debug_assert!(buffer.len() <= PAGE_CONTENT_SIZE);

        if buffer.len() < PAGE_CONTENT_SIZE / 2 && path.len() > 1 {
            match insert {
                Node::Leaf(leaf) => {
                    if leaf.kv.is_empty() {
                        let parents = &path[..path.len() - 1];
                        let parent_page = *parents.last().unwrap();
                        let parent_node = self.pager.owned_node(parent_page)?;
                        let Node::Internal(mut internal) = parent_node else {
                            panic!("Parent is not an internal node");
                        };
                        internal.kv.retain(|&(_, ptr)| ptr != page);
                        self.merge_insert(parents, &Node::Internal(internal))?;
                    } else {
                        // merge to left sibling
                        let parents = &path[..path.len() - 1];
                        let parent_page = *parents.last().unwrap();
                        let parent_node = self.pager.owned_node(parent_page)?;
                        let Node::Internal(mut internal) = parent_node else {
                            panic!("Parent is not an internal node");
                        };
                        let index = internal
                            .kv
                            .iter()
                            .position(|&(_, ptr)| ptr == page)
                            .unwrap();
                        if index > 0 {
                            let left_sibling_page = internal.kv[index - 1].1;
                            let left_sibling_node = self.pager.owned_node(left_sibling_page)?;
                            let Node::Leaf(mut left_sibling) = left_sibling_node else {
                                panic!("Left sibling is not a leaf node");
                            };
                            left_sibling.kv.extend(leaf.kv.iter().cloned());
                            let buffer = rkyv::to_bytes(&Node::Leaf(left_sibling))?;
                            if buffer.len() <= PAGE_CONTENT_SIZE {
                                self.pager.write_buffer(page, buffer)?;
                                internal.kv.remove(index - 1);
                                self.merge_insert(parents, &Node::Internal(internal))?;
                                // TODO: Add page to free list
                            } else {
                                self.pager.write_node(page, insert)?;
                            }
                        } else {
                            // No left sibling, just write back
                            self.pager.write_node(page, insert)?;
                        }
                    }
                }
                Node::Internal(internal) => {
                    let parents = &path[..path.len() - 1];
                    let parent_page = *parents.last().unwrap();
                    let parent_node = self.pager.owned_node(parent_page)?;

                    if let Node::Internal(mut parent_internal) = parent_node {
                        let index = parent_internal
                            .kv
                            .iter()
                            .position(|&(_, ptr)| ptr == page)
                            .unwrap();
                        if index > 0 {
                            let left_sibling_page = parent_internal.kv[index - 1].1;
                            let left_sibling_node = self.pager.owned_node(left_sibling_page)?;
                            if let Node::Internal(mut left_sibling) = left_sibling_node {
                                left_sibling.kv.extend(internal.kv.iter().cloned());
                                let buffer = rkyv::to_bytes(&Node::Internal(left_sibling))?;
                                if buffer.len() <= PAGE_CONTENT_SIZE {
                                    self.pager.write_buffer(left_sibling_page, buffer)?;
                                    parent_internal.kv.remove(index);
                                    self.pager.write_node(page, insert)?; // TODO: Add page to free list
                                    self.merge_insert(parents, &Node::Internal(parent_internal))?;
                                } else {
                                    self.pager.write_node(page, insert)?;
                                }
                            } else {
                                self.pager.write_node(page, insert)?;
                            }
                        } else {
                            self.pager.write_node(page, insert)?;
                        }
                    }
                }
            }
            Ok(())
        } else {
            self.pager.write_buffer(page, buffer)?;
            Ok(())
        }
    }

    pub fn read<T>(&mut self, key: Key, f: impl FnOnce(Option<&[u8]>) -> T) -> Result<T, Error> {
        let mut current = ROOT_PAGE_NUM;

        let mut ff = Some(f);
        let f = &mut ff;

        loop {
            let result = self
                .pager
                .read_node(current, |archived_node| match archived_node {
                    rkyv::Archived::<Node>::Leaf(leaf) => {
                        match leaf.kv.binary_search_by_key(&key, |t| t.0.to_native()) {
                            Ok(index) => Some(f.take().unwrap()(Some(leaf.kv[index].1.as_ref()))),
                            Err(_) => Some(f.take().unwrap()(None)),
                        }
                    }
                    rkyv::Archived::<Node>::Internal(internal) => {
                        match internal.kv.binary_search_by_key(&key, |t| t.0.to_native()) {
                            Ok(index) | Err(index) => {
                                if let Some(next_page) =
                                    internal.kv.get(index).map(|t| t.1.to_native())
                                {
                                    current = next_page;
                                    None
                                } else {
                                    Some(f.take().unwrap()(None))
                                }
                            }
                        }
                    }
                })?;

            if let Some(value) = result {
                return Ok(value);
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
                        f(k, &v);
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
                    println!("  Key: {}, Value Length: {}", k, v.len());
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

    fn build_btree(count: u64) -> Btree {
        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file);
        let mut btree = Btree::new(pager);
        btree.init().unwrap();

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

    #[test]
    fn test_insert() {
        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file);
        let mut btree = Btree::new(pager);
        btree.init().unwrap();

        btree.insert(1, b"one".to_vec()).unwrap();
    }

    #[test]
    fn test_read() {
        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file);
        let mut btree = Btree::new(pager);
        let root_leaf = Leaf {
            kv: vec![(0, b"zero".to_vec())],
        };
        btree
            .pager
            .write_node(ROOT_PAGE_NUM, &Node::Leaf(root_leaf))
            .unwrap();
        assert!(btree.read(0, |v| v == Some(b"zero".as_ref())).unwrap());
    }

    #[test]
    fn test_insert_and_read() {
        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file);
        let mut btree = Btree::new(pager);
        btree.init().unwrap();

        btree.insert(42, b"forty-two".to_vec()).unwrap();
        assert!(
            btree
                .read(42, |v| v == Some(b"forty-two".as_ref()))
                .unwrap()
        );
    }

    #[test]
    fn test_insert_multiple_and_read() {
        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file);
        let mut btree = Btree::new(pager);
        btree.init().unwrap();

        let mut map = HashMap::new();

        for i in 0u64..100 {
            let mut value = [0; 512];
            value[0..8].copy_from_slice(&i.to_le_bytes());
            btree.insert(i, value.to_vec()).unwrap();
            map.insert(i, value.to_vec());

            match btree.pager.owned_node(ROOT_PAGE_NUM).unwrap() {
                Node::Internal(internal) => {
                    dbg!(internal);
                }
                Node::Leaf(leaf) => {
                    dbg!(leaf.kv.iter().map(|(k, _)| *k).collect::<Vec<_>>());
                }
            }

            for j in 0u64..=i {
                let expected = map.get(&j).unwrap();
                assert!(
                    btree.read(j, |v| v == Some(expected.as_ref())).unwrap(),
                    "Failed at {} {}, expected {:?}, got {:?}",
                    i,
                    j,
                    &expected[0..8],
                    btree.read(j, |v| v.map(|v| v[0..8].to_vec())).unwrap()
                );
            }
        }
    }

    #[test]
    fn test_remove() {
        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file);
        let mut btree = Btree::new(pager);
        btree.init().unwrap();

        btree.insert(1, b"one".to_vec()).unwrap();
        btree.insert(2, b"two".to_vec()).unwrap();
        btree.insert(3, b"three".to_vec()).unwrap();

        assert_eq!(btree.remove(2).unwrap(), Some(b"two".to_vec()));
        assert!(btree.read(2, |v| v.is_none()).unwrap());
        assert!(btree.read(1, |v| v == Some(b"one".as_ref())).unwrap());
        assert!(btree.read(3, |v| v == Some(b"three".as_ref())).unwrap());

        assert_eq!(btree.remove(999).unwrap(), None);
    }

    #[test]
    fn test_remove_seq() {
        const LEN: u64 = 1000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file);
        let mut btree = Btree::new(pager);
        btree.init().unwrap();

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
            assert!(btree.read(i, |v| v.is_none()).unwrap());
        }
    }

    #[test]
    fn test_read_range_full() {
        let mut btree = build_btree(200);
        let keys = read_range_keys(&mut btree, 0..=199);
        let expected = (0..200).collect::<Vec<_>>();
        assert_eq!(keys, expected);
    }

    #[test]
    fn test_read_range_exclusive_start() {
        let mut btree = build_btree(200);
        let range = (Bound::Excluded(10), Bound::Included(20));
        let keys = read_range_keys(&mut btree, range);
        let expected = (11..=20).collect::<Vec<_>>();
        assert_eq!(keys, expected);
    }

    #[test]
    fn test_read_range_unbounded_end() {
        let mut btree = build_btree(200);
        let keys = read_range_keys(&mut btree, ..=5);
        let expected = (0..=5).collect::<Vec<_>>();
        assert_eq!(keys, expected);
    }

    #[test]
    fn test_read_range_empty() {
        let mut btree = build_btree(200);
        let keys = read_range_keys(&mut btree, 500..=600);
        assert!(keys.is_empty());
    }

    #[test]
    fn test_read_range_order() {
        let mut btree = build_btree(500);
        let keys = read_range_keys(&mut btree, 123..=321);
        let expected = (123..=321).collect::<Vec<_>>();
        assert_eq!(keys, expected);
    }
}
