use core::panic;
use std::{
    io::{Read, Seek, Write},
    ops::Range,
};

use rkyv::{Archive, Deserialize, Serialize, deserialize, rancor::Error, util::AlignedVec};

pub const PAGE_SIZE: usize = 4096;
pub type Key = u64;
pub type NodePtr = u64;
pub const ROOT_PAGE_NUM: u64 = 0;
pub const PAGE_CONTENT_SIZE: usize = PAGE_SIZE - 2;

#[derive(Archive, Deserialize, Serialize, Debug)]
pub struct Leaf {
    pub kv: Vec<(Key, Vec<u8>)>,
}

#[derive(Archive, Deserialize, Serialize, Debug)]
pub struct Internal {
    pub kv: Vec<(Key, NodePtr)>,
}

#[derive(Archive, Deserialize, Serialize, Debug)]
pub enum Node {
    Leaf(Leaf),
    Internal(Internal),
}

pub struct Pager {
    pub file: std::fs::File,
    pub next_page_num: u64,
}

fn from_page(buffer: &AlignedVec<16>) -> &[u8] {
    let len =
        u16::from_le_bytes([buffer[PAGE_CONTENT_SIZE], buffer[PAGE_CONTENT_SIZE + 1]]) as usize;
    &buffer[..len]
}

fn to_page(buffer: &mut AlignedVec<16>) {
    assert!(buffer.len() <= PAGE_CONTENT_SIZE);
    let len = buffer.len() as u16;
    buffer.resize(PAGE_SIZE, 0);
    buffer[PAGE_CONTENT_SIZE..PAGE_SIZE].copy_from_slice(&len.to_le_bytes());
}

impl Pager {
    pub fn new(file: std::fs::File) -> Self {
        let file_size = file.metadata().unwrap().len();
        let next_page_num = file_size / PAGE_SIZE as u64;
        Pager {
            file,
            next_page_num,
        }
    }

    pub fn next_page_num(&mut self) -> u64 {
        let page_num = self.next_page_num;
        self.next_page_num += 1;
        page_num
    }

    pub fn read_node<T>(
        &mut self,
        page_num: u64,
        f: impl FnOnce(&rkyv::Archived<Node>) -> T,
    ) -> Result<T, Error> {
        let mut buffer = AlignedVec::<16>::with_capacity(PAGE_SIZE);
        buffer.resize(PAGE_SIZE, 0);
        self.file
            .seek(std::io::SeekFrom::Start(page_num * PAGE_SIZE as u64))
            .unwrap();
        self.file.read_exact(&mut buffer).unwrap();
        let buffer = from_page(&buffer);
        let archived = rkyv::api::high::access::<rkyv::Archived<Node>, Error>(&buffer)?;
        Ok(f(archived))
    }

    pub fn owned_node(&mut self, page_num: u64) -> Result<Node, Error> {
        let mut buffer = AlignedVec::<16>::with_capacity(PAGE_SIZE);
        buffer.resize(PAGE_SIZE, 0);
        self.file
            .seek(std::io::SeekFrom::Start(page_num * PAGE_SIZE as u64))
            .unwrap();
        self.file.read_exact(&mut buffer).unwrap();
        let buffer = from_page(&buffer);
        let archived = rkyv::access::<ArchivedNode, Error>(&buffer)?;
        let node: Node = deserialize(archived)?;
        Ok(node)
    }

    pub fn write_buffer(&mut self, page_num: u64, mut buffer: AlignedVec<16>) -> Result<(), Error> {
        to_page(&mut buffer);
        self.file
            .seek(std::io::SeekFrom::Start(page_num * PAGE_SIZE as u64))
            .unwrap();
        self.file.write_all(&buffer).unwrap();
        Ok(())
    }

    pub fn write_node(&mut self, page_num: u64, node: &Node) -> Result<(), Error> {
        let buffer = rkyv::to_bytes(node)?;
        self.write_buffer(page_num, buffer)
    }
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
        self.pager.file.set_len(PAGE_SIZE as u64).unwrap();

        let root_leaf = Leaf { kv: vec![] };
        self.pager
            .write_node(ROOT_PAGE_NUM, &Node::Leaf(root_leaf))?;
        Ok(())
    }

    fn path_to_alloc(&mut self, key: Key, dest: Option<NodePtr>) -> Result<Vec<NodePtr>, Error> {
        let mut current = ROOT_PAGE_NUM;
        let mut path = vec![];

        enum NextNode {
            Leaf,
            Next(NodePtr),
            NeedAlloc,
        }

        loop {
            path.push(current);

            if Some(current) == dest {
                return Ok(path);
            }
            let next = self
                .pager
                .read_node(current, |archived_node| match archived_node {
                    ArchivedNode::Leaf(_) => NextNode::Leaf,
                    ArchivedNode::Internal(internal) => {
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
                    return Ok(path);
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
                        let new_leaf = Leaf { kv: vec![] };
                        self.pager
                            .write_node(new_leaf_page, &Node::Leaf(new_leaf))?;

                        internal.kv.push((key, new_leaf_page));
                        self.pager.write_node(current, &Node::Internal(internal))?;

                        return Ok(path);
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
                    ArchivedNode::Leaf(leaf) => {
                        match leaf.kv.binary_search_by_key(&key, |t| t.0.to_native()) {
                            Ok(_) => NextNode::Leaf,
                            Err(_) => NextNode::NotFound,
                        }
                    }
                    ArchivedNode::Internal(internal) => {
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

    fn search_leaf(&mut self, key: Key) -> Result<Option<Vec<NodePtr>>, Error> {
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
                    ArchivedNode::Leaf(leaf) => {
                        match leaf.kv.binary_search_by_key(&key, |t| t.0.to_native()) {
                            Ok(_) => NextNode::Leaf,
                            Err(_) => NextNode::Leaf,
                        }
                    }
                    ArchivedNode::Internal(internal) => {
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

    pub fn insert(&mut self, key: Key, value: Vec<u8>) -> Result<Option<Vec<u8>>, Error> {
        let path = self.path_to_alloc(key, None)?;

        let leaf_page = *path.last().unwrap();
        let leaf_node = self.pager.owned_node(leaf_page)?;
        let ret = if let Node::Leaf(mut leaf) = leaf_node {
            let ret = match leaf.kv.binary_search_by_key(&key, |t| t.0) {
                Ok(index) => {
                    let old_value = std::mem::replace(&mut leaf.kv[index].1, value);
                    Some(old_value)
                }
                Err(index) => {
                    leaf.kv.insert(index, (key, value));
                    None
                }
            };

            self.split_insert(&path, &Node::Leaf(leaf))?;
            ret
        } else {
            panic!("Expected leaf node");
        };

        Ok(ret)
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
                let mid = leaf.kv.len() / 2;
                let left_leaf = Leaf {
                    kv: leaf.kv[..mid].to_vec(),
                };
                let right_leaf = Leaf {
                    kv: leaf.kv[mid..].to_vec(),
                };

                let left_key = left_leaf.kv.last().unwrap().0;
                let right_key = right_leaf.kv.last().unwrap().0;

                if let Some(&parent_page) = parents.last() {
                    let parent_node = self.pager.owned_node(parent_page)?;
                    if let Node::Internal(mut internal) = parent_node {
                        let new_left_page = self.pager.next_page_num();

                        match internal.kv.binary_search_by_key(&left_key, |t| t.0) {
                            Ok(_) => {
                                panic!("Duplicate key in internal node");
                            }
                            Err(index) => {
                                internal.kv.insert(index, (left_key, new_left_page));
                            }
                        }

                        self.split_insert(parents, &Node::Internal(internal))?;

                        // Parent may be split
                        let left_path = self.path_to_alloc(left_key, Some(new_left_page))?;
                        let right_path = self.path_to_alloc(right_key, Some(page))?;

                        self.split_insert(&left_path, &Node::Leaf(left_leaf))?;
                        self.split_insert(&right_path, &Node::Leaf(right_leaf))?;
                    } else {
                        panic!("Parent is not an internal node");
                    }
                } else {
                    let new_left_page = self.pager.next_page_num();
                    let new_right_page = self.pager.next_page_num();
                    let new_root_internal = Internal {
                        kv: vec![(left_key, new_left_page), (right_key, new_right_page)],
                    };

                    self.pager
                        .write_node(ROOT_PAGE_NUM, &Node::Internal(new_root_internal))?;

                    self.split_insert(&[ROOT_PAGE_NUM, new_left_page], &Node::Leaf(left_leaf))?;
                    self.split_insert(&[ROOT_PAGE_NUM, new_right_page], &Node::Leaf(right_leaf))?;
                }
            }
            Node::Internal(internal) => {
                let mid = internal.kv.len() / 2;
                let left_internal = Internal {
                    kv: internal.kv[..mid].to_vec(),
                };
                let right_internal = Internal {
                    kv: internal.kv[mid..].to_vec(),
                };

                let left_key = left_internal.kv.last().unwrap().0;
                let right_key = right_internal.kv.last().unwrap().0;

                if let Some(&parent_page) = parents.last() {
                    let parent_node = self.pager.owned_node(parent_page)?;
                    if let Node::Internal(mut internal) = parent_node {
                        let new_left_page = self.pager.next_page_num();
                        match internal.kv.binary_search_by_key(&left_key, |t| t.0) {
                            Ok(_) => {
                                panic!("Duplicate key in internal node");
                            }
                            Err(index) => {
                                internal.kv.insert(index, (left_key, new_left_page));
                            }
                        }

                        // Never fail
                        self.pager
                            .write_node(new_left_page, &Node::Internal(left_internal))?;
                        self.pager
                            .write_node(page, &Node::Internal(right_internal))?;

                        self.split_insert(parents, &Node::Internal(internal))?;
                    } else {
                        panic!("Parent is not an internal node");
                    }
                } else {
                    let new_left_page = self.pager.next_page_num();
                    let new_right_page = self.pager.next_page_num();
                    let new_root_internal = Internal {
                        kv: vec![(left_key, new_left_page), (right_key, new_right_page)],
                    };

                    self.pager
                        .write_node(ROOT_PAGE_NUM, &Node::Internal(new_root_internal))?;
                    self.pager
                        .write_node(new_left_page, &Node::Internal(left_internal))?;
                    self.pager
                        .write_node(new_right_page, &Node::Internal(right_internal))?;
                }
            }
        }
        Ok(())
    }

    pub fn remove(&mut self, key: Key) -> Result<Option<Vec<u8>>, Error> {
        if let Some(path) = self.search(key)? {
            let leaf_page = *path.last().unwrap();
            let leaf_node = self.pager.owned_node(leaf_page)?;
            if let Node::Leaf(mut leaf) = leaf_node {
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
            } else {
                panic!("Expected leaf node");
            }
        }
        Ok(None)
    }

    pub fn remove_range(&mut self, mut range: Range<Key>) -> Result<(), Error> {
        while !range.is_empty() {
            let path = match self.search_leaf(range.start)? {
                Some(p) => p,
                None => break,
            };
            let leaf_page = *path.last().unwrap();
            let leaf_node = self.pager.owned_node(leaf_page)?;
            if let Node::Leaf(mut leaf) = leaf_node {
                let Some(last_key) = leaf.kv.last().map(|(k, _)| *k) else {
                    break;
                };
                leaf.kv.retain(|(k, _)| !range.contains(k));
                self.merge_insert(&path, &Node::Leaf(leaf))?;
                if last_key == Key::MAX {
                    break;
                }
                range.start = last_key + 1;
            } else {
                panic!("Expected leaf node");
            }
        }

        Ok(())
    }

    fn merge_insert(&mut self, path: &[NodePtr], insert: &Node) -> Result<(), Error> {
        let buffer = rkyv::to_bytes(insert)?;

        let page = *path.last().unwrap();

        assert!(buffer.len() <= PAGE_CONTENT_SIZE);

        if buffer.len() < PAGE_CONTENT_SIZE / 2 && path.len() > 1 {
            match insert {
                Node::Leaf(leaf) => {
                    if leaf.kv.is_empty() {
                        let parents = &path[..path.len() - 1];
                        let parent_page = *parents.last().unwrap();
                        let parent_node = self.pager.owned_node(parent_page)?;
                        if let Node::Internal(mut internal) = parent_node {
                            internal.kv.retain(|&(_, ptr)| ptr != page);
                            self.merge_insert(parents, &Node::Internal(internal))?;
                        } else {
                            panic!("Parent is not an internal node");
                        }
                    } else {
                        // merge to left sibling
                        let parents = &path[..path.len() - 1];
                        let parent_page = *parents.last().unwrap();
                        let parent_node = self.pager.owned_node(parent_page)?;
                        if let Node::Internal(mut internal) = parent_node {
                            let index = internal
                                .kv
                                .iter()
                                .position(|&(_, ptr)| ptr == page)
                                .unwrap();
                            if index > 0 {
                                let left_sibling_page = internal.kv[index - 1].1;
                                let left_sibling_node = self.pager.owned_node(left_sibling_page)?;
                                if let Node::Leaf(mut left_sibling) = left_sibling_node {
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
                                    panic!("Left sibling is not a leaf node");
                                }
                            } else {
                                // No left sibling, just write back
                                self.pager.write_node(page, insert)?;
                            }
                        } else {
                            panic!("Parent is not an internal node");
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
                    ArchivedNode::Leaf(leaf) => {
                        match leaf.kv.binary_search_by_key(&key, |t| t.0.to_native()) {
                            Ok(index) => Some(f.take().unwrap()(Some(leaf.kv[index].1.as_ref()))),
                            Err(_) => Some(f.take().unwrap()(None)),
                        }
                    }
                    ArchivedNode::Internal(internal) => {
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

    use super::*;
    #[test]
    fn test_search() {
        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file);
        let mut btree = Btree::new(pager);
        btree.init().unwrap();
        let leaf = Leaf {
            kv: vec![(0, b"zero".to_vec()), (10, b"ten".to_vec())],
        };
        btree
            .pager
            .write_node(ROOT_PAGE_NUM, &Node::Leaf(leaf))
            .unwrap();

        btree.path_to_alloc(0, None).unwrap();
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
}
