use std::io::{Read, Seek, Write};

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

    pub fn search_leaf(&mut self, key: Key) -> Result<NodePtr, Error> {
        let mut current = ROOT_PAGE_NUM;

        loop {
            if let Some(page) =
                self.pager
                    .read_node(current, |archived_node| match archived_node {
                        ArchivedNode::Leaf(_) => Some(current),
                        ArchivedNode::Internal(internal) => {
                            match internal.kv.binary_search_by_key(&key, |t| t.0.to_native()) {
                                Ok(index) => {
                                    current = internal.kv[index].1.to_native();
                                }
                                Err(index) => {
                                    current = internal.kv[index].1.to_native();
                                }
                            }
                            None
                        }
                    })?
            {
                return Ok(page);
            }
        }
    }

    pub fn path_to_leaf(&mut self, key: Key) -> Result<Vec<NodePtr>, Error> {
        let mut path = Vec::new();
        let mut current = ROOT_PAGE_NUM;

        loop {
            path.push(current);
            if self
                .pager
                .read_node(current, |archived_node| match archived_node {
                    ArchivedNode::Leaf(_) => true,
                    ArchivedNode::Internal(internal) => {
                        match internal.kv.binary_search_by_key(&key, |t| t.0.to_native()) {
                            Ok(index) => {
                                current = internal.kv[index].1.to_native();
                            }
                            Err(index) => {
                                current = internal.kv[index].1.to_native();
                            }
                        }
                        false
                    }
                })?
            {
                return Ok(path);
            }
        }
    }

    pub fn insert(&mut self, key: Key, value: Vec<u8>) -> Result<(), Error> {
        let path = self.path_to_leaf(key)?;

        let Node::Leaf(mut leaf) = self.pager.owned_node(path.last().copied().unwrap())? else {
            panic!("Expected leaf node");
        };

        match leaf.kv.binary_search_by_key(&key, |t| t.0) {
            Ok(index) => {
                leaf.kv[index].1 = value;
            }
            Err(index) => {
                leaf.kv.insert(index, (key, value));
            }
        }

        self.split_insert(&path, &Node::Leaf(leaf))?;

        Ok(())
    }

    pub fn split_insert(&mut self, path: &[NodePtr], insert: &Node) -> Result<(), Error> {
        let byte = rkyv::to_bytes(insert)?;

        if byte.len() <= PAGE_CONTENT_SIZE {
            self.pager
                .write_buffer(path.last().copied().unwrap(), byte)?;
            return Ok(());
        }

        match insert {
            Node::Leaf(leaf) => {
                let mid = leaf.kv.len() / 2;
                assert!(mid > 0);

                let right_leaf = Leaf {
                    kv: leaf.kv[mid..].to_vec(),
                };
                let left_leaf = Leaf {
                    kv: leaf.kv[..mid].to_vec(),
                };

                let left_key = left_leaf.kv.last().unwrap().0;
                let right_key = right_leaf.kv.last().unwrap().0;

                if path.len() == 1 {
                    let left_page_num = self.pager.next_page_num();
                    let right_page_num = self.pager.next_page_num();

                    let new_root = Internal {
                        kv: vec![(left_key, left_page_num), (right_key, right_page_num)],
                    };
                    self.pager
                        .write_node(ROOT_PAGE_NUM, &Node::Internal(new_root))?;

                    self.split_insert(&[ROOT_PAGE_NUM, left_key], &Node::Leaf(left_leaf))?;
                    self.split_insert(&[ROOT_PAGE_NUM, right_key], &Node::Leaf(right_leaf))?;
                } else {
                    let left_page_num = self.pager.next_page_num();
                    let right_page_num = path.last().copied().unwrap();

                    let Node::Internal(mut parent) = self.pager.owned_node(path[path.len() - 2])?
                    else {
                        panic!("Expected internal node");
                    };

                    match parent.kv.binary_search_by_key(&left_key, |t| t.0) {
                        Ok(_) => panic!("Duplicate key in internal node"),
                        Err(index) => {
                            parent.kv.insert(index, (left_key, left_page_num));
                        }
                    }
                    self.split_insert(&path[..path.len() - 1], &Node::Internal(parent))?;

                    let mut path = path.to_vec();
                    path.pop();
                    path.push(left_page_num);
                    self.split_insert(&path, &Node::Leaf(left_leaf))?;
                    path.pop();
                    path.push(right_page_num);
                    self.split_insert(&path, &Node::Leaf(right_leaf))?;
                }
            }
            Node::Internal(internal) => {
                let mid = internal.kv.len() / 2;
                assert!(mid > 0);

                let right_internal = Internal {
                    kv: internal.kv[mid..].to_vec(),
                };
                let left_internal = Internal {
                    kv: internal.kv[..mid].to_vec(),
                };

                let left_key = left_internal.kv.last().unwrap().0;
                let right_key = right_internal.kv.last().unwrap().0;

                if path.len() == 1 {
                    let left_page_num = self.pager.next_page_num();
                    self.pager
                        .write_node(left_page_num, &Node::Internal(left_internal))?;
                    let right_page_num = self.pager.next_page_num();
                    self.pager
                        .write_node(right_page_num, &Node::Internal(right_internal))?;

                    let new_root = Internal {
                        kv: vec![(left_key, left_page_num), (right_key, right_page_num)],
                    };
                    self.pager
                        .write_node(ROOT_PAGE_NUM, &Node::Internal(new_root))?;
                } else {
                    let left_page_num = self.pager.next_page_num();
                    self.pager
                        .write_node(left_page_num, &Node::Internal(left_internal))?;
                    let right_page_num = path.last().copied().unwrap();
                    self.pager
                        .write_node(right_page_num, &Node::Internal(right_internal))?;

                    let Node::Internal(mut parent) = self.pager.owned_node(path[path.len() - 2])?
                    else {
                        panic!("Expected internal node");
                    };

                    match parent.kv.binary_search_by_key(&left_key, |t| t.0) {
                        Ok(_) => panic!("Duplicate key in internal node"),
                        Err(index) => {
                            parent.kv.insert(index, (left_key, left_page_num));
                        }
                    }
                    self.split_insert(&path[..path.len() - 1], &Node::Internal(parent))?;
                }
            }
        }
        Ok(())
    }

    pub fn read<T>(&mut self, key: Key, f: impl FnOnce(Option<&[u8]>) -> T) -> Result<T, Error> {
        let leaf_page = self.search_leaf(key)?;

        let Node::Leaf(leaf) = self.pager.owned_node(leaf_page)? else {
            panic!("Expected leaf node");
        };

        match leaf.kv.binary_search_by_key(&key, |t| t.0) {
            Ok(index) => Ok(f(Some(&leaf.kv[index].1))),
            Err(_) => Ok(f(None)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_search() {
        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file);
        let mut btree = Btree::new(pager);
        let root_leaf = Leaf { kv: vec![] };
        btree
            .pager
            .write_node(ROOT_PAGE_NUM, &Node::Leaf(root_leaf))
            .unwrap();

        btree.search_leaf(0).unwrap();
    }

    #[test]
    fn test_insert() {
        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file);
        let mut btree = Btree::new(pager);
        let root_leaf = Leaf { kv: vec![] };
        btree
            .pager
            .write_node(ROOT_PAGE_NUM, &Node::Leaf(root_leaf))
            .unwrap();

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
        let root_leaf = Leaf { kv: vec![] };
        btree
            .pager
            .write_node(ROOT_PAGE_NUM, &Node::Leaf(root_leaf))
            .unwrap();

        btree.insert(42, b"forty-two".to_vec()).unwrap();
        assert!(
            btree
                .read(42, |v| v == Some(b"forty-two".as_ref()))
                .unwrap()
        );
    }
}
