use std::io::{Read, Seek, Write};

use rkyv::{Archive, Deserialize, Serialize, deserialize, rancor::Error, util::AlignedVec};

pub const PAGE_SIZE: usize = 4096;
pub type Key = u64;
pub type NodePtr = u64;
pub const ROOT_PAGE_NUM: u64 = 0;
pub const PAGE_CONTENT_SIZE: usize = PAGE_SIZE - 2;

#[derive(Archive, Deserialize, Serialize, Debug)]
pub struct Leaf {
    pub parent: Option<NodePtr>,
    pub kv: Vec<(Key, Vec<u8>)>,
}

#[derive(Archive, Deserialize, Serialize, Debug)]
pub struct Internal {
    pub parent: Option<NodePtr>,
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

        let root_leaf = Leaf {
            parent: None,
            kv: vec![],
        };
        self.pager
            .write_node(ROOT_PAGE_NUM, &Node::Leaf(root_leaf))?;
        Ok(())
    }

    pub fn alloc_leaf(&mut self, key: Key) -> Result<NodePtr, Error> {
        let mut current = ROOT_PAGE_NUM;

        enum NextNode {
            Leaf(NodePtr),
            Next(NodePtr),
            NeedAlloc,
        }

        loop {
            let next = self
                .pager
                .read_node(current, |archived_node| match archived_node {
                    ArchivedNode::Leaf(_) => NextNode::Leaf(current),
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
                NextNode::Leaf(leaf_page) => return Ok(leaf_page),
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
                            parent: Some(current),
                            kv: vec![],
                        };
                        self.pager
                            .write_node(new_leaf_page, &Node::Leaf(new_leaf))?;

                        internal.kv.push((key, new_leaf_page));
                        self.pager.write_node(current, &Node::Internal(internal))?;

                        return Ok(new_leaf_page);
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

    pub fn insert(&mut self, key: Key, value: Vec<u8>) -> Result<(), Error> {
        let leaf_page = self.alloc_leaf(key)?;

        let Node::Leaf(mut leaf) = self.pager.owned_node(leaf_page)? else {
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

        self.split_insert(leaf_page, &Node::Leaf(leaf))?;

        Ok(())
    }

    pub fn split_insert(&mut self, page: NodePtr, insert: &Node) -> Result<(), Error> {
        let buffer = rkyv::to_bytes(insert)?;

        if buffer.len() <= PAGE_CONTENT_SIZE {
            self.pager.write_buffer(page, buffer)?;
            return Ok(());
        }

        match insert {
            Node::Leaf(leaf) => {
                let mid = leaf.kv.len() / 2;
                assert!(mid > 0);

                let left_leaf = Leaf {
                    parent: leaf.parent,
                    kv: leaf.kv[..mid].to_vec(),
                };
                let right_leaf = Leaf {
                    parent: leaf.parent,
                    kv: leaf.kv[mid..].to_vec(),
                };

                let left_key = left_leaf.kv.last().unwrap().0;
                let right_key = right_leaf.kv.last().unwrap().0;
                let left_page = self.pager.next_page_num();

                if let Some(parent) = leaf.parent {
                    let parent_node = self.pager.owned_node(parent)?;
                    if let Node::Internal(mut internal) = parent_node {
                        match internal.kv.binary_search_by_key(&left_key, |t| t.0) {
                            Ok(_) => {
                                panic!("Duplicate key in internal node");
                            }
                            Err(index) => {
                                internal.kv[index].0 = right_key;
                                internal.kv.insert(index, (left_key, left_page));
                            }
                        }

                        self.split_insert(parent, &Node::Internal(internal))?;
                        self.split_insert(left_page, &Node::Leaf(left_leaf))?;
                        self.split_insert(page, &Node::Leaf(right_leaf))?;
                        Ok(())
                    } else {
                        panic!("Parent is not an internal node");
                    }
                } else {
                    let right_page = self.pager.next_page_num();
                    let new_internal = Internal {
                        parent: None,
                        kv: vec![(left_key, left_page), (right_key, right_page)],
                    };
                    self.pager
                        .write_node(ROOT_PAGE_NUM, &Node::Internal(new_internal))?;
                    self.split_insert(left_page, &Node::Leaf(left_leaf))?;
                    self.split_insert(right_page, &Node::Leaf(right_leaf))?;
                    Ok(())
                }
            }
            Node::Internal(internal) => {
                let mid = internal.kv.len() / 2;

                let left_internal = Internal {
                    parent: internal.parent,
                    kv: internal.kv[..mid].to_vec(),
                };
                let right_internal = Internal {
                    parent: internal.parent,
                    kv: internal.kv[mid..].to_vec(),
                };

                let left_key = left_internal.kv.last().unwrap().0;
                let right_key = right_internal.kv.last().unwrap().0;
                let left_page = self.pager.next_page_num();

                if let Some(parent) = internal.parent {
                    let parent_node = self.pager.owned_node(parent)?;
                    if let Node::Internal(mut parent_internal) = parent_node {
                        match parent_internal.kv.binary_search_by_key(&left_key, |t| t.0) {
                            Ok(_) => {
                                panic!("Duplicate key in internal node");
                            }
                            Err(index) => {
                                parent_internal.kv.insert(index, (left_key, left_page));
                            }
                        }

                        self.split_insert(parent, &Node::Internal(parent_internal))?;
                        self.split_insert(left_page, &Node::Internal(left_internal))?;
                        self.split_insert(page, &Node::Internal(right_internal))?;
                        Ok(())
                    } else {
                        panic!("Parent is not an internal node");
                    }
                } else {
                    let right_page = self.pager.next_page_num();
                    let new_internal = Internal {
                        parent: None,
                        kv: vec![(left_key, left_page), (right_key, right_page)],
                    };
                    self.pager
                        .write_node(ROOT_PAGE_NUM, &Node::Internal(new_internal))?;
                    self.split_insert(left_page, &Node::Internal(left_internal))?;
                    self.split_insert(right_page, &Node::Internal(right_internal))?;
                    Ok(())
                }
            }
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
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    #[test]
    fn test_search() {
        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file);
        let mut btree = Btree::new(pager);
        btree.init().unwrap();
        let leaf = Leaf {
            parent: None,
            kv: vec![(0, b"zero".to_vec()), (10, b"ten".to_vec())],
        };
        btree
            .pager
            .write_node(ROOT_PAGE_NUM, &Node::Leaf(leaf))
            .unwrap();

        btree.alloc_leaf(0).unwrap();
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
            parent: None,
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
}
