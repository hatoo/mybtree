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
}

impl Pager {
    pub fn new(file: std::fs::File) -> Self {
        Pager { file }
    }

    pub fn read_node<T>(
        &mut self,
        page_num: u64,
        f: impl FnOnce(&rkyv::Archived<Node>) -> T,
    ) -> Result<T, Error> {
        let mut buffer = AlignedVec::<PAGE_SIZE>::with_capacity(PAGE_SIZE);
        buffer.resize(PAGE_SIZE, 0);
        self.file
            .seek(std::io::SeekFrom::Start(page_num * PAGE_SIZE as u64))
            .unwrap();
        self.file.read_exact(&mut buffer).unwrap();
        let len =
            u16::from_le_bytes([buffer[PAGE_CONTENT_SIZE], buffer[PAGE_CONTENT_SIZE + 1]]) as usize;
        let buffer = &buffer[..len];
        let archived = rkyv::api::high::access::<rkyv::Archived<Node>, Error>(&buffer)?;
        Ok(f(archived))
    }

    pub fn owned_node(&mut self, page_num: u64) -> Result<Node, Error> {
        let mut buffer = AlignedVec::<PAGE_SIZE>::with_capacity(PAGE_SIZE);
        buffer.resize(PAGE_SIZE, 0);
        self.file
            .seek(std::io::SeekFrom::Start(page_num * PAGE_SIZE as u64))
            .unwrap();
        self.file.read_exact(&mut buffer).unwrap();
        let len =
            u16::from_le_bytes([buffer[PAGE_CONTENT_SIZE], buffer[PAGE_CONTENT_SIZE + 1]]) as usize;
        let buffer = &buffer[..len];
        let archived = rkyv::access::<ArchivedNode, Error>(&buffer)?;
        let node: Node = deserialize(archived)?;
        Ok(node)
    }

    pub fn write_node(&mut self, page_num: u64, node: &Node) -> Result<(), Error> {
        let mut bytes = rkyv::to_bytes(node)?;
        assert!(bytes.len() <= PAGE_CONTENT_SIZE);
        let len = bytes.len() as u16;
        bytes.resize(PAGE_SIZE, 0);
        bytes[PAGE_CONTENT_SIZE..PAGE_SIZE].copy_from_slice(&len.to_le_bytes());

        self.file
            .seek(std::io::SeekFrom::Start(page_num * PAGE_SIZE as u64))
            .unwrap();
        self.file.write_all(&bytes).unwrap();

        Ok(())
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

    pub fn insert(&mut self, key: Key, value: Vec<u8>) -> Result<(), Error> {
        let leaf_page = self.search_leaf(key)?;

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

        self.pager.write_node(leaf_page, &Node::Leaf(leaf))?;

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
