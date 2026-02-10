use rkyv::{Archive, Deserialize, Serialize};

pub type Key = u64;
pub type NodePtr = u64;
pub const FREE_LIST_PAGE_NUM: u64 = 0;

#[derive(Archive, Deserialize, Serialize, Debug, Clone, PartialEq)]
pub enum Value {
    Inline(Vec<u8>),
    Overflow { start_page: u64, total_len: u64 },
}

#[derive(Archive, Deserialize, Serialize, Debug)]
pub struct Leaf {
    pub kv: Vec<(Key, Value)>,
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

/// Leaf node for an index tree, sorted by value (secondary index).
#[derive(Archive, Deserialize, Serialize, Debug)]
pub struct IndexLeaf {
    pub kv: Vec<(Value, Key)>,
}

/// Internal node for an index tree, routing by value.
#[derive(Archive, Deserialize, Serialize, Debug)]
pub struct IndexInternal {
    pub kv: Vec<(Value, NodePtr)>,
}

/// Node type for an index tree (sorted by value).
#[derive(Archive, Deserialize, Serialize, Debug)]
pub enum IndexNode {
    Leaf(IndexLeaf),
    Internal(IndexInternal),
}
