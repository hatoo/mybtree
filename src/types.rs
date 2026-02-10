use rkyv::{Archive, Deserialize, Serialize};

pub type Key = u64;
pub type NodePtr = u64;
pub const ROOT_PAGE_NUM: u64 = 0;
pub const FREE_LIST_PAGE_NUM: u64 = 1;

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
