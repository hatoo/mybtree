use rkyv::{Archive, Deserialize, Serialize};

pub type Key = u64;
pub type NodePtr = u64;
pub const ROOT_PAGE_NUM: u64 = 0;

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
