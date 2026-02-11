use rkyv::rancor::Error;
use std::ops::{Bound, RangeBounds};

use crate::types::{IndexInternal, IndexLeaf, IndexNode};
use crate::types::{Key, Leaf, Node, NodePtr, Value};

pub fn split_leaf(
    kv: Vec<(Key, Value)>,
    page_content_size: usize,
) -> Result<Vec<Vec<(Key, Value)>>, Error> {
    let mut result = vec![];
    let mut current = vec![kv];

    while let Some(mut kv) = current.pop() {
        debug_assert!(kv.len() >= 2);
        let mid = kv.len() / 2;
        let right = kv.split_off(mid);
        let left = kv;

        if rkyv::to_bytes(&Node::Leaf(Leaf { kv: left.clone() }))?.len() <= page_content_size {
            result.push(left);
        } else {
            current.push(left);
        }

        if rkyv::to_bytes(&Node::Leaf(Leaf { kv: right.clone() }))?.len() <= page_content_size {
            result.push(right);
        } else {
            current.push(right);
        }
    }
    result.sort_by_key(|kv| kv.last().unwrap().0);
    Ok(result)
}

pub fn split_internal(
    kv: Vec<(Key, u64)>,
    page_content_size: usize,
) -> Result<Vec<Vec<(Key, u64)>>, Error> {
    let mut result = vec![];
    let mut current = vec![kv];

    while let Some(mut kv) = current.pop() {
        debug_assert!(kv.len() >= 2);
        let mid = kv.len() / 2;
        let right = kv.split_off(mid);
        let left = kv;

        if rkyv::to_bytes(&Node::Internal(crate::types::Internal { kv: left.clone() }))?.len()
            <= page_content_size
        {
            result.push(left);
        } else {
            current.push(left);
        }

        if rkyv::to_bytes(&Node::Internal(crate::types::Internal {
            kv: right.clone(),
        }))?
        .len()
            <= page_content_size
        {
            result.push(right);
        } else {
            current.push(right);
        }
    }
    result.sort_by_key(|kv| kv.last().unwrap().0);
    Ok(result)
}

pub fn split_index_leaf(
    kv: Vec<(Value, Key)>,
    page_content_size: usize,
) -> Result<Vec<Vec<(Value, Key)>>, Error> {
    let mut result = vec![];
    let mut current = vec![kv];

    while let Some(mut kv) = current.pop() {
        debug_assert!(kv.len() >= 2);
        let mid = kv.len() / 2;
        let right = kv.split_off(mid);
        let left = kv;

        if rkyv::to_bytes(&IndexNode::Leaf(IndexLeaf { kv: left.clone() }))?.len()
            <= page_content_size
        {
            result.push(left);
        } else {
            current.push(left);
        }

        if rkyv::to_bytes(&IndexNode::Leaf(IndexLeaf { kv: right.clone() }))?.len()
            <= page_content_size
        {
            result.push(right);
        } else {
            current.push(right);
        }
    }
    // Caller must sort the result chunks (Value comparison requires I/O).
    Ok(result)
}

pub fn split_index_internal(
    kv: Vec<(Value, NodePtr)>,
    page_content_size: usize,
) -> Result<Vec<Vec<(Value, NodePtr)>>, Error> {
    let mut result = vec![];
    let mut current = vec![kv];

    while let Some(mut kv) = current.pop() {
        debug_assert!(kv.len() >= 2);
        let mid = kv.len() / 2;
        let right = kv.split_off(mid);
        let left = kv;

        if rkyv::to_bytes(&IndexNode::Internal(IndexInternal { kv: left.clone() }))?.len()
            <= page_content_size
        {
            result.push(left);
        } else {
            current.push(left);
        }

        if rkyv::to_bytes(&IndexNode::Internal(IndexInternal { kv: right.clone() }))?.len()
            <= page_content_size
        {
            result.push(right);
        } else {
            current.push(right);
        }
    }
    // Caller must sort the result chunks (Value comparison requires I/O).
    Ok(result)
}

pub fn is_overlap<R1: RangeBounds<Key>, R2: RangeBounds<Key>>(range1: &R1, range2: &R2) -> bool {
    let start1 = match range1.start_bound() {
        Bound::Included(&b) => b,
        Bound::Excluded(&b) => {
            if b == Key::MAX {
                return false;
            }
            b + 1
        }
        Bound::Unbounded => Key::MIN,
    };
    let end1 = match range1.end_bound() {
        Bound::Included(&b) => b,
        Bound::Excluded(&b) => {
            if b == Key::MIN {
                return false;
            }
            b - 1
        }
        Bound::Unbounded => Key::MAX,
    };
    let start2 = match range2.start_bound() {
        Bound::Included(&b) => b,
        Bound::Excluded(&b) => {
            if b == Key::MAX {
                return false;
            }
            b + 1
        }
        Bound::Unbounded => Key::MIN,
    };
    let end2 = match range2.end_bound() {
        Bound::Included(&b) => b,
        Bound::Excluded(&b) => {
            if b == Key::MIN {
                return false;
            }
            b - 1
        }
        Bound::Unbounded => Key::MAX,
    };

    !(end1 < start2 || end2 < start1)
}
