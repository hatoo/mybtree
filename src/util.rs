use rkyv::rancor::Error;
use std::ops::{Bound, RangeBounds};

use crate::types::{Key, Leaf, Node, PAGE_CONTENT_SIZE};

pub fn split_leaf(kv: Vec<(Key, Vec<u8>)>) -> Result<Vec<Vec<(Key, Vec<u8>)>>, Error> {
    let mut result = vec![];
    let mut current = vec![kv];

    while let Some(kv) = current.pop() {
        let mid = kv.len() / 2;
        let left = kv[..mid].to_vec();
        let right = kv[mid..].to_vec();

        if rkyv::to_bytes(&Node::Leaf(Leaf { kv: left.clone() }))?.len() <= PAGE_CONTENT_SIZE {
            result.push(left);
        } else {
            current.push(left);
        }

        if rkyv::to_bytes(&Node::Leaf(Leaf { kv: right.clone() }))?.len() <= PAGE_CONTENT_SIZE {
            result.push(right);
        } else {
            current.push(right);
        }
    }
    result.sort_by_key(|kv| kv.last().unwrap().0);
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
