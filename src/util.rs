use std::ops::{Bound, RangeBounds};

use crate::types::Key;

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
