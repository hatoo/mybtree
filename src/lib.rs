mod types;
mod pager;
mod util;
mod tree;

// Re-export public API
pub use types::{Leaf, Internal, Node, Key, NodePtr, ROOT_PAGE_NUM, PAGE_SIZE, PAGE_CONTENT_SIZE};
pub use pager::Pager;
pub use tree::Btree;

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use std::collections::HashMap;

    use super::*;

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

    #[test]
    fn test_remove_range() {
        const LEN: u64 = 200;
        let file = tempfile::tempfile().unwrap();
        let pager = Pager::new(file);
        let mut btree = Btree::new(pager);
        btree.init().unwrap();

        for i in 0..LEN {
            btree
                .insert(i, format!("value-{}", i).as_bytes().to_vec())
                .unwrap();
        }

        btree.remove_range(50..150).unwrap();

        for i in 0..LEN {
            let found = btree.read(i, |v| v.map(|v| v.to_vec())).unwrap();
            if (50..150).contains(&i) {
                assert!(found.is_none(), "Expected key {} to be removed", i);
            } else {
                assert_eq!(
                    found,
                    Some(format!("value-{}", i).as_bytes().to_vec()),
                    "Expected key {} to remain",
                    i
                );
            }
        }
    }

    #[test]
    fn test_remove_range_edge_cases() {
        const LEN: u64 = 50;

        let setup = || {
            let file = tempfile::tempfile().unwrap();
            let pager = Pager::new(file);
            let mut btree = Btree::new(pager);
            btree.init().unwrap();

            for i in 0..LEN {
                btree
                    .insert(i, format!("value-{}", i).as_bytes().to_vec())
                    .unwrap();
            }
            btree
        };

        // Empty range should be a no-op.
        let mut btree = setup();
        btree.remove_range(10..10).unwrap();
        for i in 0..LEN {
            assert!(
                btree
                    .read(i, |v| v == Some(format!("value-{}", i).as_bytes()))
                    .unwrap(),
                "Expected key {} to remain after empty range",
                i
            );
        }

        btree.remove_range(10..11).unwrap();
        for i in 0..LEN {
            if i == 10 {
                assert!(
                    btree.read(i, |v| v.is_none()).unwrap(),
                    "Expected key {} to be removed",
                    i
                );
            } else {
                assert!(
                    btree
                        .read(i, |v| v == Some(format!("value-{}", i).as_bytes()))
                        .unwrap(),
                    "Expected key {} to remain after empty range",
                    i
                );
            }
        }

        // Range outside existing keys should be a no-op.
        let mut btree = setup();
        btree.remove_range(100..150).unwrap();
        for i in 0..LEN {
            assert!(
                btree
                    .read(i, |v| v == Some(format!("value-{}", i).as_bytes()))
                    .unwrap(),
                "Expected key {} to remain after out-of-range delete",
                i
            );
        }

        // Full range should remove everything.
        let mut btree = setup();
        btree.remove_range(0..LEN).unwrap();
        for i in 0..LEN {
            assert!(
                btree.read(i, |v| v.is_none()).unwrap(),
                "Expected key {} to be removed by full range",
                i
            );
        }
    }
}
