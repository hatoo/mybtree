mod pager;
mod tree;
mod types;
mod util;

// Re-export public API
pub use pager::Pager;
pub use tree::Btree;
pub use types::{Internal, Key, Leaf, Node, NodePtr, PAGE_CONTENT_SIZE, PAGE_SIZE, ROOT_PAGE_NUM};

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
}
