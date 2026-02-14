use criterion::{Criterion, criterion_group, criterion_main};
use mybtree::{Btree, Pager};

fn setup_tree() -> (Btree<4096>, u64) {
    let file = tempfile::tempfile().unwrap();
    let pager = Pager::new(file);
    let mut btree = Btree::new(pager);
    btree.init().unwrap();
    let root = btree.init_tree().unwrap();
    (btree, root)
}

fn setup_index_tree() -> (Btree<4096>, u64) {
    let file = tempfile::tempfile().unwrap();
    let pager = Pager::new(file);
    let mut btree = Btree::new(pager);
    btree.init().unwrap();
    let root = btree.init_index().unwrap();
    (btree, root)
}

fn bench_sequential_insert(c: &mut Criterion) {
    let value = vec![0u8; 64];

    for n in [100, 1000, 10000] {
        c.bench_function(&format!("sequential_insert_{n}"), |b| {
            b.iter(|| {
                let (mut btree, root) = setup_tree();
                for i in 0..n as u64 {
                    btree.insert(root, i, &value).unwrap();
                }
                // flush on drop
            });
        });
        c.bench_function(&format!("sequential_remove_{n}"), |b| {
            let (mut btree, root) = setup_tree();
            for i in 0..n as u64 {
                btree.insert(root, i, &value).unwrap();
            }
            b.iter(|| {
                for i in 0..n as u64 {
                    btree.remove(root, i).unwrap();
                }
                // flush on drop
            });
        });

        c.bench_function(&format!("sequential_read_{n}"), |b| {
            let (mut btree, root) = setup_tree();
            for i in 0..n as u64 {
                btree.insert(root, i, &value).unwrap();
            }
            b.iter(|| {
                for i in 0..n as u64 {
                    btree.read(root, i).unwrap();
                }
                // flush on drop
            });
        });

        c.bench_function(&format!("sequential_index_insert_{n}"), |b| {
            let (mut btree, root) = setup_index_tree();
            let mut value = vec![0u8; 64];
            b.iter(|| {
                for i in 0..n as u64 {
                    value[0..8].copy_from_slice(&i.to_be_bytes());
                    btree.index_insert(root, i, &value).unwrap();
                }
                // flush on drop
            });
        });

        c.bench_function(&format!("sequential_index_remove_{n}"), |b| {
            let (mut btree, root) = setup_index_tree();
            let mut value = vec![0u8; 64];

            for i in 0..n as u64 {
                value[0..8].copy_from_slice(&i.to_be_bytes());
                btree.index_insert(root, i, &value).unwrap();
            }

            b.iter(|| {
                for i in 0..n as u64 {
                    value[0..8].copy_from_slice(&i.to_be_bytes());
                    btree.index_remove(root, &value, i).unwrap();
                }
                // flush on drop
            });
        });
    }
}

criterion_group!(benches, bench_sequential_insert);
criterion_main!(benches);
