use criterion::{Criterion, criterion_group, criterion_main};
use mybtree::{Btree, Pager};

fn setup() -> (Btree<4096>, u64) {
    let file = tempfile::tempfile().unwrap();
    let pager = Pager::new(file);
    let mut btree = Btree::new(pager);
    btree.init().unwrap();
    let root = btree.init_tree().unwrap();
    (btree, root)
}

fn bench_sequential_insert(c: &mut Criterion) {
    let value = vec![0u8; 64];

    for n in [100, 1000, 10000] {
        c.bench_function(&format!("sequential_insert_{n}"), |b| {
            b.iter(|| {
                let (mut btree, root) = setup();
                for i in 0..n as u64 {
                    btree.insert(root, i, value.clone()).unwrap();
                }
                // flush on drop
            });
        });
    }
}

criterion_group!(benches, bench_sequential_insert);
criterion_main!(benches);
