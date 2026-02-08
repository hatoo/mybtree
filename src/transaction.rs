use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

use rkyv::rancor::{Error, fail};

use crate::{Btree, Key};

#[derive(Debug, thiserror::Error)]
pub enum TransactionError {
    #[error("Transaction conflict detected")]
    Conflict,
}

struct TransactionStoreInner {
    btree: Btree,
    next_tx_id: usize,
    active_transactions: BTreeMap<usize, Operation>,
}

pub struct TransactionStore {
    inner: Arc<Mutex<TransactionStoreInner>>,
}

pub struct Transaction {
    store: Arc<Mutex<TransactionStoreInner>>,
    tx_id: usize,
}

pub struct Operation {
    reads: BTreeSet<Key>,
    writes: BTreeMap<Key, Vec<u8>>,
}

impl TransactionStore {
    pub fn new(btree: Btree) -> Self {
        TransactionStore {
            inner: Arc::new(Mutex::new(TransactionStoreInner {
                btree,
                next_tx_id: 0,
                active_transactions: BTreeMap::new(),
            })),
        }
    }

    pub fn begin_transaction(&self) -> Transaction {
        let mut inner = self.inner.lock().unwrap();
        let tx_id = inner.next_tx_id;
        inner.next_tx_id = inner.next_tx_id.wrapping_add(1);
        inner.active_transactions.insert(
            tx_id,
            Operation {
                reads: BTreeSet::new(),
                writes: BTreeMap::new(),
            },
        );
        Transaction {
            store: Arc::clone(&self.inner),
            tx_id,
        }
    }
}

impl Transaction {
    pub fn read(&self, key: Key) -> Result<Option<Vec<u8>>, Error> {
        let mut inner = self.store.lock().unwrap();
        if let Some(op) = inner.active_transactions.get(&self.tx_id) {
            if let Some(value) = op.writes.get(&key) {
                return Ok(Some(value.clone()));
            }
        }
        let value = inner.btree.read(key, |f| f.map(|v| v.to_vec()))?;
        Ok(value)
    }

    pub fn write(&self, key: Key, value: Vec<u8>) {
        let mut inner = self.store.lock().unwrap();
        if let Some(op) = inner.active_transactions.get_mut(&self.tx_id) {
            op.writes.insert(key, value);
        }
    }

    pub fn commit(self) -> Result<(), Error> {
        let mut inner = self.store.lock().unwrap();

        let current_op = inner.active_transactions.remove(&self.tx_id).unwrap();

        for (other_tx_id, other_op) in &inner.active_transactions {
            if *other_tx_id != self.tx_id {
                for read_key in &other_op.reads {
                    if current_op.writes.contains_key(read_key) {
                        fail!(TransactionError::Conflict);
                    }
                }
                for write_key in other_op.writes.keys() {
                    if current_op.reads.contains(write_key) {
                        fail!(TransactionError::Conflict);
                    }
                    if current_op.writes.contains_key(write_key) {
                        fail!(TransactionError::Conflict);
                    }
                }
            }
        }

        for (key, value) in current_op.writes {
            inner.btree.insert(key, value)?;
        }

        Ok(())
    }
}
