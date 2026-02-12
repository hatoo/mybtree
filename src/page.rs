use crate::{Key, NodePtr};

#[repr(u32)]
pub enum PageType {
    Leaf,
    Internal,
}

#[repr(align(4096))]
#[derive(Clone)]
pub struct InternalPage<const N: usize> {
    page: [u8; N],
}

impl<const N: usize> InternalPage<N> {
    const LEN_OFFSET: usize = 4;
    const HEADER_SIZE: usize = 6;
    const ELEMENT_SIZE: usize = std::mem::size_of::<Key>() + std::mem::size_of::<NodePtr>();

    pub fn new() -> Self {
        let mut internal = Self { page: [0; N] };
        internal.page[0..4].copy_from_slice(&(PageType::Internal as u32).to_le_bytes());
        internal
    }

    pub fn len(&self) -> usize {
        usize::from(u16::from_le_bytes([
            self.page[Self::LEN_OFFSET],
            self.page[Self::LEN_OFFSET + 1],
        ]))
    }

    fn set_len(&mut self, len: usize) {
        debug_assert!(len <= u16::MAX as usize);
        let len_bytes = (len as u16).to_le_bytes();
        self.page[Self::LEN_OFFSET] = len_bytes[0];
        self.page[Self::LEN_OFFSET + 1] = len_bytes[1];
    }

    pub fn can_insert(&self) -> bool {
        self.len() * (Self::ELEMENT_SIZE + 1) + Self::HEADER_SIZE <= N
    }

    fn key(&self, index: usize) -> Key {
        debug_assert!(index < self.len());
        let offset = Self::HEADER_SIZE + index * Self::ELEMENT_SIZE;
        Key::from_le_bytes(
            self.page[offset..offset + std::mem::size_of::<Key>()]
                .try_into()
                .unwrap(),
        )
    }

    fn ptr(&self, index: usize) -> NodePtr {
        debug_assert!(index < self.len());
        let offset = Self::HEADER_SIZE + index * Self::ELEMENT_SIZE + std::mem::size_of::<Key>();
        NodePtr::from_le_bytes(
            self.page[offset..offset + std::mem::size_of::<NodePtr>()]
                .try_into()
                .unwrap(),
        )
    }

    fn write_element(&mut self, index: usize, key: Key, ptr: NodePtr) {
        debug_assert!(index <= self.len());
        let offset = Self::HEADER_SIZE + index * Self::ELEMENT_SIZE;
        self.page[offset..offset + std::mem::size_of::<Key>()].copy_from_slice(&key.to_le_bytes());
        self.page[offset + std::mem::size_of::<Key>()..offset + Self::ELEMENT_SIZE]
            .copy_from_slice(&ptr.to_le_bytes());
    }

    fn search_index(&self, key: Key) -> Option<usize> {
        let mut left = 0;
        let mut right = self.len();
        while left < right {
            let mid = (left + right) / 2;
            if self.key(mid) < key {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if left == self.len() { None } else { Some(left) }
    }

    pub fn insert(&mut self, key: Key, ptr: NodePtr) {
        debug_assert!(self.can_insert());

        match self.search_index(key) {
            Some(idx) => {
                let len = self.len();
                self.set_len(len + 1);
                for i in (idx..len).rev() {
                    let k = self.key(i);
                    let p = self.ptr(i);
                    self.write_element(i + 1, k, p);
                }
                self.write_element(idx, key, ptr);
            }
            None => {
                let len = self.len();
                self.set_len(len + 1);
                self.write_element(len, key, ptr);
            }
        }
    }

    pub fn remove(&mut self, index: usize) {
        debug_assert!(index < self.len());
        let len = self.len();
        for i in index + 1..len {
            let k = self.key(i);
            let p = self.ptr(i);
            self.write_element(i - 1, k, p);
        }
        self.set_len(len - 1);
    }

    pub fn split(&mut self) -> Self {
        let len = self.len();
        let mid = len / 2;
        let mut new_page = Self::new();
        for i in mid..len {
            let k = self.key(i);
            let p = self.ptr(i);
            new_page.insert(k, p);
        }
        self.set_len(mid);
        new_page
    }
}
