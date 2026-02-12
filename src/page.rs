use std::fmt;

use crate::{Key, NodePtr};

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageType {
    Leaf,
    Internal,
}

#[repr(align(4096))]
#[derive(Clone)]
pub struct InternalPage<const N: usize> {
    page: [u8; N],
}

#[repr(align(4096))]
#[derive(Clone)]
pub struct LeafPage<const N: usize> {
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

impl<const N: usize> fmt::Debug for InternalPage<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for i in 0..self.len() {
            list.entry(&(self.key(i), self.ptr(i)));
        }
        list.finish()
    }
}

#[repr(C)]
struct LeafElement {
    key: Key,
    value_offset: u16,
    value_len: u16,
}

impl<const N: usize> LeafPage<N> {
    const LEN_OFFSET: usize = 4;
    const HEADER_SIZE: usize = 6;
    

    fn new() -> Self {
        let mut leaf = Self { page: [0; N] };
        leaf.page[0..4].copy_from_slice(&(PageType::Leaf as u32).to_le_bytes());
        leaf
    }

    fn len(&self) -> usize {
        usize::from(u16::from_le_bytes([
            self.page[LeafPage::<N>::LEN_OFFSET],
            self.page[LeafPage::<N>::LEN_OFFSET + 1],
        ]))
    }

    fn set_len(&mut self, len: usize) {
        debug_assert!(len <= u16::MAX as usize);
        let len_bytes = (len as u16).to_le_bytes();
        self.page[LeafPage::<N>::LEN_OFFSET] = len_bytes[0];
        self.page[LeafPage::<N>::LEN_OFFSET + 1] = len_bytes[1];
    }

    fn key(&self, index: usize) -> Key {
        debug_assert!(index < self.len());
        let offset = LeafPage::<N>::HEADER_SIZE + index * std::mem::size_of::<LeafElement>();
        Key::from_le_bytes(
            self.page[offset..offset + std::mem::size_of::<Key>()]
                .try_into()
                .unwrap(),
        )
    }

    fn value(&self, index: usize) -> &[u8] {
        debug_assert!(index < self.len());
        let offset = LeafPage::<N>::HEADER_SIZE + index * std::mem::size_of::<LeafElement>();
        let value_offset = u16::from_le_bytes([
            self.page[offset + std::mem::size_of::<Key>()],
            self.page[offset + std::mem::size_of::<Key>() + 1],
        ]) as usize;
        let value_len = u16::from_le_bytes([
            self.page[offset + std::mem::size_of::<Key>() + 2],
            self.page[offset + std::mem::size_of::<Key>() + 3],
        ]) as usize;
        &self.page[value_offset..value_offset + value_len]
    }

    fn search(&self, key: Key) -> Option<usize> {
        let mut left = 0;
        let mut right = self.len();
        while left < right {
            let mid = (left + right) / 2;
            let mid_key = self.key(mid);
            if mid_key < key {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if left == self.len() { None } else { Some(left) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_page_new_is_empty() {
        let page = InternalPage::<4096>::new();
        assert_eq!(page.len(), 0);
    }

    #[test]
    fn internal_page_insert_single() {
        let mut page = InternalPage::<4096>::new();
        page.insert(10, 100);
        assert_eq!(page.len(), 1);
        assert_eq!(page.key(0), 10);
        assert_eq!(page.ptr(0), 100);
    }

    #[test]
    fn internal_page_insert_maintains_sorted_order() {
        let mut page = InternalPage::<4096>::new();
        page.insert(30, 300);
        page.insert(10, 100);
        page.insert(20, 200);

        assert_eq!(page.len(), 3);
        assert_eq!(page.key(0), 10);
        assert_eq!(page.key(1), 20);
        assert_eq!(page.key(2), 30);
        assert_eq!(page.ptr(0), 100);
        assert_eq!(page.ptr(1), 200);
        assert_eq!(page.ptr(2), 300);
    }

    #[test]
    fn internal_page_remove_middle() {
        let mut page = InternalPage::<4096>::new();
        page.insert(10, 100);
        page.insert(20, 200);
        page.insert(30, 300);

        page.remove(1);
        assert_eq!(page.len(), 2);
        assert_eq!(page.key(0), 10);
        assert_eq!(page.key(1), 30);
    }

    #[test]
    fn internal_page_split_divides_elements() {
        let mut page = InternalPage::<4096>::new();
        for i in 0..6 {
            page.insert(i * 10, i * 100);
        }

        let right = page.split();
        assert_eq!(page.len(), 3);
        assert_eq!(right.len(), 3);

        // left has first half
        assert_eq!(page.key(0), 0);
        assert_eq!(page.key(2), 20);

        // right has second half
        assert_eq!(right.key(0), 30);
        assert_eq!(right.key(2), 50);
    }

    #[test]
    fn internal_page_can_insert_on_new() {
        let page = InternalPage::<4096>::new();
        assert!(page.can_insert());
    }
}
