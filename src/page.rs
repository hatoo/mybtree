use std::borrow::Cow;
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

impl<const N: usize> LeafPage<N> {
    const LEN_OFFSET: usize = 4;
    const DATA_OFFSET: usize = 6;
    const HEADER_SIZE: usize = 8;
    const SLOT_SIZE: usize = std::mem::size_of::<Key>() + 2 + 2; // key + value_offset + value_len

    pub fn new() -> Self {
        let mut leaf = Self { page: [0; N] };
        leaf.page[0..4].copy_from_slice(&(PageType::Leaf as u32).to_le_bytes());
        leaf.set_data_offset(N);
        leaf
    }

    pub fn len(&self) -> usize {
        usize::from(u16::from_le_bytes([
            self.page[Self::LEN_OFFSET],
            self.page[Self::LEN_OFFSET + 1],
        ]))
    }

    fn set_len(&mut self, len: usize) {
        debug_assert!(len <= u16::MAX as usize);
        self.page[Self::LEN_OFFSET..Self::LEN_OFFSET + 2]
            .copy_from_slice(&(len as u16).to_le_bytes());
    }

    fn data_offset(&self) -> usize {
        usize::from(u16::from_le_bytes([
            self.page[Self::DATA_OFFSET],
            self.page[Self::DATA_OFFSET + 1],
        ]))
    }

    fn set_data_offset(&mut self, offset: usize) {
        debug_assert!(offset <= u16::MAX as usize);
        self.page[Self::DATA_OFFSET..Self::DATA_OFFSET + 2]
            .copy_from_slice(&(offset as u16).to_le_bytes());
    }

    pub fn key(&self, index: usize) -> Key {
        debug_assert!(index < self.len());
        let offset = Self::HEADER_SIZE + index * Self::SLOT_SIZE;
        Key::from_le_bytes(
            self.page[offset..offset + std::mem::size_of::<Key>()]
                .try_into()
                .unwrap(),
        )
    }

    pub fn value(&self, index: usize) -> &[u8] {
        debug_assert!(index < self.len());
        let slot = Self::HEADER_SIZE + index * Self::SLOT_SIZE + std::mem::size_of::<Key>();
        let value_offset = u16::from_le_bytes([self.page[slot], self.page[slot + 1]]) as usize;
        let value_len = u16::from_le_bytes([self.page[slot + 2], self.page[slot + 3]]) as usize;
        &self.page[value_offset..value_offset + value_len]
    }

    fn write_slot(&mut self, index: usize, key: Key, value_offset: u16, value_len: u16) {
        let offset = Self::HEADER_SIZE + index * Self::SLOT_SIZE;
        self.page[offset..offset + std::mem::size_of::<Key>()].copy_from_slice(&key.to_le_bytes());
        self.page[offset + std::mem::size_of::<Key>()..offset + std::mem::size_of::<Key>() + 2]
            .copy_from_slice(&value_offset.to_le_bytes());
        self.page[offset + std::mem::size_of::<Key>() + 2..offset + Self::SLOT_SIZE]
            .copy_from_slice(&value_len.to_le_bytes());
    }

    fn read_slot(&self, index: usize) -> (Key, u16, u16) {
        let offset = Self::HEADER_SIZE + index * Self::SLOT_SIZE;
        let key = Key::from_le_bytes(
            self.page[offset..offset + std::mem::size_of::<Key>()]
                .try_into()
                .unwrap(),
        );
        let vo = u16::from_le_bytes([
            self.page[offset + std::mem::size_of::<Key>()],
            self.page[offset + std::mem::size_of::<Key>() + 1],
        ]);
        let vl = u16::from_le_bytes([
            self.page[offset + std::mem::size_of::<Key>() + 2],
            self.page[offset + std::mem::size_of::<Key>() + 3],
        ]);
        (key, vo, vl)
    }

    fn search(&self, key: Key) -> Option<usize> {
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

    pub fn get(&self, key: Key) -> Option<Cow<'_, [u8]>> {
        let idx = self.search(key)?;
        if self.key(idx) == key {
            Some(Cow::Borrowed(self.value(idx)))
        } else {
            None
        }
    }

    /// Contiguous free space between end of slot array and start of value data.
    fn contiguous_free_space(&self) -> usize {
        let slots_end = Self::HEADER_SIZE + self.len() * Self::SLOT_SIZE;
        self.data_offset().saturating_sub(slots_end)
    }

    /// Total free space including dead gaps left by removed values.
    fn free_space(&self) -> usize {
        let used_value_space: usize = (0..self.len()).map(|i| self.read_slot(i).2 as usize).sum();
        N - Self::HEADER_SIZE - self.len() * Self::SLOT_SIZE - used_value_space
    }

    pub fn can_insert(&self, value_len: usize) -> bool {
        self.free_space() >= Self::SLOT_SIZE + value_len
    }

    /// Compact the value data region, eliminating dead gaps.
    fn compact(&mut self) {
        let len = self.len();
        let mut new_data_offset = N;
        for i in 0..len {
            let (k, vo, vl) = self.read_slot(i);
            let val_len = vl as usize;
            new_data_offset -= val_len;
            // Copy value to new position (use copy_within to handle overlap)
            self.page
                .copy_within(vo as usize..vo as usize + val_len, new_data_offset);
            self.write_slot(i, k, new_data_offset as u16, vl);
        }
        self.set_data_offset(new_data_offset);
    }

    pub fn insert(&mut self, key: Key, value: &[u8]) {
        debug_assert!(self.can_insert(value.len()));

        // Compact if contiguous free space is insufficient
        if self.contiguous_free_space() < Self::SLOT_SIZE + value.len() {
            self.compact();
        }

        // Allocate value space from the end
        let new_data_offset = self.data_offset() - value.len();
        self.page[new_data_offset..new_data_offset + value.len()].copy_from_slice(value);
        self.set_data_offset(new_data_offset);

        let idx = self.search(key).unwrap_or(self.len());
        let len = self.len();

        // Shift slots right
        for i in (idx..len).rev() {
            let (k, vo, vl) = self.read_slot(i);
            self.write_slot(i + 1, k, vo, vl);
        }

        self.write_slot(idx, key, new_data_offset as u16, value.len() as u16);
        self.set_len(len + 1);
    }

    pub fn remove(&mut self, index: usize) {
        debug_assert!(index < self.len());
        let len = self.len();

        // Shift slots left (value data becomes dead space)
        for i in index + 1..len {
            let (k, vo, vl) = self.read_slot(i);
            self.write_slot(i - 1, k, vo, vl);
        }

        self.set_len(len - 1);
    }

    pub fn split(&mut self) -> Self {
        self.compact();
        let len = self.len();
        let mid = len / 2;
        let mut new_page = Self::new();
        for i in mid..len {
            new_page.insert(self.key(i), self.value(i));
        }
        self.set_len(mid);
        self.compact();
        new_page
    }
}

impl<const N: usize> fmt::Debug for LeafPage<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for i in 0..self.len() {
            list.entry(&(self.key(i), self.value(i)));
        }
        list.finish()
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

    #[test]
    fn leaf_page_new_is_empty() {
        let page = LeafPage::<4096>::new();
        assert_eq!(page.len(), 0);
    }

    #[test]
    fn leaf_page_insert_single() {
        let mut page = LeafPage::<4096>::new();
        page.insert(10, b"hello");
        assert_eq!(page.len(), 1);
        assert_eq!(page.key(0), 10);
        assert_eq!(page.value(0), b"hello");
    }

    #[test]
    fn leaf_page_insert_maintains_sorted_order() {
        let mut page = LeafPage::<4096>::new();
        page.insert(30, b"ccc");
        page.insert(10, b"aaa");
        page.insert(20, b"bbb");

        assert_eq!(page.len(), 3);
        assert_eq!(page.key(0), 10);
        assert_eq!(page.key(1), 20);
        assert_eq!(page.key(2), 30);
        assert_eq!(page.value(0), b"aaa");
        assert_eq!(page.value(1), b"bbb");
        assert_eq!(page.value(2), b"ccc");
    }

    #[test]
    fn leaf_page_can_insert() {
        let page = LeafPage::<4096>::new();
        assert!(page.can_insert(100));
    }

    #[test]
    fn leaf_page_remove_middle() {
        let mut page = LeafPage::<4096>::new();
        page.insert(10, b"aaa");
        page.insert(20, b"bbb");
        page.insert(30, b"ccc");

        page.remove(1);
        assert_eq!(page.len(), 2);
        assert_eq!(page.key(0), 10);
        assert_eq!(page.key(1), 30);
        assert_eq!(page.value(0), b"aaa");
        assert_eq!(page.value(1), b"ccc");
    }

    #[test]
    fn leaf_page_remove_reclaims_space() {
        let mut page = LeafPage::<4096>::new();
        page.insert(10, b"aaa");
        page.insert(20, b"bbb");
        let free_before = page.free_space();
        page.remove(0);
        let free_after = page.free_space();
        assert!(free_after > free_before);
    }

    #[test]
    fn leaf_page_insert_after_remove_compacts() {
        let mut page = LeafPage::<128>::new();
        // Fill the page
        for i in 0..5 {
            page.insert(i, &[i as u8; 10]);
        }
        // Remove some entries to create dead space
        page.remove(1);
        page.remove(1);
        // Insert should succeed by compacting dead space
        assert!(page.can_insert(10));
        page.insert(100, &[0xAA; 10]);
        assert_eq!(page.value(page.len() - 1), &[0xAA; 10]);
    }

    #[test]
    fn leaf_page_split_divides_elements() {
        let mut page = LeafPage::<4096>::new();
        for i in 0u64..6 {
            page.insert(i * 10, &[i as u8; 5]);
        }

        let right = page.split();
        assert_eq!(page.len(), 3);
        assert_eq!(right.len(), 3);

        assert_eq!(page.key(0), 0);
        assert_eq!(page.key(2), 20);
        assert_eq!(page.value(0), &[0u8; 5]);
        assert_eq!(page.value(2), &[2u8; 5]);

        assert_eq!(right.key(0), 30);
        assert_eq!(right.key(2), 50);
        assert_eq!(right.value(0), &[3u8; 5]);
        assert_eq!(right.value(2), &[5u8; 5]);
    }
}
