use std::borrow::Cow;
use std::fmt;

use crate::{Key, NodePtr, Pager};

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageType {
    Leaf,
    Internal,
    IndexInternal,
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

#[repr(align(4096))]
#[derive(Clone)]
pub struct IndexInternalPage<const N: usize> {
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
    const OVERFLOW_FLAG: u16 = 0x8000;
    const OVERFLOW_META_SIZE: usize = 16;

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

    fn inline_len(value_len: u16) -> usize {
        (value_len & !Self::OVERFLOW_FLAG) as usize
    }

    pub fn is_overflow(&self, index: usize) -> bool {
        debug_assert!(index < self.len());
        let slot = Self::HEADER_SIZE + index * Self::SLOT_SIZE + std::mem::size_of::<Key>();
        let value_len = u16::from_le_bytes([self.page[slot + 2], self.page[slot + 3]]);
        value_len & Self::OVERFLOW_FLAG != 0
    }

    pub fn needs_overflow(value_len: usize) -> bool {
        value_len > (N - Self::HEADER_SIZE) / 2
    }

    fn write_overflow(pager: &mut Pager, data: &[u8]) -> u64 {
        let data_per_page = pager.page_size() - 8;
        let num_pages = (data.len() + data_per_page - 1) / data_per_page;
        assert!(num_pages > 0);

        let pages: Vec<u64> = (0..num_pages).map(|_| pager.next_page_num()).collect();

        for (i, &page_num) in pages.iter().enumerate() {
            let next_page = if i + 1 < pages.len() {
                pages[i + 1]
            } else {
                u64::MAX
            };
            let start = i * data_per_page;
            let end = std::cmp::min(start + data_per_page, data.len());
            let chunk = &data[start..end];

            let mut page_data = vec![0u8; pager.page_size()];
            page_data[0..8].copy_from_slice(&next_page.to_le_bytes());
            page_data[8..8 + chunk.len()].copy_from_slice(chunk);

            pager.write_raw_page(page_num, &page_data).unwrap();
        }

        pages[0]
    }

    pub fn read_overflow(pager: &mut Pager, start_page: u64, total_len: u64) -> Vec<u8> {
        let data_per_page = pager.page_size() - 8;
        let mut result = Vec::with_capacity(total_len as usize);
        let mut current_page = start_page;
        let mut remaining = total_len as usize;

        while remaining > 0 {
            let buffer = pager.read_raw_page(current_page).unwrap();
            let next_page = u64::from_le_bytes(buffer[..8].try_into().unwrap());
            let chunk_len = std::cmp::min(data_per_page, remaining);
            result.extend_from_slice(&buffer[8..8 + chunk_len]);
            remaining -= chunk_len;
            current_page = next_page;
        }

        result
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
        let raw_value_len = u16::from_le_bytes([self.page[slot + 2], self.page[slot + 3]]);
        let value_len = Self::inline_len(raw_value_len);
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

    pub fn get(&self, key: Key, pager: &mut Pager) -> Option<Cow<'_, [u8]>> {
        let idx = self.search(key)?;
        if self.key(idx) == key {
            if self.is_overflow(idx) {
                let meta = self.value(idx);
                let start_page = u64::from_le_bytes(meta[0..8].try_into().unwrap());
                let total_len = u64::from_le_bytes(meta[8..16].try_into().unwrap());
                Some(Cow::Owned(Self::read_overflow(pager, start_page, total_len)))
            } else {
                Some(Cow::Borrowed(self.value(idx)))
            }
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
        let used_value_space: usize = (0..self.len())
            .map(|i| Self::inline_len(self.read_slot(i).2))
            .sum();
        N - Self::HEADER_SIZE - self.len() * Self::SLOT_SIZE - used_value_space
    }

    pub fn can_insert(&self, value_len: usize) -> bool {
        let inline_size = if Self::needs_overflow(value_len) {
            Self::OVERFLOW_META_SIZE
        } else {
            value_len
        };
        self.free_space() >= Self::SLOT_SIZE + inline_size
    }

    /// Compact the value data region, eliminating dead gaps.
    fn compact(&mut self) {
        let len = self.len();
        let mut new_data_offset = N;
        for i in 0..len {
            let (k, vo, vl) = self.read_slot(i);
            let val_len = Self::inline_len(vl);
            new_data_offset -= val_len;
            // Copy value to new position (use copy_within to handle overlap)
            self.page
                .copy_within(vo as usize..vo as usize + val_len, new_data_offset);
            self.write_slot(i, k, new_data_offset as u16, vl);
        }
        self.set_data_offset(new_data_offset);
    }

    /// Internal insert that stores raw inline bytes with the given raw_value_len
    /// (which may include OVERFLOW_FLAG).
    fn insert_raw(&mut self, key: Key, inline_data: &[u8], raw_value_len: u16) {
        let inline_size = inline_data.len();

        // Compact if contiguous free space is insufficient
        if self.contiguous_free_space() < Self::SLOT_SIZE + inline_size {
            self.compact();
        }

        // Allocate value space from the end
        let new_data_offset = self.data_offset() - inline_size;
        self.page[new_data_offset..new_data_offset + inline_size].copy_from_slice(inline_data);
        self.set_data_offset(new_data_offset);

        let idx = self.search(key).unwrap_or(self.len());
        let len = self.len();

        // Shift slots right
        for i in (idx..len).rev() {
            let (k, vo, vl) = self.read_slot(i);
            self.write_slot(i + 1, k, vo, vl);
        }

        self.write_slot(idx, key, new_data_offset as u16, raw_value_len);
        self.set_len(len + 1);
    }

    pub fn insert(&mut self, key: Key, value: &[u8], pager: &mut Pager) {
        debug_assert!(self.can_insert(value.len()));

        if Self::needs_overflow(value.len()) {
            let start_page = Self::write_overflow(pager, value);
            let mut meta = [0u8; 16];
            meta[0..8].copy_from_slice(&start_page.to_le_bytes());
            meta[8..16].copy_from_slice(&(value.len() as u64).to_le_bytes());
            self.insert_raw(key, &meta, Self::OVERFLOW_META_SIZE as u16 | Self::OVERFLOW_FLAG);
        } else {
            self.insert_raw(key, value, value.len() as u16);
        }
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
            let (_, _, vl) = self.read_slot(i);
            new_page.insert_raw(self.key(i), self.value(i), vl);
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

impl<const N: usize> IndexInternalPage<N> {
    const LEN_OFFSET: usize = 4;
    const DATA_OFFSET: usize = 6;
    const HEADER_SIZE: usize = 8;
    const SLOT_SIZE: usize = 2 + 2 + 8; // key_offset + key_len + ptr
    const OVERFLOW_FLAG: u16 = 0x8000;
    const OVERFLOW_META_SIZE: usize = 16;

    pub fn new() -> Self {
        let mut page = Self { page: [0; N] };
        page.page[0..4].copy_from_slice(&(PageType::IndexInternal as u32).to_le_bytes());
        page.set_data_offset(N);
        page
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

    fn inline_len(key_len: u16) -> usize {
        (key_len & !Self::OVERFLOW_FLAG) as usize
    }

    pub fn is_overflow(&self, index: usize) -> bool {
        debug_assert!(index < self.len());
        let (_, key_len, _) = self.read_slot(index);
        key_len & Self::OVERFLOW_FLAG != 0
    }

    pub fn needs_overflow(key_len: usize) -> bool {
        key_len > (N - Self::HEADER_SIZE) / 2
    }

    fn write_overflow(pager: &mut Pager, data: &[u8]) -> u64 {
        let data_per_page = pager.page_size() - 8;
        let num_pages = (data.len() + data_per_page - 1) / data_per_page;
        assert!(num_pages > 0);

        let pages: Vec<u64> = (0..num_pages).map(|_| pager.next_page_num()).collect();

        for (i, &page_num) in pages.iter().enumerate() {
            let next_page = if i + 1 < pages.len() {
                pages[i + 1]
            } else {
                u64::MAX
            };
            let start = i * data_per_page;
            let end = std::cmp::min(start + data_per_page, data.len());
            let chunk = &data[start..end];

            let mut page_data = vec![0u8; pager.page_size()];
            page_data[0..8].copy_from_slice(&next_page.to_le_bytes());
            page_data[8..8 + chunk.len()].copy_from_slice(chunk);

            pager.write_raw_page(page_num, &page_data).unwrap();
        }

        pages[0]
    }

    pub fn read_overflow(pager: &mut Pager, start_page: u64, total_len: u64) -> Vec<u8> {
        let data_per_page = pager.page_size() - 8;
        let mut result = Vec::with_capacity(total_len as usize);
        let mut current_page = start_page;
        let mut remaining = total_len as usize;

        while remaining > 0 {
            let buffer = pager.read_raw_page(current_page).unwrap();
            let next_page = u64::from_le_bytes(buffer[..8].try_into().unwrap());
            let chunk_len = std::cmp::min(data_per_page, remaining);
            result.extend_from_slice(&buffer[8..8 + chunk_len]);
            remaining -= chunk_len;
            current_page = next_page;
        }

        result
    }

    pub fn key(&self, index: usize) -> &[u8] {
        debug_assert!(index < self.len());
        let (key_offset, key_len, _) = self.read_slot(index);
        let len = Self::inline_len(key_len);
        &self.page[key_offset as usize..key_offset as usize + len]
    }

    pub fn ptr(&self, index: usize) -> NodePtr {
        debug_assert!(index < self.len());
        let (_, _, ptr) = self.read_slot(index);
        ptr
    }

    fn write_slot(&mut self, index: usize, key_offset: u16, key_len: u16, ptr: NodePtr) {
        let offset = Self::HEADER_SIZE + index * Self::SLOT_SIZE;
        self.page[offset..offset + 2].copy_from_slice(&key_offset.to_le_bytes());
        self.page[offset + 2..offset + 4].copy_from_slice(&key_len.to_le_bytes());
        self.page[offset + 4..offset + 12].copy_from_slice(&ptr.to_le_bytes());
    }

    fn read_slot(&self, index: usize) -> (u16, u16, NodePtr) {
        let offset = Self::HEADER_SIZE + index * Self::SLOT_SIZE;
        let ko = u16::from_le_bytes([self.page[offset], self.page[offset + 1]]);
        let kl = u16::from_le_bytes([self.page[offset + 2], self.page[offset + 3]]);
        let ptr = NodePtr::from_le_bytes(
            self.page[offset + 4..offset + 12].try_into().unwrap(),
        );
        (ko, kl, ptr)
    }

    pub fn resolved_key<'a>(&'a self, index: usize, pager: &mut Pager) -> Cow<'a, [u8]> {
        if self.is_overflow(index) {
            let meta = self.key(index);
            let start_page = u64::from_le_bytes(meta[0..8].try_into().unwrap());
            let total_len = u64::from_le_bytes(meta[8..16].try_into().unwrap());
            Cow::Owned(Self::read_overflow(pager, start_page, total_len))
        } else {
            Cow::Borrowed(self.key(index))
        }
    }

    fn search(&self, target: &[u8], pager: &mut Pager) -> Option<usize> {
        let mut left = 0;
        let mut right = self.len();
        while left < right {
            let mid = (left + right) / 2;
            let mid_key = self.resolved_key(mid, pager);
            if mid_key.as_ref() < target {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if left == self.len() { None } else { Some(left) }
    }

    pub fn find_child(&self, target: &[u8], pager: &mut Pager) -> Option<NodePtr> {
        match self.search(target, pager) {
            Some(idx) => Some(self.ptr(idx)),
            None => None,
        }
    }

    fn contiguous_free_space(&self) -> usize {
        let slots_end = Self::HEADER_SIZE + self.len() * Self::SLOT_SIZE;
        self.data_offset().saturating_sub(slots_end)
    }

    fn free_space(&self) -> usize {
        let used_key_space: usize = (0..self.len())
            .map(|i| Self::inline_len(self.read_slot(i).1))
            .sum();
        N - Self::HEADER_SIZE - self.len() * Self::SLOT_SIZE - used_key_space
    }

    pub fn can_insert(&self, key_len: usize) -> bool {
        let inline_size = if Self::needs_overflow(key_len) {
            Self::OVERFLOW_META_SIZE
        } else {
            key_len
        };
        self.free_space() >= Self::SLOT_SIZE + inline_size
    }

    fn compact(&mut self) {
        let len = self.len();
        let mut new_data_offset = N;
        for i in 0..len {
            let (ko, kl, ptr) = self.read_slot(i);
            let key_len = Self::inline_len(kl);
            new_data_offset -= key_len;
            self.page
                .copy_within(ko as usize..ko as usize + key_len, new_data_offset);
            self.write_slot(i, new_data_offset as u16, kl, ptr);
        }
        self.set_data_offset(new_data_offset);
    }

    fn insert_raw(&mut self, key_bytes: &[u8], raw_key_len: u16, ptr: NodePtr, pager: &mut Pager) {
        let inline_size = key_bytes.len();

        if self.contiguous_free_space() < Self::SLOT_SIZE + inline_size {
            self.compact();
        }

        let new_data_offset = self.data_offset() - inline_size;
        self.page[new_data_offset..new_data_offset + inline_size].copy_from_slice(key_bytes);
        self.set_data_offset(new_data_offset);

        let idx = self.search(key_bytes, pager).unwrap_or(self.len());
        let len = self.len();

        for i in (idx..len).rev() {
            let (ko, kl, p) = self.read_slot(i);
            self.write_slot(i + 1, ko, kl, p);
        }

        self.write_slot(idx, new_data_offset as u16, raw_key_len, ptr);
        self.set_len(len + 1);
    }

    pub fn insert(&mut self, key: &[u8], ptr: NodePtr, pager: &mut Pager) {
        debug_assert!(self.can_insert(key.len()));

        if Self::needs_overflow(key.len()) {
            let start_page = Self::write_overflow(pager, key);
            let mut meta = [0u8; 16];
            meta[0..8].copy_from_slice(&start_page.to_le_bytes());
            meta[8..16].copy_from_slice(&(key.len() as u64).to_le_bytes());
            self.insert_raw(
                &meta,
                Self::OVERFLOW_META_SIZE as u16 | Self::OVERFLOW_FLAG,
                ptr,
                pager,
            );
        } else {
            self.insert_raw(key, key.len() as u16, ptr, pager);
        }
    }

    pub fn remove(&mut self, index: usize) {
        debug_assert!(index < self.len());
        let len = self.len();

        for i in index + 1..len {
            let (ko, kl, p) = self.read_slot(i);
            self.write_slot(i - 1, ko, kl, p);
        }

        self.set_len(len - 1);
    }

    pub fn split(&mut self, pager: &mut Pager) -> Self {
        self.compact();
        let len = self.len();
        let mid = len / 2;
        let mut new_page = Self::new();
        for i in mid..len {
            let (_, kl, ptr) = self.read_slot(i);
            new_page.insert_raw(self.key(i), kl, ptr, pager);
        }
        self.set_len(mid);
        self.compact();
        new_page
    }
}

impl<const N: usize> fmt::Debug for IndexInternalPage<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for i in 0..self.len() {
            list.entry(&(self.key(i), self.ptr(i)));
        }
        list.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pager(page_size: usize) -> Pager {
        let file = tempfile::tempfile().unwrap();
        Pager::new(file, page_size)
    }

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
        let mut pager = test_pager(4096);
        page.insert(10, b"hello", &mut pager);
        assert_eq!(page.len(), 1);
        assert_eq!(page.key(0), 10);
        assert_eq!(page.value(0), b"hello");
    }

    #[test]
    fn leaf_page_insert_maintains_sorted_order() {
        let mut page = LeafPage::<4096>::new();
        let mut pager = test_pager(4096);
        page.insert(30, b"ccc", &mut pager);
        page.insert(10, b"aaa", &mut pager);
        page.insert(20, b"bbb", &mut pager);

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
        let mut pager = test_pager(4096);
        page.insert(10, b"aaa", &mut pager);
        page.insert(20, b"bbb", &mut pager);
        page.insert(30, b"ccc", &mut pager);

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
        let mut pager = test_pager(4096);
        page.insert(10, b"aaa", &mut pager);
        page.insert(20, b"bbb", &mut pager);
        let free_before = page.free_space();
        page.remove(0);
        let free_after = page.free_space();
        assert!(free_after > free_before);
    }

    #[test]
    fn leaf_page_insert_after_remove_compacts() {
        let mut page = LeafPage::<128>::new();
        let mut pager = test_pager(4096);
        // Fill the page
        for i in 0..5 {
            page.insert(i, &[i as u8; 10], &mut pager);
        }
        // Remove some entries to create dead space
        page.remove(1);
        page.remove(1);
        // Insert should succeed by compacting dead space
        assert!(page.can_insert(10));
        page.insert(100, &[0xAA; 10], &mut pager);
        assert_eq!(page.value(page.len() - 1), &[0xAA; 10]);
    }

    #[test]
    fn leaf_page_split_divides_elements() {
        let mut page = LeafPage::<4096>::new();
        let mut pager = test_pager(4096);
        for i in 0u64..6 {
            page.insert(i * 10, &[i as u8; 5], &mut pager);
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

    #[test]
    fn leaf_page_overflow_insert_and_get() {
        let mut page = LeafPage::<4096>::new();
        let mut pager = test_pager(4096);

        // Value larger than (4096 - 8) / 2 = 2044 triggers overflow
        let big_value = vec![0xABu8; 2100];
        assert!(LeafPage::<4096>::needs_overflow(big_value.len()));

        page.insert(42, &big_value, &mut pager);
        assert_eq!(page.len(), 1);
        assert!(page.is_overflow(0));

        // get should read back the full value from overflow pages
        let retrieved = page.get(42, &mut pager).unwrap();
        assert_eq!(retrieved.as_ref(), &big_value[..]);
    }

    #[test]
    fn leaf_page_overflow_mixed_with_inline() {
        let mut page = LeafPage::<4096>::new();
        let mut pager = test_pager(4096);

        let small_value = b"hello";
        let big_value = vec![0xCDu8; 2100];

        page.insert(10, small_value, &mut pager);
        page.insert(20, &big_value, &mut pager);
        page.insert(30, b"world", &mut pager);

        assert!(!page.is_overflow(0));
        assert!(page.is_overflow(1));
        assert!(!page.is_overflow(2));

        assert_eq!(page.get(10, &mut pager).unwrap().as_ref(), b"hello");
        assert_eq!(page.get(20, &mut pager).unwrap().as_ref(), &big_value[..]);
        assert_eq!(page.get(30, &mut pager).unwrap().as_ref(), b"world");
    }

    #[test]
    fn leaf_page_overflow_split_preserves_data() {
        let mut page = LeafPage::<4096>::new();
        let mut pager = test_pager(4096);

        // Insert several small values and one overflow
        for i in 0u64..4 {
            page.insert(i * 10, &[i as u8; 5], &mut pager);
        }
        let big_value = vec![0xEFu8; 2100];
        page.insert(40, &big_value, &mut pager);
        page.insert(50, &[5u8; 5], &mut pager);

        let right = page.split();

        // Verify overflow entry is preserved in the right page
        // mid = 3, so right has keys 30, 40, 50
        assert!(right.is_overflow(1)); // key 40
        let retrieved = right.get(40, &mut pager).unwrap();
        assert_eq!(retrieved.as_ref(), &big_value[..]);
    }

    #[test]
    fn index_internal_page_new_is_empty() {
        let page = IndexInternalPage::<4096>::new();
        assert_eq!(page.len(), 0);
    }

    #[test]
    fn index_internal_page_insert_single() {
        let mut page = IndexInternalPage::<4096>::new();
        let mut pager = test_pager(4096);
        page.insert(b"hello", 100, &mut pager);
        assert_eq!(page.len(), 1);
        assert_eq!(page.key(0), b"hello");
        assert_eq!(page.ptr(0), 100);
    }

    #[test]
    fn index_internal_page_insert_maintains_sorted_order() {
        let mut page = IndexInternalPage::<4096>::new();
        let mut pager = test_pager(4096);
        page.insert(b"cherry", 300, &mut pager);
        page.insert(b"apple", 100, &mut pager);
        page.insert(b"banana", 200, &mut pager);

        assert_eq!(page.len(), 3);
        assert_eq!(page.key(0), b"apple");
        assert_eq!(page.key(1), b"banana");
        assert_eq!(page.key(2), b"cherry");
        assert_eq!(page.ptr(0), 100);
        assert_eq!(page.ptr(1), 200);
        assert_eq!(page.ptr(2), 300);
    }

    #[test]
    fn index_internal_page_find_child() {
        let mut page = IndexInternalPage::<4096>::new();
        let mut pager = test_pager(4096);
        page.insert(b"bbb", 100, &mut pager);
        page.insert(b"ddd", 200, &mut pager);
        page.insert(b"fff", 300, &mut pager);

        // Exact match
        assert_eq!(page.find_child(b"bbb", &mut pager), Some(100));
        assert_eq!(page.find_child(b"ddd", &mut pager), Some(200));
        assert_eq!(page.find_child(b"fff", &mut pager), Some(300));

        // Key before first -> routes to first
        assert_eq!(page.find_child(b"aaa", &mut pager), Some(100));

        // Key between entries -> routes to next >= entry
        assert_eq!(page.find_child(b"ccc", &mut pager), Some(200));
        assert_eq!(page.find_child(b"eee", &mut pager), Some(300));

        // Key after all -> None
        assert_eq!(page.find_child(b"zzz", &mut pager), None);
    }

    #[test]
    fn index_internal_page_remove() {
        let mut page = IndexInternalPage::<4096>::new();
        let mut pager = test_pager(4096);
        page.insert(b"aaa", 100, &mut pager);
        page.insert(b"bbb", 200, &mut pager);
        page.insert(b"ccc", 300, &mut pager);

        page.remove(1);
        assert_eq!(page.len(), 2);
        assert_eq!(page.key(0), b"aaa");
        assert_eq!(page.key(1), b"ccc");
        assert_eq!(page.ptr(0), 100);
        assert_eq!(page.ptr(1), 300);
    }

    #[test]
    fn index_internal_page_split() {
        let mut page = IndexInternalPage::<4096>::new();
        let mut pager = test_pager(4096);
        let keys = [b"aaa", b"bbb", b"ccc", b"ddd", b"eee", b"fff"];
        for (i, key) in keys.iter().enumerate() {
            page.insert(*key, (i * 100) as u64, &mut pager);
        }

        let right = page.split(&mut pager);
        assert_eq!(page.len(), 3);
        assert_eq!(right.len(), 3);

        assert_eq!(page.key(0), b"aaa");
        assert_eq!(page.key(2), b"ccc");

        assert_eq!(right.key(0), b"ddd");
        assert_eq!(right.key(2), b"fff");
    }

    #[test]
    fn index_internal_page_can_insert() {
        let page = IndexInternalPage::<4096>::new();
        assert!(page.can_insert(100));
    }

    #[test]
    fn index_internal_page_overflow_insert_and_find() {
        let mut page = IndexInternalPage::<4096>::new();
        let mut pager = test_pager(4096);

        let big_key = vec![0xABu8; 2100];
        assert!(IndexInternalPage::<4096>::needs_overflow(big_key.len()));

        page.insert(&big_key, 42, &mut pager);
        assert_eq!(page.len(), 1);
        assert!(page.is_overflow(0));

        let resolved = page.resolved_key(0, &mut pager);
        assert_eq!(resolved.as_ref(), &big_key[..]);

        assert_eq!(page.find_child(&big_key, &mut pager), Some(42));
    }
}
