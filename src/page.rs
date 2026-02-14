use std::fmt;
use std::{borrow::Cow, io};

use crate::{Key, NodePtr, Pager};

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageType {
    Leaf,
    Internal,
    IndexInternal,
    IndexLeaf,
}

pub const OVERFLOW_FLAG: u16 = 0x8000;
pub const OVERFLOW_META_SIZE: usize = 16;

#[repr(C)]
#[derive(Clone)]
pub struct AnyPage<const N: usize> {
    pub page: [u8; N],
}

#[repr(C)]
#[derive(Clone)]
pub struct InternalPage<const N: usize> {
    page: [u8; N],
}

#[repr(C)]
#[derive(Clone)]
pub struct LeafPage<const N: usize> {
    page: [u8; N],
}

#[repr(C)]
#[derive(Clone)]
pub struct IndexInternalPage<const N: usize> {
    page: [u8; N],
}

#[repr(C)]
#[derive(Clone)]
pub struct IndexLeafPage<const N: usize> {
    page: [u8; N],
}

impl<'a, const N: usize> TryFrom<&'a AnyPage<N>> for &'a IndexInternalPage<N> {
    type Error = &'static str;

    fn try_from(value: &'a AnyPage<N>) -> Result<Self, Self::Error> {
        if value.page_type() == PageType::IndexInternal {
            Ok(unsafe { &*(value as *const AnyPage<N> as *const IndexInternalPage<N>) })
        } else {
            Err("Page is not of type IndexInternal")
        }
    }
}

impl<'a, const N: usize> TryFrom<&'a mut AnyPage<N>> for &'a mut IndexInternalPage<N> {
    type Error = &'static str;

    fn try_from(value: &'a mut AnyPage<N>) -> Result<Self, Self::Error> {
        if value.page_type() == PageType::IndexInternal {
            Ok(unsafe { &mut *(value as *mut AnyPage<N> as *mut IndexInternalPage<N>) })
        } else {
            Err("Page is not of type IndexInternal")
        }
    }
}

impl<'a, const N: usize> TryFrom<&'a AnyPage<N>> for &'a IndexLeafPage<N> {
    type Error = &'static str;

    fn try_from(value: &'a AnyPage<N>) -> Result<Self, Self::Error> {
        if value.page_type() == PageType::IndexLeaf {
            Ok(unsafe { &*(value as *const AnyPage<N> as *const IndexLeafPage<N>) })
        } else {
            Err("Page is not of type IndexLeaf")
        }
    }
}

impl<'a, const N: usize> TryFrom<&'a mut AnyPage<N>> for &'a mut IndexLeafPage<N> {
    type Error = &'static str;

    fn try_from(value: &'a mut AnyPage<N>) -> Result<Self, Self::Error> {
        if value.page_type() == PageType::IndexLeaf {
            Ok(unsafe { &mut *(value as *mut AnyPage<N> as *mut IndexLeafPage<N>) })
        } else {
            Err("Page is not of type IndexLeaf")
        }
    }
}

impl<'a, const N: usize> TryFrom<&'a AnyPage<N>> for &'a InternalPage<N> {
    type Error = &'static str;

    fn try_from(value: &'a AnyPage<N>) -> Result<Self, Self::Error> {
        if value.page_type() == PageType::Internal {
            Ok(unsafe { &*(value as *const AnyPage<N> as *const InternalPage<N>) })
        } else {
            Err("Page is not of type Internal")
        }
    }
}

impl<'a, const N: usize> TryFrom<&'a mut AnyPage<N>> for &'a mut InternalPage<N> {
    type Error = &'static str;

    fn try_from(value: &'a mut AnyPage<N>) -> Result<Self, Self::Error> {
        if value.page_type() == PageType::Internal {
            Ok(unsafe { &mut *(value as *mut AnyPage<N> as *mut InternalPage<N>) })
        } else {
            Err("Page is not of type Internal")
        }
    }
}

impl<'a, const N: usize> TryFrom<&'a AnyPage<N>> for &'a LeafPage<N> {
    type Error = &'static str;

    fn try_from(value: &'a AnyPage<N>) -> Result<Self, Self::Error> {
        if value.page_type() == PageType::Leaf {
            Ok(unsafe { &*(value as *const AnyPage<N> as *const LeafPage<N>) })
        } else {
            Err("Page is not of type Leaf")
        }
    }
}

impl<'a, const N: usize> TryFrom<&'a mut AnyPage<N>> for &'a mut LeafPage<N> {
    type Error = &'static str;

    fn try_from(value: &'a mut AnyPage<N>) -> Result<Self, Self::Error> {
        if value.page_type() == PageType::Leaf {
            Ok(unsafe { &mut *(value as *mut AnyPage<N> as *mut LeafPage<N>) })
        } else {
            Err("Page is not of type Leaf")
        }
    }
}

impl<const N: usize> From<InternalPage<N>> for AnyPage<N> {
    fn from(internal: InternalPage<N>) -> Self {
        Self {
            page: internal.page,
        }
    }
}

impl<const N: usize> From<LeafPage<N>> for AnyPage<N> {
    fn from(leaf: LeafPage<N>) -> Self {
        Self { page: leaf.page }
    }
}

impl<const N: usize> From<IndexInternalPage<N>> for AnyPage<N> {
    fn from(index_internal: IndexInternalPage<N>) -> Self {
        Self {
            page: index_internal.page,
        }
    }
}

impl<const N: usize> From<IndexLeafPage<N>> for AnyPage<N> {
    fn from(index_leaf: IndexLeafPage<N>) -> Self {
        Self {
            page: index_leaf.page,
        }
    }
}

impl<const N: usize> TryInto<InternalPage<N>> for AnyPage<N> {
    type Error = &'static str;

    fn try_into(self) -> Result<InternalPage<N>, Self::Error> {
        if self.page_type() == PageType::Internal {
            Ok(InternalPage { page: self.page })
        } else {
            Err("Page is not of type Internal")
        }
    }
}

impl<const N: usize> TryInto<LeafPage<N>> for AnyPage<N> {
    type Error = &'static str;

    fn try_into(self) -> Result<LeafPage<N>, Self::Error> {
        if self.page_type() == PageType::Leaf {
            Ok(LeafPage { page: self.page })
        } else {
            Err("Page is not of type Leaf")
        }
    }
}

impl<const N: usize> TryInto<IndexInternalPage<N>> for AnyPage<N> {
    type Error = &'static str;

    fn try_into(self) -> Result<IndexInternalPage<N>, Self::Error> {
        if self.page_type() == PageType::IndexInternal {
            Ok(IndexInternalPage { page: self.page })
        } else {
            Err("Page is not of type IndexInternal")
        }
    }
}

impl<const N: usize> TryInto<IndexLeafPage<N>> for AnyPage<N> {
    type Error = &'static str;

    fn try_into(self) -> Result<IndexLeafPage<N>, Self::Error> {
        if self.page_type() == PageType::IndexLeaf {
            Ok(IndexLeafPage { page: self.page })
        } else {
            Err("Page is not of type IndexLeaf")
        }
    }
}

impl<const N: usize> AnyPage<N> {
    pub fn page_type(&self) -> PageType {
        let pt = u32::from_le_bytes(self.page[0..4].try_into().unwrap());
        match pt {
            x if x == PageType::Leaf as u32 => PageType::Leaf,
            x if x == PageType::Internal as u32 => PageType::Internal,
            x if x == PageType::IndexInternal as u32 => PageType::IndexInternal,
            x if x == PageType::IndexLeaf as u32 => PageType::IndexLeaf,
            _ => panic!("Invalid page type: {}", pt),
        }
    }
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
        (self.len() + 1) * Self::ELEMENT_SIZE + Self::HEADER_SIZE <= N
    }

    pub fn key(&self, index: usize) -> Key {
        debug_assert!(index < self.len());
        let offset = Self::HEADER_SIZE + index * Self::ELEMENT_SIZE;
        Key::from_le_bytes(
            self.page[offset..offset + std::mem::size_of::<Key>()]
                .try_into()
                .unwrap(),
        )
    }

    pub fn ptr(&self, index: usize) -> NodePtr {
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

    pub fn search_index(&self, key: Key) -> Option<usize> {
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

pub enum ValueToken<const N: usize> {
    Inline(Vec<u8>),
    Overflow(NodePtr, u64), // (start_page, total_len)
}

impl<const N: usize> ValueToken<N> {
    pub fn into_value(self, pager: &mut Pager<N>) -> io::Result<Vec<u8>> {
        match self {
            ValueToken::Inline(data) => Ok(data),
            ValueToken::Overflow(start_page, total_len) => {
                Ok(pager.read_overflow(start_page, total_len)?)
            }
        }
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

    fn inline_len(value_len: u16) -> usize {
        (value_len & !OVERFLOW_FLAG) as usize
    }

    pub fn is_overflow(&self, index: usize) -> bool {
        debug_assert!(index < self.len());
        let slot = Self::HEADER_SIZE + index * Self::SLOT_SIZE + std::mem::size_of::<Key>();
        let value_len = u16::from_le_bytes([self.page[slot + 2], self.page[slot + 3]]);
        value_len & OVERFLOW_FLAG != 0
    }

    pub fn needs_overflow(value_len: usize) -> bool {
        // Calculate the maximum number of slots that can fit in a page
        let max_slots = (N - Self::HEADER_SIZE) / Self::SLOT_SIZE;
        // After split, the worst case is inserting into a page with ceil(max_slots/2) slots
        let half_slots = (max_slots + 1) / 2;
        // Space used by slots in half-full page
        let used_slot_space = half_slots * Self::SLOT_SIZE;
        // Remaining space for value
        let available = N - Self::HEADER_SIZE - used_slot_space;
        Self::SLOT_SIZE + value_len > available
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

    pub fn value_token(&self, index: usize) -> ValueToken<N> {
        debug_assert!(index < self.len());

        if self.is_overflow(index) {
            let value = self.value(index);
            let start_page = Key::from_le_bytes(value[0..8].try_into().unwrap());
            let total_len = u64::from_le_bytes(value[8..16].try_into().unwrap());
            ValueToken::Overflow(start_page, total_len)
        } else {
            ValueToken::Inline(self.value(index).to_vec())
        }
    }

    fn write_slot(&mut self, index: usize, key: Key, value_offset: u16, value_len: u16) {
        let offset = Self::HEADER_SIZE + index * Self::SLOT_SIZE;
        self.page[offset..offset + std::mem::size_of::<Key>()].copy_from_slice(&key.to_le_bytes());
        self.page[offset + std::mem::size_of::<Key>()..offset + std::mem::size_of::<Key>() + 2]
            .copy_from_slice(&value_offset.to_le_bytes());
        self.page[offset + std::mem::size_of::<Key>() + 2..offset + Self::SLOT_SIZE]
            .copy_from_slice(&value_len.to_le_bytes());
    }

    pub fn read_slot(&self, index: usize) -> (Key, u16, u16) {
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

    /// Binary search for `key`. Returns `Ok(index)` if found, `Err(index)` for insertion point.
    pub fn search_key(&self, key: Key) -> Result<usize, usize> {
        let idx = self.search(key).unwrap_or(self.len());
        if idx < self.len() && self.key(idx) == key {
            Ok(idx)
        } else {
            Err(idx)
        }
    }

    pub fn get(&self, key: Key, pager: &mut Pager<N>) -> io::Result<Option<Cow<'_, [u8]>>> {
        if let Some(idx) = self.search(key) {
            if self.key(idx) == key {
                if self.is_overflow(idx) {
                    let meta = self.value(idx);
                    let start_page = u64::from_le_bytes(meta[0..8].try_into().unwrap());
                    let total_len = u64::from_le_bytes(meta[8..16].try_into().unwrap());
                    Ok(Some(Cow::Owned(
                        pager.read_overflow(start_page, total_len)?,
                    )))
                } else {
                    Ok(Some(Cow::Borrowed(self.value(idx))))
                }
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Contiguous free space between end of slot array and start of value data.
    fn contiguous_free_space(&self) -> usize {
        let slots_end = Self::HEADER_SIZE + self.len() * Self::SLOT_SIZE;
        self.data_offset().saturating_sub(slots_end)
    }

    /// Total free space including dead gaps left by removed values.
    pub fn free_space(&self) -> usize {
        let used_value_space: usize = (0..self.len())
            .map(|i| Self::inline_len(self.read_slot(i).2))
            .sum();
        N - Self::HEADER_SIZE - self.len() * Self::SLOT_SIZE - used_value_space
    }

    pub fn can_insert(&self, value_len: usize) -> bool {
        let inline_size = if Self::needs_overflow(value_len) {
            OVERFLOW_META_SIZE
        } else {
            value_len
        };
        self.free_space() >= Self::SLOT_SIZE + inline_size
    }

    /// Compact the value data region, eliminating dead gaps.
    fn compact(&mut self) {
        let len = self.len();
        // Snapshot old page bytes so that moving one value cannot overwrite
        // another value's source data (overlapping regions across entries).
        let old = self.page;
        let mut new_data_offset = N;
        for i in 0..len {
            let (k, vo, vl) = self.read_slot(i);
            let val_len = Self::inline_len(vl);
            new_data_offset -= val_len;
            self.page[new_data_offset..new_data_offset + val_len]
                .copy_from_slice(&old[vo as usize..vo as usize + val_len]);
            self.write_slot(i, k, new_data_offset as u16, vl);
        }
        self.set_data_offset(new_data_offset);
    }

    /// Insert that stores raw inline bytes with the given raw_value_len
    /// (which may include OVERFLOW_FLAG).
    pub fn insert_raw(&mut self, key: Key, inline_data: &[u8], raw_value_len: u16) {
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

    pub fn insert(&mut self, key: Key, value: &[u8], pager: &mut Pager<N>) -> io::Result<()> {
        debug_assert!(self.can_insert(value.len()));

        if Self::needs_overflow(value.len()) {
            let start_page = pager.write_overflow(value)?;
            let mut meta = [0u8; 16];
            meta[0..8].copy_from_slice(&start_page.to_le_bytes());
            meta[8..16].copy_from_slice(&(value.len() as u64).to_le_bytes());
            self.insert_raw(key, &meta, OVERFLOW_META_SIZE as u16 | OVERFLOW_FLAG);
        } else {
            self.insert_raw(key, value, value.len() as u16);
        }

        Ok(())
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
        let len = self.len();
        let mid = len / 2;
        let mut new_page = Self::new();
        for i in mid..len {
            let (_, _, vl) = self.read_slot(i);
            new_page.insert_raw(self.key(i), self.value(i), vl);
        }
        self.set_len(mid);
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

/// Shared implementation for bytes-keyed pages (IndexInternalPage / IndexLeafPage).
///
/// Layout: `[page_type: u32][len: u16][data_offset: u16][slots...][free...][key_data...]`
/// Slot: `[key_offset: u16][key_len: u16][value: u64]` (12 bytes)
macro_rules! impl_bytes_keyed_page {
    ($name:ident, $page_type:expr) => {
        impl<const N: usize> $name<N> {
            const LEN_OFFSET: usize = 4;
            const DATA_OFFSET: usize = 6;
            const HEADER_SIZE: usize = 8;
            const SLOT_SIZE: usize = 2 + 2 + 8; // key_offset + key_len + value(u64)

            pub fn new() -> Self {
                let mut page = Self { page: [0; N] };
                page.page[0..4].copy_from_slice(&($page_type as u32).to_le_bytes());
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
                (key_len & !OVERFLOW_FLAG) as usize
            }

            pub fn is_overflow(&self, index: usize) -> bool {
                debug_assert!(index < self.len());
                let (_, key_len, _) = self.read_slot(index);
                key_len & OVERFLOW_FLAG != 0
            }

            pub fn needs_overflow(key_len: usize) -> bool {
                // Calculate the maximum number of slots that can fit in a page
                let max_slots = (N - Self::HEADER_SIZE) / Self::SLOT_SIZE;
                // After split, the worst case is inserting into a page with ceil(max_slots/2) slots
                let half_slots = (max_slots + 1) / 2;
                // Space used by slots in half-full page
                let used_slot_space = half_slots * Self::SLOT_SIZE;
                // Remaining space for value
                let available = N - Self::HEADER_SIZE - used_slot_space;
                Self::SLOT_SIZE + key_len > available
            }

            pub fn key(&self, index: usize) -> &[u8] {
                debug_assert!(index < self.len());
                let (key_offset, key_len, _) = self.read_slot(index);
                let len = Self::inline_len(key_len);
                &self.page[key_offset as usize..key_offset as usize + len]
            }

            pub fn slot_value(&self, index: usize) -> u64 {
                debug_assert!(index < self.len());
                let (_, _, v) = self.read_slot(index);
                v
            }

            fn write_slot(&mut self, index: usize, key_offset: u16, key_len: u16, value: u64) {
                let offset = Self::HEADER_SIZE + index * Self::SLOT_SIZE;
                self.page[offset..offset + 2].copy_from_slice(&key_offset.to_le_bytes());
                self.page[offset + 2..offset + 4].copy_from_slice(&key_len.to_le_bytes());
                self.page[offset + 4..offset + 12].copy_from_slice(&value.to_le_bytes());
            }

            pub fn read_slot(&self, index: usize) -> (u16, u16, u64) {
                let offset = Self::HEADER_SIZE + index * Self::SLOT_SIZE;
                let ko = u16::from_le_bytes([self.page[offset], self.page[offset + 1]]);
                let kl = u16::from_le_bytes([self.page[offset + 2], self.page[offset + 3]]);
                let v = u64::from_le_bytes(self.page[offset + 4..offset + 12].try_into().unwrap());
                (ko, kl, v)
            }

            pub fn resolved_key<'a>(
                &'a self,
                index: usize,
                pager: &mut Pager<N>,
            ) -> io::Result<Cow<'a, [u8]>> {
                if self.is_overflow(index) {
                    let meta = self.key(index);
                    let start_page = u64::from_le_bytes(meta[0..8].try_into().unwrap());
                    let total_len = u64::from_le_bytes(meta[8..16].try_into().unwrap());
                    Ok(Cow::Owned(pager.read_overflow(start_page, total_len)?))
                } else {
                    Ok(Cow::Borrowed(self.key(index)))
                }
            }

            pub fn search(&self, target: &[u8], pager: &mut Pager<N>) -> io::Result<Option<usize>> {
                let mut left = 0;
                let mut right = self.len();
                while left < right {
                    let mid = (left + right) / 2;
                    let mid_key = self.resolved_key(mid, pager)?;
                    if mid_key.as_ref() < target {
                        left = mid + 1;
                    } else {
                        right = mid;
                    }
                }
                if left == self.len() {
                    Ok(None)
                } else {
                    Ok(Some(left))
                }
            }

            fn contiguous_free_space(&self) -> usize {
                let slots_end = Self::HEADER_SIZE + self.len() * Self::SLOT_SIZE;
                self.data_offset().saturating_sub(slots_end)
            }

            pub fn free_space(&self) -> usize {
                let used_key_space: usize = (0..self.len())
                    .map(|i| Self::inline_len(self.read_slot(i).1))
                    .sum();
                N - Self::HEADER_SIZE - self.len() * Self::SLOT_SIZE - used_key_space
            }

            pub fn can_insert(&self, key_len: usize) -> bool {
                let inline_size = if Self::needs_overflow(key_len) {
                    OVERFLOW_META_SIZE
                } else {
                    key_len
                };
                self.free_space() >= Self::SLOT_SIZE + inline_size
            }

            fn compact(&mut self) {
                let len = self.len();
                let old = self.page;
                let mut new_data_offset = N;
                for i in 0..len {
                    let (ko, kl, v) = self.read_slot(i);
                    let key_len = Self::inline_len(kl);
                    new_data_offset -= key_len;
                    self.page[new_data_offset..new_data_offset + key_len]
                        .copy_from_slice(&old[ko as usize..ko as usize + key_len]);
                    self.write_slot(i, new_data_offset as u16, kl, v);
                }
                self.set_data_offset(new_data_offset);
            }

            pub fn insert_raw_at(
                &mut self,
                idx: usize,
                key_bytes: &[u8],
                raw_key_len: u16,
                value: u64,
            ) {
                let inline_size = key_bytes.len();

                if self.contiguous_free_space() < Self::SLOT_SIZE + inline_size {
                    self.compact();
                }

                let new_data_offset = self.data_offset() - inline_size;
                self.page[new_data_offset..new_data_offset + inline_size]
                    .copy_from_slice(key_bytes);
                self.set_data_offset(new_data_offset);

                let len = self.len();

                for i in (idx..len).rev() {
                    let (ko, kl, v) = self.read_slot(i);
                    self.write_slot(i + 1, ko, kl, v);
                }

                self.write_slot(idx, new_data_offset as u16, raw_key_len, value);
                self.set_len(len + 1);
            }

            fn insert_raw(
                &mut self,
                key_bytes: &[u8],
                raw_key_len: u16,
                value: u64,
                pager: &mut Pager<N>,
            ) -> io::Result<()> {
                let idx = self.search(key_bytes, pager)?.unwrap_or(self.len());
                self.insert_raw_at(idx, key_bytes, raw_key_len, value);
                Ok(())
            }

            pub fn insert(
                &mut self,
                key: &[u8],
                value: u64,
                pager: &mut Pager<N>,
            ) -> io::Result<()> {
                debug_assert!(self.can_insert(key.len()));

                if Self::needs_overflow(key.len()) {
                    let start_page = pager.write_overflow(key)?;
                    let mut meta = [0u8; 16];
                    meta[0..8].copy_from_slice(&start_page.to_le_bytes());
                    meta[8..16].copy_from_slice(&(key.len() as u64).to_le_bytes());
                    self.insert_raw(
                        &meta,
                        OVERFLOW_META_SIZE as u16 | OVERFLOW_FLAG,
                        value,
                        pager,
                    )?;
                } else {
                    self.insert_raw(key, key.len() as u16, value, pager)?;
                }
                Ok(())
            }

            pub fn remove(&mut self, index: usize) {
                debug_assert!(index < self.len());
                let len = self.len();

                for i in index + 1..len {
                    let (ko, kl, v) = self.read_slot(i);
                    self.write_slot(i - 1, ko, kl, v);
                }

                self.set_len(len - 1);
            }

            pub fn split(&mut self) -> Self {
                let len = self.len();
                let mid = len / 2;
                let mut new_page = Self::new();
                for i in mid..len {
                    let (_, kl, v) = self.read_slot(i);
                    let end = new_page.len();
                    new_page.insert_raw_at(end, self.key(i), kl, v);
                }
                self.set_len(mid);
                new_page
            }
        }

        impl<const N: usize> fmt::Debug for $name<N> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut list = f.debug_list();
                for i in 0..self.len() {
                    list.entry(&(self.key(i), self.slot_value(i)));
                }
                list.finish()
            }
        }
    };
}

impl_bytes_keyed_page!(IndexInternalPage, PageType::IndexInternal);
impl_bytes_keyed_page!(IndexLeafPage, PageType::IndexLeaf);

// Type-specific accessors for IndexInternalPage
impl<const N: usize> IndexInternalPage<N> {
    pub fn ptr(&self, index: usize) -> NodePtr {
        self.slot_value(index)
    }

    pub fn find_child(&self, target: &[u8], pager: &mut Pager<N>) -> io::Result<Option<NodePtr>> {
        match self.search(target, pager)? {
            Some(idx) => Ok(Some(self.ptr(idx))),
            None => Ok(None),
        }
    }
}

// Type-specific accessors for IndexLeafPage
impl<const N: usize> IndexLeafPage<N> {
    pub fn value(&self, index: usize) -> Key {
        self.slot_value(index)
    }

    pub fn get(&self, target: &[u8], pager: &mut Pager<N>) -> io::Result<Option<Key>> {
        if let Some(idx) = self.search(target, pager)? {
            let resolved = self.resolved_key(idx, pager)?;
            if resolved.as_ref() == target {
                Ok(Some(self.value(idx)))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Binary search using composite `(bytes_key, key)` ordering.
    /// Returns the index of the first entry >= `(target, target_key)`.
    /// Returns `None` if all entries are less than the target.
    pub fn search_entry(
        &self,
        target: &[u8],
        target_key: Key,
        pager: &mut Pager<N>,
    ) -> io::Result<Option<usize>> {
        let mut left = 0;
        let mut right = self.len();
        while left < right {
            let mid = (left + right) / 2;
            let mid_key_bytes = self.resolved_key(mid, pager)?;
            let mid_value = self.value(mid);
            match mid_key_bytes
                .as_ref()
                .cmp(target)
                .then(mid_value.cmp(&target_key))
            {
                std::cmp::Ordering::Less => left = mid + 1,
                _ => right = mid,
            }
        }
        if left == self.len() {
            Ok(None)
        } else {
            Ok(Some(left))
        }
    }

    /// Find the exact entry matching `(target, target_key)`.
    /// Returns `Some(index)` if found, `None` otherwise.
    pub fn find_entry(
        &self,
        target: &[u8],
        target_key: Key,
        pager: &mut Pager<N>,
    ) -> io::Result<Option<usize>> {
        if let Some(idx) = self.search_entry(target, target_key, pager)? {
            let resolved = self.resolved_key(idx, pager)?;
            if resolved.as_ref() == target && self.value(idx) == target_key {
                Ok(Some(idx))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Insert using composite `(bytes_key, key)` ordering,
    /// allowing multiple entries with the same byte key but different Key values.
    pub fn insert_entry(
        &mut self,
        key: &[u8],
        entry_key: Key,
        pager: &mut Pager<N>,
    ) -> io::Result<()> {
        debug_assert!(self.can_insert(key.len()));

        if Self::needs_overflow(key.len()) {
            let start_page = pager.write_overflow(key)?;
            let mut meta = [0u8; 16];
            meta[0..8].copy_from_slice(&start_page.to_le_bytes());
            meta[8..16].copy_from_slice(&(key.len() as u64).to_le_bytes());
            let raw_key_len = OVERFLOW_META_SIZE as u16 | OVERFLOW_FLAG;
            let idx = self
                .search_entry(&meta, entry_key, pager)?
                .unwrap_or(self.len());
            self.insert_raw_at(idx, &meta, raw_key_len, entry_key);
        } else {
            let idx = self
                .search_entry(key, entry_key, pager)?
                .unwrap_or(self.len());
            self.insert_raw_at(idx, key, key.len() as u16, entry_key);
        }

        Ok(())
    }

    /// Remove the entry matching `(target, target_key)`.
    /// Returns `true` if an entry was removed.
    pub fn remove_entry(
        &mut self,
        target: &[u8],
        target_key: Key,
        pager: &mut Pager<N>,
    ) -> io::Result<bool> {
        if let Some(idx) = self.find_entry(target, target_key, pager)? {
            self.remove(idx);
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pager<const N: usize>() -> Pager<N> {
        let file = tempfile::tempfile().unwrap();
        Pager::new(file)
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
        let mut pager = test_pager();
        page.insert(10, b"hello", &mut pager).unwrap();
        assert_eq!(page.len(), 1);
        assert_eq!(page.key(0), 10);
        assert_eq!(page.value(0), b"hello");
    }

    #[test]
    fn leaf_page_insert_maintains_sorted_order() {
        let mut page = LeafPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(30, b"ccc", &mut pager).unwrap();
        page.insert(10, b"aaa", &mut pager).unwrap();
        page.insert(20, b"bbb", &mut pager).unwrap();

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
        let mut pager = test_pager();
        page.insert(10, b"aaa", &mut pager).unwrap();
        page.insert(20, b"bbb", &mut pager).unwrap();
        page.insert(30, b"ccc", &mut pager).unwrap();

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
        let mut pager = test_pager();
        page.insert(10, b"aaa", &mut pager).unwrap();
        page.insert(20, b"bbb", &mut pager).unwrap();
        let free_before = page.free_space();
        page.remove(0);
        let free_after = page.free_space();
        assert!(free_after > free_before);
    }

    #[test]
    fn leaf_page_insert_after_remove_compacts() {
        let mut page = LeafPage::<128>::new();
        let mut pager = test_pager();
        // Fill the page
        for i in 0..5 {
            page.insert(i, &[i as u8; 10], &mut pager).unwrap();
        }
        // Remove some entries to create dead space
        page.remove(1);
        page.remove(1);
        // Insert should succeed by compacting dead space
        assert!(page.can_insert(10));
        page.insert(100, &[0xAA; 10], &mut pager).unwrap();
        assert_eq!(page.value(page.len() - 1), &[0xAA; 10]);
    }

    #[test]
    fn leaf_page_split_divides_elements() {
        let mut page = LeafPage::<4096>::new();
        let mut pager = test_pager();
        for i in 0u64..6 {
            page.insert(i * 10, &[i as u8; 5], &mut pager).unwrap();
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
        let mut pager = test_pager();

        // Value larger than (4096 - 8) / 2 = 2044 triggers overflow
        let big_value = vec![0xABu8; 2100];
        assert!(LeafPage::<4096>::needs_overflow(big_value.len()));

        page.insert(42, &big_value, &mut pager).unwrap();
        assert_eq!(page.len(), 1);
        assert!(page.is_overflow(0));

        // get should read back the full value from overflow pages
        let retrieved = page.get(42, &mut pager).unwrap().unwrap();
        assert_eq!(retrieved.as_ref(), &big_value[..]);
    }

    #[test]
    fn leaf_page_overflow_mixed_with_inline() {
        let mut page = LeafPage::<4096>::new();
        let mut pager = test_pager();

        let small_value = b"hello";
        let big_value = vec![0xCDu8; 2100];

        page.insert(10, small_value, &mut pager).unwrap();
        page.insert(20, &big_value, &mut pager).unwrap();
        page.insert(30, b"world", &mut pager).unwrap();

        assert!(!page.is_overflow(0));
        assert!(page.is_overflow(1));
        assert!(!page.is_overflow(2));

        assert_eq!(
            page.get(10, &mut pager).unwrap().unwrap().as_ref(),
            b"hello"
        );
        assert_eq!(
            page.get(20, &mut pager).unwrap().unwrap().as_ref(),
            &big_value[..]
        );
        assert_eq!(
            page.get(30, &mut pager).unwrap().unwrap().as_ref(),
            b"world"
        );
    }

    #[test]
    fn leaf_page_overflow_split_preserves_data() {
        let mut page = LeafPage::<4096>::new();
        let mut pager = test_pager();

        // Insert several small values and one overflow
        for i in 0u64..4 {
            page.insert(i * 10, &[i as u8; 5], &mut pager).unwrap();
        }
        let big_value = vec![0xEFu8; 2100];
        page.insert(40, &big_value, &mut pager).unwrap();
        page.insert(50, &[5u8; 5], &mut pager).unwrap();

        let right = page.split();

        // Verify overflow entry is preserved in the right page
        // mid = 3, so right has keys 30, 40, 50
        assert!(right.is_overflow(1)); // key 40
        let retrieved = right.get(40, &mut pager).unwrap().unwrap();
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
        let mut pager = test_pager();
        page.insert(b"hello", 100, &mut pager).unwrap();
        assert_eq!(page.len(), 1);
        assert_eq!(page.key(0), b"hello");
        assert_eq!(page.ptr(0), 100);
    }

    #[test]
    fn index_internal_page_insert_maintains_sorted_order() {
        let mut page = IndexInternalPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(b"cherry", 300, &mut pager).unwrap();
        page.insert(b"apple", 100, &mut pager).unwrap();
        page.insert(b"banana", 200, &mut pager).unwrap();

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
        let mut pager = test_pager();
        page.insert(b"bbb", 100, &mut pager).unwrap();
        page.insert(b"ddd", 200, &mut pager).unwrap();
        page.insert(b"fff", 300, &mut pager).unwrap();

        // Exact match
        assert_eq!(page.find_child(b"bbb", &mut pager).unwrap(), Some(100));
        assert_eq!(page.find_child(b"ddd", &mut pager).unwrap(), Some(200));
        assert_eq!(page.find_child(b"fff", &mut pager).unwrap(), Some(300));

        // Key before first -> routes to first
        assert_eq!(page.find_child(b"aaa", &mut pager).unwrap(), Some(100));

        // Key between entries -> routes to next >= entry
        assert_eq!(page.find_child(b"ccc", &mut pager).unwrap(), Some(200));
        assert_eq!(page.find_child(b"eee", &mut pager).unwrap(), Some(300));

        // Key after all -> None
        assert_eq!(page.find_child(b"zzz", &mut pager).unwrap(), None);
    }

    #[test]
    fn index_internal_page_remove() {
        let mut page = IndexInternalPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(b"aaa", 100, &mut pager).unwrap();
        page.insert(b"bbb", 200, &mut pager).unwrap();
        page.insert(b"ccc", 300, &mut pager).unwrap();

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
        let mut pager = test_pager();
        let keys = [b"aaa", b"bbb", b"ccc", b"ddd", b"eee", b"fff"];
        for (i, key) in keys.iter().enumerate() {
            page.insert(*key, (i * 100) as u64, &mut pager).unwrap();
        }

        let right = page.split();
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
        let mut pager = test_pager();

        let big_key = vec![0xABu8; 2100];
        assert!(IndexInternalPage::<4096>::needs_overflow(big_key.len()));

        page.insert(&big_key, 42, &mut pager).unwrap();
        assert_eq!(page.len(), 1);
        assert!(page.is_overflow(0));

        let resolved = page.resolved_key(0, &mut pager).unwrap();
        assert_eq!(resolved.as_ref(), &big_key[..]);

        assert_eq!(page.find_child(&big_key, &mut pager).unwrap(), Some(42));
    }

    #[test]
    fn index_leaf_page_new_is_empty() {
        let page = IndexLeafPage::<4096>::new();
        assert_eq!(page.len(), 0);
    }

    #[test]
    fn index_leaf_page_insert_single() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(b"hello", 100, &mut pager).unwrap();
        assert_eq!(page.len(), 1);
        assert_eq!(page.key(0), b"hello");
        assert_eq!(page.value(0), 100);
    }

    #[test]
    fn index_leaf_page_insert_maintains_sorted_order() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(b"cherry", 300, &mut pager).unwrap();
        page.insert(b"apple", 100, &mut pager).unwrap();
        page.insert(b"banana", 200, &mut pager).unwrap();

        assert_eq!(page.len(), 3);
        assert_eq!(page.key(0), b"apple");
        assert_eq!(page.key(1), b"banana");
        assert_eq!(page.key(2), b"cherry");
        assert_eq!(page.value(0), 100);
        assert_eq!(page.value(1), 200);
        assert_eq!(page.value(2), 300);
    }

    #[test]
    fn index_leaf_page_get() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(b"aaa", 10, &mut pager).unwrap();
        page.insert(b"bbb", 20, &mut pager).unwrap();
        page.insert(b"ccc", 30, &mut pager).unwrap();

        assert_eq!(page.get(b"aaa", &mut pager).unwrap(), Some(10));
        assert_eq!(page.get(b"bbb", &mut pager).unwrap(), Some(20));
        assert_eq!(page.get(b"ccc", &mut pager).unwrap(), Some(30));
        assert_eq!(page.get(b"ddd", &mut pager).unwrap(), None);
        assert_eq!(page.get(b"ab", &mut pager).unwrap(), None);
    }

    #[test]
    fn index_leaf_page_remove() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(b"aaa", 100, &mut pager).unwrap();
        page.insert(b"bbb", 200, &mut pager).unwrap();
        page.insert(b"ccc", 300, &mut pager).unwrap();

        page.remove(1);
        assert_eq!(page.len(), 2);
        assert_eq!(page.key(0), b"aaa");
        assert_eq!(page.key(1), b"ccc");
        assert_eq!(page.value(0), 100);
        assert_eq!(page.value(1), 300);
    }

    #[test]
    fn index_leaf_page_split() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();
        let keys = [b"aaa", b"bbb", b"ccc", b"ddd", b"eee", b"fff"];
        for (i, key) in keys.iter().enumerate() {
            page.insert(*key, (i * 100) as u64, &mut pager).unwrap();
        }

        let right = page.split();
        assert_eq!(page.len(), 3);
        assert_eq!(right.len(), 3);

        assert_eq!(page.key(0), b"aaa");
        assert_eq!(page.key(2), b"ccc");

        assert_eq!(right.key(0), b"ddd");
        assert_eq!(right.key(2), b"fff");
    }

    #[test]
    fn index_leaf_page_can_insert() {
        let page = IndexLeafPage::<4096>::new();
        assert!(page.can_insert(100));
    }

    #[test]
    fn index_leaf_page_overflow_insert_and_get() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();

        let big_key = vec![0xABu8; 2100];
        assert!(IndexLeafPage::<4096>::needs_overflow(big_key.len()));

        page.insert(&big_key, 42, &mut pager).unwrap();
        assert_eq!(page.len(), 1);
        assert!(page.is_overflow(0));

        let resolved = page.resolved_key(0, &mut pager).unwrap();
        assert_eq!(resolved.as_ref(), &big_key[..]);

        assert_eq!(page.get(&big_key, &mut pager).unwrap(), Some(42));
    }

    #[test]
    fn index_leaf_page_insert_entry_duplicate_keys() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();

        // Insert multiple entries with the same byte key but different Key values
        page.insert_entry(b"aaa", 30, &mut pager).unwrap();
        page.insert_entry(b"aaa", 10, &mut pager).unwrap();
        page.insert_entry(b"aaa", 20, &mut pager).unwrap();

        assert_eq!(page.len(), 3);
        // Should be sorted by (bytes, Key)
        assert_eq!(page.value(0), 10);
        assert_eq!(page.value(1), 20);
        assert_eq!(page.value(2), 30);
    }

    #[test]
    fn index_leaf_page_insert_entry_mixed_keys() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();

        page.insert_entry(b"bbb", 2, &mut pager).unwrap();
        page.insert_entry(b"aaa", 3, &mut pager).unwrap();
        page.insert_entry(b"bbb", 1, &mut pager).unwrap();
        page.insert_entry(b"aaa", 1, &mut pager).unwrap();

        assert_eq!(page.len(), 4);
        // (aaa,1), (aaa,3), (bbb,1), (bbb,2)
        assert_eq!(page.key(0), b"aaa");
        assert_eq!(page.value(0), 1);
        assert_eq!(page.key(1), b"aaa");
        assert_eq!(page.value(1), 3);
        assert_eq!(page.key(2), b"bbb");
        assert_eq!(page.value(2), 1);
        assert_eq!(page.key(3), b"bbb");
        assert_eq!(page.value(3), 2);
    }

    #[test]
    fn index_leaf_page_find_entry() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();

        page.insert_entry(b"aaa", 10, &mut pager).unwrap();
        page.insert_entry(b"aaa", 20, &mut pager).unwrap();
        page.insert_entry(b"bbb", 10, &mut pager).unwrap();

        assert_eq!(page.find_entry(b"aaa", 10, &mut pager).unwrap(), Some(0));
        assert_eq!(page.find_entry(b"aaa", 20, &mut pager).unwrap(), Some(1));
        assert_eq!(page.find_entry(b"bbb", 10, &mut pager).unwrap(), Some(2));
        assert_eq!(page.find_entry(b"aaa", 99, &mut pager).unwrap(), None);
        assert_eq!(page.find_entry(b"ccc", 10, &mut pager).unwrap(), None);
    }

    #[test]
    fn index_leaf_page_remove_entry() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();

        page.insert_entry(b"aaa", 10, &mut pager).unwrap();
        page.insert_entry(b"aaa", 20, &mut pager).unwrap();
        page.insert_entry(b"aaa", 30, &mut pager).unwrap();

        assert!(page.remove_entry(b"aaa", 20, &mut pager).unwrap());
        assert_eq!(page.len(), 2);
        assert_eq!(page.value(0), 10);
        assert_eq!(page.value(1), 30);

        assert!(!page.remove_entry(b"aaa", 99, &mut pager).unwrap());
        assert_eq!(page.len(), 2);
    }

    #[test]
    fn index_leaf_page_search_entry() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();

        page.insert_entry(b"aaa", 10, &mut pager).unwrap();
        page.insert_entry(b"aaa", 20, &mut pager).unwrap();
        page.insert_entry(b"bbb", 5, &mut pager).unwrap();

        // First entry >= (aaa, 10) is at index 0
        assert_eq!(page.search_entry(b"aaa", 10, &mut pager).unwrap(), Some(0));
        // First entry >= (aaa, 15) is at index 1 (aaa,20)
        assert_eq!(page.search_entry(b"aaa", 15, &mut pager).unwrap(), Some(1));
        // First entry >= (aaa, 21) is at index 2 (bbb,5)
        assert_eq!(page.search_entry(b"aaa", 21, &mut pager).unwrap(), Some(2));
        // All entries are less than (ccc, 0)
        assert_eq!(page.search_entry(b"ccc", 0, &mut pager).unwrap(), None);
    }

    // ── Edge case tests ──────────────────────────────────────────────

    #[test]
    fn internal_page_insert_duplicate_key_appends() {
        let mut page = InternalPage::<4096>::new();
        page.insert(10, 100);
        page.insert(10, 200);
        assert_eq!(page.len(), 2);
        // Both at the same key position
        assert_eq!(page.key(0), 10);
        assert_eq!(page.key(1), 10);
    }

    #[test]
    fn internal_page_remove_first_and_last() {
        let mut page = InternalPage::<4096>::new();
        page.insert(10, 100);
        page.insert(20, 200);
        page.insert(30, 300);

        page.remove(0);
        assert_eq!(page.len(), 2);
        assert_eq!(page.key(0), 20);

        page.remove(page.len() - 1);
        assert_eq!(page.len(), 1);
        assert_eq!(page.key(0), 20);
    }

    #[test]
    fn internal_page_remove_only_element() {
        let mut page = InternalPage::<4096>::new();
        page.insert(10, 100);
        page.remove(0);
        assert_eq!(page.len(), 0);
    }

    #[test]
    fn internal_page_split_odd_count() {
        let mut page = InternalPage::<4096>::new();
        for i in 0..5 {
            page.insert(i * 10, i * 100);
        }
        let right = page.split();
        // mid = 2, left gets 2 elements, right gets 3
        assert_eq!(page.len(), 2);
        assert_eq!(right.len(), 3);
    }

    #[test]
    fn internal_page_split_single_element() {
        let mut page = InternalPage::<4096>::new();
        page.insert(10, 100);
        let right = page.split();
        // mid = 0, left gets 0, right gets 1
        assert_eq!(page.len(), 0);
        assert_eq!(right.len(), 1);
        assert_eq!(right.key(0), 10);
    }

    #[test]
    fn leaf_page_get_absent_key() {
        let mut page = LeafPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(10, b"hello", &mut pager).unwrap();
        assert!(page.get(999, &mut pager).unwrap().is_none());
    }

    #[test]
    fn leaf_page_get_empty_page() {
        let page = LeafPage::<4096>::new();
        let mut pager = test_pager();
        assert!(page.get(10, &mut pager).unwrap().is_none());
    }

    #[test]
    fn leaf_page_insert_empty_value() {
        let mut page = LeafPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(1, b"", &mut pager).unwrap();
        assert_eq!(page.len(), 1);
        assert_eq!(page.value(0), b"");
        assert_eq!(page.get(1, &mut pager).unwrap().unwrap().as_ref(), b"");
    }

    #[test]
    fn leaf_page_remove_first_and_last() {
        let mut page = LeafPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(10, b"a", &mut pager).unwrap();
        page.insert(20, b"b", &mut pager).unwrap();
        page.insert(30, b"c", &mut pager).unwrap();

        page.remove(0);
        assert_eq!(page.len(), 2);
        assert_eq!(page.key(0), 20);

        page.remove(page.len() - 1);
        assert_eq!(page.len(), 1);
        assert_eq!(page.key(0), 20);
        assert_eq!(page.value(0), b"b");
    }

    #[test]
    fn leaf_page_remove_only_element() {
        let mut page = LeafPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(10, b"hi", &mut pager).unwrap();
        page.remove(0);
        assert_eq!(page.len(), 0);
    }

    #[test]
    fn leaf_page_split_odd_count() {
        let mut page = LeafPage::<4096>::new();
        let mut pager = test_pager();
        for i in 0u64..5 {
            page.insert(i, &[i as u8], &mut pager).unwrap();
        }
        let right = page.split();
        assert_eq!(page.len(), 2);
        assert_eq!(right.len(), 3);
    }

    #[test]
    fn leaf_page_split_single_element() {
        let mut page = LeafPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(10, b"only", &mut pager).unwrap();
        let right = page.split();
        assert_eq!(page.len(), 0);
        assert_eq!(right.len(), 1);
        assert_eq!(right.value(0), b"only");
    }

    #[test]
    fn leaf_page_fill_to_capacity() {
        let mut page = LeafPage::<128>::new();
        let mut pager = test_pager();
        let mut i = 0u64;
        while page.can_insert(3) {
            page.insert(i, &[0xAA; 3], &mut pager).unwrap();
            i += 1;
        }
        assert!(i > 0);
        assert!(!page.can_insert(3));
        // All entries should be readable
        for j in 0..i {
            assert_eq!(
                page.get(j, &mut pager).unwrap().unwrap().as_ref(),
                &[0xAA; 3]
            );
        }
    }

    #[test]
    fn index_internal_page_find_child_empty() {
        let page = IndexInternalPage::<4096>::new();
        let mut pager = test_pager();
        assert_eq!(page.find_child(b"anything", &mut pager).unwrap(), None);
    }

    #[test]
    fn index_internal_page_find_child_single_entry() {
        let mut page = IndexInternalPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(b"mmm", 42, &mut pager).unwrap();

        assert_eq!(page.find_child(b"aaa", &mut pager).unwrap(), Some(42));
        assert_eq!(page.find_child(b"mmm", &mut pager).unwrap(), Some(42));
        assert_eq!(page.find_child(b"zzz", &mut pager).unwrap(), None);
    }

    #[test]
    fn index_internal_page_insert_empty_key() {
        let mut page = IndexInternalPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(b"", 10, &mut pager).unwrap();
        page.insert(b"aaa", 20, &mut pager).unwrap();
        assert_eq!(page.len(), 2);
        assert_eq!(page.key(0), b"");
        assert_eq!(page.ptr(0), 10);
        assert_eq!(page.find_child(b"", &mut pager).unwrap(), Some(10));
    }

    #[test]
    fn index_internal_page_remove_first_and_last() {
        let mut page = IndexInternalPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(b"aaa", 1, &mut pager).unwrap();
        page.insert(b"bbb", 2, &mut pager).unwrap();
        page.insert(b"ccc", 3, &mut pager).unwrap();

        page.remove(0);
        assert_eq!(page.len(), 2);
        assert_eq!(page.key(0), b"bbb");

        page.remove(page.len() - 1);
        assert_eq!(page.len(), 1);
        assert_eq!(page.key(0), b"bbb");
    }

    #[test]
    fn index_internal_page_remove_only_element() {
        let mut page = IndexInternalPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(b"only", 99, &mut pager).unwrap();
        page.remove(0);
        assert_eq!(page.len(), 0);
    }

    #[test]
    fn index_internal_page_split_odd_count() {
        let mut page = IndexInternalPage::<4096>::new();
        let mut pager = test_pager();
        for i in 0..5u64 {
            page.insert(&[b'a' + i as u8], i, &mut pager).unwrap();
        }
        let right = page.split();
        assert_eq!(page.len(), 2);
        assert_eq!(right.len(), 3);
    }

    #[test]
    fn index_internal_page_split_single_element() {
        let mut page = IndexInternalPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(b"only", 1, &mut pager).unwrap();
        let right = page.split();
        assert_eq!(page.len(), 0);
        assert_eq!(right.len(), 1);
        assert_eq!(right.key(0), b"only");
    }

    #[test]
    fn index_internal_page_insert_binary_keys() {
        let mut page = IndexInternalPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(&[0x00, 0xFF], 1, &mut pager).unwrap();
        page.insert(&[0x00, 0x00], 2, &mut pager).unwrap();
        page.insert(&[0xFF, 0x00], 3, &mut pager).unwrap();

        assert_eq!(page.key(0), &[0x00, 0x00]);
        assert_eq!(page.key(1), &[0x00, 0xFF]);
        assert_eq!(page.key(2), &[0xFF, 0x00]);
    }

    #[test]
    fn index_internal_page_variable_length_keys() {
        let mut page = IndexInternalPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(b"a", 1, &mut pager).unwrap();
        page.insert(b"aa", 2, &mut pager).unwrap();
        page.insert(b"aaa", 3, &mut pager).unwrap();
        page.insert(b"ab", 4, &mut pager).unwrap();

        // "a" < "aa" < "aaa" < "ab"
        assert_eq!(page.key(0), b"a");
        assert_eq!(page.key(1), b"aa");
        assert_eq!(page.key(2), b"aaa");
        assert_eq!(page.key(3), b"ab");
    }

    #[test]
    fn index_internal_page_fill_to_capacity() {
        let mut page = IndexInternalPage::<128>::new();
        let mut pager = test_pager();
        let mut i = 0u64;
        while page.can_insert(3) {
            let key = format!("{:03}", i);
            page.insert(key.as_bytes(), i, &mut pager).unwrap();
            i += 1;
        }
        assert!(i > 0);
        assert!(!page.can_insert(3));
        assert_eq!(page.len(), i as usize);
    }

    #[test]
    fn index_leaf_page_get_empty() {
        let page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();
        assert_eq!(page.get(b"anything", &mut pager).unwrap(), None);
    }

    #[test]
    fn index_leaf_page_insert_empty_key() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(b"", 10, &mut pager).unwrap();
        assert_eq!(page.len(), 1);
        assert_eq!(page.key(0), b"");
        assert_eq!(page.get(b"", &mut pager).unwrap(), Some(10));
    }

    #[test]
    fn index_leaf_page_remove_first_and_last() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(b"aaa", 1, &mut pager).unwrap();
        page.insert(b"bbb", 2, &mut pager).unwrap();
        page.insert(b"ccc", 3, &mut pager).unwrap();

        page.remove(0);
        assert_eq!(page.len(), 2);
        assert_eq!(page.key(0), b"bbb");

        page.remove(page.len() - 1);
        assert_eq!(page.len(), 1);
        assert_eq!(page.key(0), b"bbb");
        assert_eq!(page.value(0), 2);
    }

    #[test]
    fn index_leaf_page_remove_only_element() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(b"only", 1, &mut pager).unwrap();
        page.remove(0);
        assert_eq!(page.len(), 0);
    }

    #[test]
    fn index_leaf_page_split_odd_count() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();
        for i in 0..5u64 {
            page.insert(&[b'a' + i as u8], i, &mut pager).unwrap();
        }
        let right = page.split();
        assert_eq!(page.len(), 2);
        assert_eq!(right.len(), 3);
    }

    #[test]
    fn index_leaf_page_split_single_element() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();
        page.insert(b"only", 1, &mut pager).unwrap();
        let right = page.split();
        assert_eq!(page.len(), 0);
        assert_eq!(right.len(), 1);
        assert_eq!(right.key(0), b"only");
    }

    #[test]
    fn index_leaf_page_split_preserves_duplicate_keys() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();
        // Insert 6 entries: 3 with "aaa", 3 with "bbb"
        for i in 0..3u64 {
            page.insert_entry(b"aaa", i, &mut pager).unwrap();
        }
        for i in 10..13u64 {
            page.insert_entry(b"bbb", i, &mut pager).unwrap();
        }
        let right = page.split();
        assert_eq!(page.len(), 3);
        assert_eq!(right.len(), 3);

        // Left should have (aaa,0), (aaa,1), (aaa,2)
        for i in 0..3 {
            assert_eq!(page.key(i), b"aaa");
            assert_eq!(page.value(i), i as u64);
        }
        // Right should have (bbb,10), (bbb,11), (bbb,12)
        for i in 0..3 {
            assert_eq!(right.key(i), b"bbb");
            assert_eq!(right.value(i), 10 + i as u64);
        }
    }

    #[test]
    fn index_leaf_page_search_entry_empty() {
        let page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();
        assert_eq!(page.search_entry(b"aaa", 0, &mut pager).unwrap(), None);
    }

    #[test]
    fn index_leaf_page_find_entry_empty() {
        let page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();
        assert_eq!(page.find_entry(b"aaa", 0, &mut pager).unwrap(), None);
    }

    #[test]
    fn index_leaf_page_remove_entry_empty() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();
        assert!(!page.remove_entry(b"aaa", 0, &mut pager).unwrap());
    }

    #[test]
    fn index_leaf_page_insert_entry_with_max_key_values() {
        let mut page = IndexLeafPage::<4096>::new();
        let mut pager = test_pager();
        page.insert_entry(b"x", u64::MAX, &mut pager).unwrap();
        page.insert_entry(b"x", 0, &mut pager).unwrap();
        page.insert_entry(b"x", u64::MAX - 1, &mut pager).unwrap();

        assert_eq!(page.len(), 3);
        assert_eq!(page.value(0), 0);
        assert_eq!(page.value(1), u64::MAX - 1);
        assert_eq!(page.value(2), u64::MAX);
    }

    #[test]
    fn index_leaf_page_fill_to_capacity() {
        let mut page = IndexLeafPage::<128>::new();
        let mut pager = test_pager();
        let mut i = 0u64;
        while page.can_insert(3) {
            let key = format!("{:03}", i);
            page.insert(key.as_bytes(), i, &mut pager).unwrap();
            i += 1;
        }
        assert!(i > 0);
        assert!(!page.can_insert(3));
        assert_eq!(page.len(), i as usize);
    }

    #[test]
    fn index_leaf_page_insert_after_remove_reuses_space() {
        let mut page = IndexLeafPage::<128>::new();
        let mut pager = test_pager();
        // Fill up
        for i in 0..5u64 {
            page.insert(&[b'a' + i as u8; 3], i, &mut pager).unwrap();
        }
        let free_before = page.free_space();
        // Remove some to create dead space
        page.remove(1);
        page.remove(1);
        let free_after = page.free_space();
        assert!(free_after > free_before);
        // Should be able to insert again
        assert!(page.can_insert(3));
        page.insert(b"zzz", 99, &mut pager).unwrap();
        assert_eq!(page.get(b"zzz", &mut pager).unwrap(), Some(99));
    }

    #[test]
    fn index_internal_page_compact_after_removes() {
        let mut page = IndexInternalPage::<128>::new();
        let mut pager = test_pager();
        for i in 0..5u64 {
            page.insert(&[b'a' + i as u8; 3], i, &mut pager).unwrap();
        }
        page.remove(1);
        page.remove(1);
        // Insert should trigger compaction and succeed
        assert!(page.can_insert(3));
        page.insert(b"zzz", 99, &mut pager).unwrap();
        assert_eq!(page.find_child(b"zzz", &mut pager).unwrap(), Some(99));
    }
}
