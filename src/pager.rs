use lru::LruCache;
use std::collections::BTreeSet;
use std::io;
use std::num::NonZeroUsize;

use crate::page::AnyPage;
use crate::types::{FREE_LIST_PAGE_NUM, NodePtr};

pub const OVERFLOW_FLAG: u16 = 0x8000;
pub const OVERFLOW_META_SIZE: usize = 16;

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

    pub fn into_value_and_free_overflow_pages(self, pager: &mut Pager<N>) -> io::Result<Vec<u8>> {
        match self {
            ValueToken::Inline(data) => Ok(data),
            ValueToken::Overflow(start_page, total_len) => {
                let data = pager.read_overflow(start_page, total_len)?;
                pager.free_overflow_pages(start_page, total_len)?;
                Ok(data)
            }
        }
    }
}

#[cfg(unix)]
fn read_exact_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(unix)]
fn write_all_at(file: &std::fs::File, buf: &[u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &std::fs::File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(windows)]
fn write_all_at(file: &std::fs::File, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_write(buf, offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "failed to write whole buffer",
                ));
            }
            Ok(n) => {
                buf = &buf[n..];
                offset += n as u64;
            }
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

const DEFAULT_CACHE_CAPACITY: usize = 256;

pub struct Pager<const N: usize> {
    file: std::fs::File,
    next_page_num: u64,
    cache: LruCache<u64, Box<AnyPage<N>>>,
    dirty: BTreeSet<u64>,
}

impl<const N: usize> Pager<N> {
    pub fn new(file: std::fs::File) -> Self {
        Self::with_cache_capacity(file, DEFAULT_CACHE_CAPACITY)
    }

    pub fn with_cache_capacity(file: std::fs::File, capacity: usize) -> Self {
        let file_size = file.metadata().unwrap().len();
        let next_page_num = file_size / N as u64;
        Pager {
            file,
            next_page_num,
            cache: LruCache::new(NonZeroUsize::new(capacity).unwrap()),
            dirty: BTreeSet::new(),
        }
    }

    pub fn init(&mut self) -> io::Result<()> {
        self.file.set_len(0)?;
        self.cache.clear();
        self.dirty.clear();
        Ok(())
    }

    pub fn next_page_num(&mut self) -> u64 {
        let page_num = self.next_page_num;
        self.next_page_num += 1;
        page_num
    }

    /// Read free list head stored at page `FREE_LIST_PAGE_NUM`.
    pub fn read_free_list_head(&mut self) -> io::Result<u64> {
        let buf = self.read_raw_page(FREE_LIST_PAGE_NUM)?;
        Ok(u64::from_le_bytes(buf[..8].try_into().unwrap()))
    }

    /// Write free list head to page `FREE_LIST_PAGE_NUM`.
    pub fn write_free_list_head(&mut self, head: u64) -> io::Result<()> {
        let mut buf = vec![0u8; 8];
        buf[..8].copy_from_slice(&head.to_le_bytes());
        self.write_raw_page(FREE_LIST_PAGE_NUM, &buf)
    }

    /// Allocate a page either from the free list or by growing the file.
    pub fn alloc_page(&mut self) -> io::Result<u64> {
        let head = self.read_free_list_head()?;
        if head == u64::MAX {
            return Ok(self.next_page_num());
        }
        let buf = self.read_raw_page(head)?;
        let next = u64::from_le_bytes(buf[..8].try_into().unwrap());
        self.write_free_list_head(next)?;
        Ok(head)
    }

    /// Add `page_num` to the head of the free list.
    pub fn free_page(&mut self, page_num: u64) -> io::Result<()> {
        let head = self.read_free_list_head()?;
        let mut buf = vec![0u8; 8];
        buf[..8].copy_from_slice(&head.to_le_bytes());
        self.write_raw_page(page_num, &buf)?;
        self.write_free_list_head(page_num)
    }

    pub fn total_page_count(&self) -> u64 {
        self.next_page_num
    }

    /// Write an evicted dirty page to disk.
    fn flush_page(&self, page_num: u64, data: &AnyPage<N>) -> io::Result<()> {
        write_all_at(&self.file, &data.page, page_num * N as u64)
    }

    /// Insert a page into the cache, flushing any evicted dirty page to disk.
    fn cache_put(&mut self, page_num: u64, data: Box<AnyPage<N>>) -> io::Result<()> {
        if let Some((evicted_page, evicted_data)) = self.cache.push(page_num, data) {
            if self.dirty.remove(&evicted_page) {
                self.flush_page(evicted_page, &evicted_data)?;
            }
        }
        Ok(())
    }

    /// Read a page from cache, loading from disk on miss.
    fn cache_read(&mut self, page_num: u64) -> io::Result<&AnyPage<N>> {
        if self.cache.contains(&page_num) {
            return Ok(self.cache.get(&page_num).unwrap());
        }
        let mut buf = vec![0u8; N];
        read_exact_at(&self.file, &mut buf, page_num * N as u64)?;
        self.cache_put(
            page_num,
            Box::new(AnyPage {
                page: buf.try_into().unwrap(),
            }),
        )?;
        Ok(self.cache.get(&page_num).unwrap())
    }

    /// Write a page into the cache and mark it dirty.
    fn cache_write(&mut self, page_num: u64, data: Box<AnyPage<N>>) -> io::Result<()> {
        self.cache_put(page_num, data)?;
        self.dirty.insert(page_num);
        Ok(())
    }

    /// Flush all dirty pages to disk and sync the file.
    pub fn flush(&mut self) -> io::Result<()> {
        let dirty_pages = std::mem::take(&mut self.dirty);
        for page_num in dirty_pages {
            if let Some(data) = self.cache.peek(&page_num) {
                self.flush_page(page_num, data)?;
            }
        }
        self.file.sync_all()
    }

    pub fn read_node(&mut self, page_num: u64) -> io::Result<&AnyPage<N>> {
        let data = self.cache_read(page_num)?;
        Ok(data)
    }

    pub fn owned_node(&mut self, page_num: u64) -> io::Result<AnyPage<N>> {
        let data = self.cache_read(page_num)?;
        Ok(data.clone())
    }

    pub fn write_node(&mut self, page_num: u64, node: AnyPage<N>) -> io::Result<()> {
        self.cache_write(page_num, Box::new(node))?;
        Ok(())
    }

    pub fn mut_node(&mut self, page_num: u64) -> io::Result<&mut AnyPage<N>> {
        self.dirty.insert(page_num);
        if self.cache.contains(&page_num) {
            Ok(self.cache.get_mut(&page_num).unwrap())
        } else {
            let mut buf = vec![0u8; N];

            let _ = read_exact_at(&self.file, &mut buf, page_num * N as u64);
            self.cache_put(
                page_num,
                Box::new(AnyPage {
                    page: buf.try_into().unwrap(),
                }),
            )?;
            Ok(self.cache.get_mut(&page_num).unwrap())
        }
    }

    /// Read raw bytes from a page. Returns a reference to the page's raw data.
    pub fn read_raw_page(&mut self, page_num: u64) -> io::Result<&[u8; N]> {
        let data = self.cache_read(page_num)?;
        Ok(&data.page)
    }

    /// Write raw bytes to a page. The slice is padded with zeros to fill the page.
    pub fn write_raw_page(&mut self, page_num: u64, data: &[u8]) -> io::Result<()> {
        let mut page = AnyPage { page: [0u8; N] };
        let len = data.len().min(N);
        page.page[..len].copy_from_slice(&data[..len]);
        self.cache_write(page_num, Box::new(page))?;
        Ok(())
    }

    /// Return the page size in bytes.
    pub fn page_size(&self) -> usize {
        N
    }

    /// Parse overflow metadata bytes into (start_page, total_len).
    pub fn parse_overflow_meta(meta: &[u8]) -> (u64, u64) {
        let start_page = u64::from_le_bytes(meta[0..8].try_into().unwrap());
        let total_len = u64::from_le_bytes(meta[8..16].try_into().unwrap());
        (start_page, total_len)
    }

    /// Build overflow metadata bytes from start_page and total_len.
    pub fn make_overflow_meta(start_page: u64, total_len: u64) -> [u8; OVERFLOW_META_SIZE] {
        let mut meta = [0u8; OVERFLOW_META_SIZE];
        meta[0..8].copy_from_slice(&start_page.to_le_bytes());
        meta[8..16].copy_from_slice(&total_len.to_le_bytes());
        meta
    }

    /// Write overflow data across multiple pages, returns start page number.
    pub fn write_overflow(&mut self, data: &[u8]) -> io::Result<u64> {
        let data_per_page = N - 8;
        let num_pages = (data.len() + data_per_page - 1) / data_per_page;
        assert!(num_pages > 0);

        let pages: Vec<u64> = (0..num_pages)
            .map(|_| self.alloc_page())
            .collect::<io::Result<Vec<u64>>>()?;

        for (i, &page_num) in pages.iter().enumerate() {
            let next_page: u64 = if i + 1 < pages.len() {
                pages[i + 1]
            } else {
                u64::MAX
            };
            let start = i * data_per_page;
            let end = std::cmp::min(start + data_per_page, data.len());
            let chunk = &data[start..end];

            let mut buffer = vec![0u8; 8 + chunk.len()];
            buffer[..8].copy_from_slice(&next_page.to_le_bytes());
            buffer[8..].copy_from_slice(chunk);
            self.write_raw_page(page_num, &buffer)?;
        }

        Ok(pages[0])
    }

    /// Read overflow data starting from a page, for a given total length.
    pub fn read_overflow(&mut self, start_page: u64, total_len: u64) -> io::Result<Vec<u8>> {
        let data_per_page = N - 8;
        let mut result = Vec::with_capacity(total_len as usize);
        let mut current_page = start_page;
        let mut remaining = total_len as usize;

        while remaining > 0 {
            let buffer = self.read_raw_page(current_page)?;
            let next_page = u64::from_le_bytes(buffer[..8].try_into().unwrap());
            let chunk_len = std::cmp::min(data_per_page, remaining);
            result.extend_from_slice(&buffer[8..8 + chunk_len]);
            remaining -= chunk_len;
            current_page = next_page;
        }

        Ok(result)
    }

    /// Free all overflow pages for a given overflow chain.
    pub fn free_overflow_pages(&mut self, start_page: u64, total_len: u64) -> io::Result<()> {
        let data_per_page = N - 8;
        let mut current_page = start_page;
        let mut remaining = total_len as usize;

        while remaining > 0 {
            let buffer = self.read_raw_page(current_page)?;
            let next_page = u64::from_le_bytes(buffer[..8].try_into().unwrap());
            self.free_page(current_page)?;
            let chunk_len = std::cmp::min(data_per_page, remaining);
            remaining -= chunk_len;
            current_page = next_page;
        }
        Ok(())
    }
}

impl<const N: usize> Drop for Pager<N> {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
