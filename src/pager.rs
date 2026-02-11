use lru::LruCache;
use rkyv::{
    api::high,
    rancor::{Error, Source},
    util::AlignedVec,
};
use std::collections::BTreeSet;
use std::io;
use std::num::NonZeroUsize;

use crate::types::IndexNode;
use crate::types::Node;

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

pub struct Pager {
    page_size: usize,
    file: std::fs::File,
    next_page_num: u64,
    cache: LruCache<u64, Vec<u8>>,
    dirty: BTreeSet<u64>,
}

impl Pager {
    pub fn new(file: std::fs::File, page_size: usize) -> Self {
        Self::with_cache_capacity(file, page_size, DEFAULT_CACHE_CAPACITY)
    }

    pub fn with_cache_capacity(file: std::fs::File, page_size: usize, capacity: usize) -> Self {
        let file_size = file.metadata().unwrap().len();
        let next_page_num = file_size / page_size as u64;
        Pager {
            page_size,
            file,
            next_page_num,
            cache: LruCache::new(NonZeroUsize::new(capacity).unwrap()),
            dirty: BTreeSet::new(),
        }
    }

    pub fn init(&mut self) -> Result<(), Error> {
        self.file.set_len(0).map_err(Error::new)?;
        self.cache.clear();
        self.dirty.clear();
        Ok(())
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn page_content_size(&self) -> usize {
        self.page_size - 2
    }

    pub fn next_page_num(&mut self) -> u64 {
        let page_num = self.next_page_num;
        self.next_page_num += 1;
        page_num
    }

    pub fn total_page_count(&self) -> u64 {
        self.next_page_num
    }

    fn from_page_slice(page_size: usize, buffer: &[u8]) -> &[u8] {
        let content_size = page_size - 2;
        let len =
            u16::from_le_bytes([buffer[content_size], buffer[content_size + 1]]) as usize;
        &buffer[..len]
    }

    fn to_page(&self, buffer: &mut AlignedVec<16>) {
        assert!(buffer.len() <= self.page_content_size());
        let len = buffer.len() as u16;
        buffer.resize(self.page_size, 0);
        buffer[self.page_content_size()..self.page_size].copy_from_slice(&len.to_le_bytes());
    }

    /// Write an evicted dirty page to disk.
    fn flush_page(&self, page_num: u64, data: &[u8]) -> io::Result<()> {
        write_all_at(&self.file, data, page_num * self.page_size as u64)
    }

    /// Insert a page into the cache, flushing any evicted dirty page to disk.
    fn cache_put(&mut self, page_num: u64, data: Vec<u8>) -> io::Result<()> {
        if let Some((evicted_page, evicted_data)) = self.cache.push(page_num, data) {
            if self.dirty.remove(&evicted_page) {
                self.flush_page(evicted_page, &evicted_data)?;
            }
        }
        Ok(())
    }

    /// Read a page from cache, loading from disk on miss.
    fn cache_read(&mut self, page_num: u64) -> io::Result<&Vec<u8>> {
        if self.cache.contains(&page_num) {
            return Ok(self.cache.get(&page_num).unwrap());
        }
        let mut buf = vec![0u8; self.page_size];
        read_exact_at(&self.file, &mut buf, page_num * self.page_size as u64)?;
        self.cache_put(page_num, buf)?;
        Ok(self.cache.get(&page_num).unwrap())
    }

    /// Write a page into the cache and mark it dirty.
    fn cache_write(&mut self, page_num: u64, data: Vec<u8>) -> io::Result<()> {
        self.cache_put(page_num, data)?;
        self.dirty.insert(page_num);
        Ok(())
    }

    /// Flush all dirty pages to disk and sync the file.
    pub fn flush(&mut self) -> io::Result<()> {
        let dirty_pages = std::mem::take(&mut self.dirty);
        for page_num in dirty_pages {
            if let Some(data) = self.cache.peek(&page_num) {
                self.flush_page(page_num, &data.clone())?;
            }
        }
        self.file.sync_all()
    }

    pub fn read_node<T>(
        &mut self,
        page_num: u64,
        f: impl FnOnce(&rkyv::Archived<Node>) -> T,
    ) -> Result<T, Error> {
        let page_size = self.page_size;
        let data = self.cache_read(page_num).map_err(Error::new)?;
        let buffer = Self::from_page_slice(page_size, data);
        let archived = high::access::<rkyv::Archived<Node>, Error>(buffer)?;
        Ok(f(archived))
    }

    pub fn owned_node(&mut self, page_num: u64) -> Result<Node, Error> {
        let page_size = self.page_size;
        let data = self.cache_read(page_num).map_err(Error::new)?;
        let buffer = Self::from_page_slice(page_size, data);
        let archived = rkyv::access::<rkyv::Archived<Node>, Error>(buffer)?;
        let node: Node = rkyv::deserialize(archived)?;
        Ok(node)
    }

    pub fn write_buffer(&mut self, page_num: u64, mut buffer: AlignedVec<16>) -> Result<(), Error> {
        self.to_page(&mut buffer);
        self.cache_write(page_num, buffer.to_vec())
            .map_err(Error::new)?;
        Ok(())
    }

    pub fn write_node(&mut self, page_num: u64, node: &Node) -> Result<(), Error> {
        let buffer = rkyv::to_bytes(node)?;
        self.write_buffer(page_num, buffer)
    }

    pub fn read_index_node<T>(
        &mut self,
        page_num: u64,
        f: impl FnOnce(&rkyv::Archived<IndexNode>) -> T,
    ) -> Result<T, Error> {
        let page_size = self.page_size;
        let data = self.cache_read(page_num).map_err(Error::new)?;
        let buffer = Self::from_page_slice(page_size, data);
        let archived = high::access::<rkyv::Archived<IndexNode>, Error>(buffer)?;
        Ok(f(archived))
    }

    pub fn owned_index_node(&mut self, page_num: u64) -> Result<IndexNode, Error> {
        let page_size = self.page_size;
        let data = self.cache_read(page_num).map_err(Error::new)?;
        let buffer = Self::from_page_slice(page_size, data);
        let archived = rkyv::access::<rkyv::Archived<IndexNode>, Error>(buffer)?;
        let node: IndexNode = rkyv::deserialize(archived)?;
        Ok(node)
    }

    pub fn write_index_node(&mut self, page_num: u64, node: &IndexNode) -> Result<(), Error> {
        let buffer = rkyv::to_bytes(node)?;
        self.write_buffer(page_num, buffer)
    }

    pub fn write_raw_page(&mut self, page_num: u64, data: &[u8]) -> Result<(), Error> {
        assert!(data.len() <= self.page_size);
        let mut page = vec![0u8; self.page_size];
        page[..data.len()].copy_from_slice(data);
        self.cache_write(page_num, page).map_err(Error::new)?;
        Ok(())
    }

    pub fn read_raw_page(&mut self, page_num: u64) -> Result<Vec<u8>, Error> {
        let data = self.cache_read(page_num).map_err(Error::new)?;
        Ok(data.clone())
    }
}

impl Drop for Pager {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
