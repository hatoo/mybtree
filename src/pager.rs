use lru::LruCache;
use rkyv::{
    api::high,
    rancor::{Error, Source},
    util::AlignedVec,
};
use std::collections::BTreeSet;
use std::io;
use std::num::NonZeroUsize;

use crate::types::Node;
use crate::{page::AnyPage, types::IndexNode};

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

    pub fn init(&mut self) -> Result<(), Error> {
        self.file.set_len(0).map_err(Error::new)?;
        self.cache.clear();
        self.dirty.clear();
        Ok(())
    }

    pub fn next_page_num(&mut self) -> u64 {
        let page_num = self.next_page_num;
        self.next_page_num += 1;
        page_num
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

    pub fn read_node<T>(&mut self, page_num: u64) -> Result<&AnyPage<N>, Error> {
        let data = self.cache_read(page_num).map_err(Error::new)?;
        Ok(data)
    }

    pub fn owned_node(&mut self, page_num: u64) -> Result<AnyPage<N>, Error> {
        let data = self.cache_read(page_num).map_err(Error::new)?;
        Ok(data.clone())
    }

    pub fn write_node(&mut self, page_num: u64, node: AnyPage<N>) -> Result<(), Error> {
        self.cache_write(page_num, Box::new(node))
            .map_err(Error::new)
    }
}

impl<const N: usize> Drop for Pager<N> {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
