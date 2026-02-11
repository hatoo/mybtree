use rkyv::{
    api::high,
    rancor::{Error, Source},
    util::AlignedVec,
};
use std::io;

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

pub struct Pager {
    page_size: usize,
    file: std::fs::File,
    next_page_num: u64,
}

impl Pager {
    pub fn new(file: std::fs::File, page_size: usize) -> Self {
        let file_size = file.metadata().unwrap().len();
        let next_page_num = file_size / page_size as u64;
        Pager {
            page_size,
            file,
            next_page_num,
        }
    }

    pub fn init(&mut self) -> Result<(), Error> {
        self.file.set_len(0).map_err(Error::new)?;
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

    fn from_page<'a>(&self, buffer: &'a AlignedVec<16>) -> &'a [u8] {
        let len = u16::from_le_bytes([
            buffer[self.page_content_size()],
            buffer[self.page_content_size() + 1],
        ]) as usize;
        &buffer[..len]
    }

    fn to_page(&self, buffer: &mut AlignedVec<16>) {
        assert!(buffer.len() <= self.page_content_size());
        let len = buffer.len() as u16;
        buffer.resize(self.page_size, 0);
        buffer[self.page_content_size()..self.page_size].copy_from_slice(&len.to_le_bytes());
    }

    pub fn read_node<T>(
        &self,
        page_num: u64,
        f: impl FnOnce(&rkyv::Archived<Node>) -> T,
    ) -> Result<T, Error> {
        let mut buffer = AlignedVec::<16>::with_capacity(self.page_size);
        buffer.resize(self.page_size, 0);
        read_exact_at(&self.file, &mut buffer, page_num * self.page_size as u64)
            .map_err(Error::new)?;
        let buffer = self.from_page(&buffer);
        let archived = high::access::<rkyv::Archived<Node>, Error>(&buffer)?;
        Ok(f(archived))
    }

    pub fn owned_node(&self, page_num: u64) -> Result<Node, Error> {
        let mut buffer = AlignedVec::<16>::with_capacity(self.page_size);
        buffer.resize(self.page_size, 0);
        read_exact_at(&self.file, &mut buffer, page_num * self.page_size as u64)
            .map_err(Error::new)?;
        let buffer = self.from_page(&buffer);
        let archived = rkyv::access::<rkyv::Archived<Node>, Error>(&buffer)?;
        let node: Node = rkyv::deserialize(archived)?;
        Ok(node)
    }

    pub fn write_buffer(&self, page_num: u64, mut buffer: AlignedVec<16>) -> Result<(), Error> {
        self.to_page(&mut buffer);
        write_all_at(&self.file, &buffer, page_num * self.page_size as u64).map_err(Error::new)?;
        Ok(())
    }

    pub fn write_node(&self, page_num: u64, node: &Node) -> Result<(), Error> {
        let buffer = rkyv::to_bytes(node)?;
        self.write_buffer(page_num, buffer)
    }

    pub fn read_index_node<T>(
        &self,
        page_num: u64,
        f: impl FnOnce(&rkyv::Archived<IndexNode>) -> T,
    ) -> Result<T, Error> {
        let mut buffer = AlignedVec::<16>::with_capacity(self.page_size);
        buffer.resize(self.page_size, 0);
        read_exact_at(&self.file, &mut buffer, page_num * self.page_size as u64)
            .map_err(Error::new)?;
        let buffer = self.from_page(&buffer);
        let archived = high::access::<rkyv::Archived<IndexNode>, Error>(&buffer)?;
        Ok(f(archived))
    }

    pub fn owned_index_node(&self, page_num: u64) -> Result<IndexNode, Error> {
        let mut buffer = AlignedVec::<16>::with_capacity(self.page_size);
        buffer.resize(self.page_size, 0);
        read_exact_at(&self.file, &mut buffer, page_num * self.page_size as u64)
            .map_err(Error::new)?;
        let buffer = self.from_page(&buffer);
        let archived = rkyv::access::<rkyv::Archived<IndexNode>, Error>(&buffer)?;
        let node: IndexNode = rkyv::deserialize(archived)?;
        Ok(node)
    }

    pub fn write_index_node(&self, page_num: u64, node: &IndexNode) -> Result<(), Error> {
        let buffer = rkyv::to_bytes(node)?;
        self.write_buffer(page_num, buffer)
    }

    pub fn write_raw_page(&self, page_num: u64, data: &[u8]) -> Result<(), Error> {
        assert!(data.len() <= self.page_size);
        let offset = page_num * self.page_size as u64;
        write_all_at(&self.file, data, offset).map_err(Error::new)?;
        if data.len() < self.page_size {
            let padding = vec![0u8; self.page_size - data.len()];
            write_all_at(&self.file, &padding, offset + data.len() as u64).map_err(Error::new)?;
        }
        Ok(())
    }

    pub fn read_raw_page(&self, page_num: u64) -> Result<Vec<u8>, Error> {
        let mut buffer = vec![0u8; self.page_size];
        read_exact_at(&self.file, &mut buffer, page_num * self.page_size as u64)
            .map_err(Error::new)?;
        Ok(buffer)
    }
}
