use rkyv::{api::high, rancor::Error, util::AlignedVec};
use std::io::{Read, Seek, Write};

use crate::types::{Node, OverflowPage};

pub struct Pager {
    pub page_size: usize,
    pub file: std::fs::File,
    pub next_page_num: u64,
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

    pub fn page_content_size(&self) -> usize {
        self.page_size - 2
    }

    pub fn next_page_num(&mut self) -> u64 {
        let page_num = self.next_page_num;
        self.next_page_num += 1;
        page_num
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
        &mut self,
        page_num: u64,
        f: impl FnOnce(&rkyv::Archived<Node>) -> T,
    ) -> Result<T, Error> {
        let mut buffer = AlignedVec::<16>::with_capacity(self.page_size);
        buffer.resize(self.page_size, 0);
        self.file
            .seek(std::io::SeekFrom::Start(page_num * self.page_size as u64))
            .unwrap();
        self.file.read_exact(&mut buffer).unwrap();
        let buffer = self.from_page(&buffer);
        let archived = high::access::<rkyv::Archived<Node>, Error>(&buffer)?;
        Ok(f(archived))
    }

    pub fn owned_node(&mut self, page_num: u64) -> Result<Node, Error> {
        let mut buffer = AlignedVec::<16>::with_capacity(self.page_size);
        buffer.resize(self.page_size, 0);
        self.file
            .seek(std::io::SeekFrom::Start(page_num * self.page_size as u64))
            .unwrap();
        self.file.read_exact(&mut buffer).unwrap();
        let buffer = self.from_page(&buffer);
        let archived = rkyv::access::<rkyv::Archived<Node>, Error>(&buffer)?;
        let node: Node = rkyv::deserialize(archived)?;
        Ok(node)
    }

    pub fn write_buffer(&mut self, page_num: u64, mut buffer: AlignedVec<16>) -> Result<(), Error> {
        self.to_page(&mut buffer);
        self.file
            .seek(std::io::SeekFrom::Start(page_num * self.page_size as u64))
            .unwrap();
        self.file.write_all(&buffer).unwrap();
        Ok(())
    }

    pub fn write_node(&mut self, page_num: u64, node: &Node) -> Result<(), Error> {
        let buffer = rkyv::to_bytes(node)?;
        self.write_buffer(page_num, buffer)
    }

    pub fn owned_overflow_page(&mut self, page_num: u64) -> Result<OverflowPage, Error> {
        let mut buffer = AlignedVec::<16>::with_capacity(self.page_size);
        buffer.resize(self.page_size, 0);
        self.file
            .seek(std::io::SeekFrom::Start(page_num * self.page_size as u64))
            .unwrap();
        self.file.read_exact(&mut buffer).unwrap();
        let buffer = self.from_page(&buffer);
        let archived = rkyv::access::<rkyv::Archived<OverflowPage>, Error>(&buffer)?;
        let page: OverflowPage = rkyv::deserialize(archived)?;
        Ok(page)
    }
}
