use rkyv::{api::high, rancor::Error, util::AlignedVec};
use std::io::{Read, Seek, Write};

use crate::types::{Node, PAGE_CONTENT_SIZE, PAGE_SIZE};

pub struct Pager {
    pub file: std::fs::File,
    pub next_page_num: u64,
}

pub fn from_page(buffer: &AlignedVec<16>) -> &[u8] {
    let len =
        u16::from_le_bytes([buffer[PAGE_CONTENT_SIZE], buffer[PAGE_CONTENT_SIZE + 1]]) as usize;
    &buffer[..len]
}

pub fn to_page(buffer: &mut AlignedVec<16>) {
    assert!(buffer.len() <= PAGE_CONTENT_SIZE);
    let len = buffer.len() as u16;
    buffer.resize(PAGE_SIZE, 0);
    buffer[PAGE_CONTENT_SIZE..PAGE_SIZE].copy_from_slice(&len.to_le_bytes());
}

impl Pager {
    pub fn new(file: std::fs::File) -> Self {
        let file_size = file.metadata().unwrap().len();
        let next_page_num = file_size / PAGE_SIZE as u64;
        Pager {
            file,
            next_page_num,
        }
    }

    pub fn next_page_num(&mut self) -> u64 {
        let page_num = self.next_page_num;
        self.next_page_num += 1;
        page_num
    }

    pub fn read_node<T>(
        &mut self,
        page_num: u64,
        f: impl FnOnce(&rkyv::Archived<Node>) -> T,
    ) -> Result<T, Error> {
        let mut buffer = AlignedVec::<16>::with_capacity(PAGE_SIZE);
        buffer.resize(PAGE_SIZE, 0);
        self.file
            .seek(std::io::SeekFrom::Start(page_num * PAGE_SIZE as u64))
            .unwrap();
        self.file.read_exact(&mut buffer).unwrap();
        let buffer = from_page(&buffer);
        let archived = high::access::<rkyv::Archived<Node>, Error>(&buffer)?;
        Ok(f(archived))
    }

    pub fn owned_node(&mut self, page_num: u64) -> Result<Node, Error> {
        let mut buffer = AlignedVec::<16>::with_capacity(PAGE_SIZE);
        buffer.resize(PAGE_SIZE, 0);
        self.file
            .seek(std::io::SeekFrom::Start(page_num * PAGE_SIZE as u64))
            .unwrap();
        self.file.read_exact(&mut buffer).unwrap();
        let buffer = from_page(&buffer);
        let archived = rkyv::access::<rkyv::Archived<Node>, Error>(&buffer)?;
        let node: Node = rkyv::deserialize(archived)?;
        Ok(node)
    }

    pub fn write_buffer(&mut self, page_num: u64, mut buffer: AlignedVec<16>) -> Result<(), Error> {
        to_page(&mut buffer);
        self.file
            .seek(std::io::SeekFrom::Start(page_num * PAGE_SIZE as u64))
            .unwrap();
        self.file.write_all(&buffer).unwrap();
        Ok(())
    }

    pub fn write_node(&mut self, page_num: u64, node: &Node) -> Result<(), Error> {
        let buffer = rkyv::to_bytes(node)?;
        self.write_buffer(page_num, buffer)
    }
}
