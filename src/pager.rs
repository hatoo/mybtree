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

    /// Write a large value across one or more overflow pages (rkyv-serialized).
    /// Returns the page number of the first overflow page.
    pub fn write_overflow(&mut self, data: &[u8]) -> Result<u64, Error> {
        let data_per_page = self.overflow_data_per_page();
        let num_pages = (data.len() + data_per_page - 1) / data_per_page;
        assert!(num_pages > 0);

        let pages: Vec<u64> = (0..num_pages).map(|_| self.next_page_num()).collect();

        for (i, &page_num) in pages.iter().enumerate() {
            let start = i * data_per_page;
            let end = std::cmp::min(start + data_per_page, data.len());
            let chunk = &data[start..end];

            let next_page: u64 = if i + 1 < pages.len() {
                pages[i + 1]
            } else {
                u64::MAX
            };

            let overflow_page = OverflowPage {
                next_page,
                data: chunk.to_vec(),
            };
            let buffer = rkyv::to_bytes(&overflow_page)?;
            self.write_buffer(page_num, buffer)?;
        }

        Ok(pages[0])
    }

    /// Read a large value stored across overflow pages (rkyv-serialized).
    pub fn read_overflow(&mut self, start_page: u64, total_len: u64) -> Result<Vec<u8>, Error> {
        let mut result = Vec::with_capacity(total_len as usize);
        let mut current_page = start_page;
        let mut remaining = total_len as usize;

        while remaining > 0 {
            let overflow_page = self.read_overflow_page(current_page)?;
            let chunk_len = std::cmp::min(overflow_page.data.len(), remaining);
            result.extend_from_slice(&overflow_page.data[..chunk_len]);
            remaining -= chunk_len;
            current_page = overflow_page.next_page;
        }

        Ok(result)
    }

    fn read_overflow_page(&mut self, page_num: u64) -> Result<OverflowPage, Error> {
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

    /// Maximum data bytes per overflow page (page_content_size minus rkyv overhead for OverflowPage).
    pub fn overflow_data_per_page(&self) -> usize {
        // Serialize with a well-aligned reference size to measure fixed overhead (struct + vec metadata)
        let ref_size: usize = 8;
        let serialized_len = rkyv::to_bytes::<Error>(&OverflowPage {
            next_page: 0,
            data: vec![0u8; ref_size],
        })
        .unwrap()
        .len();
        let overhead = serialized_len - ref_size;
        let available = self.page_content_size().saturating_sub(overhead);
        // Round down to 8-byte alignment to guarantee no extra padding between data and struct
        (available / 8) * 8
    }
}
