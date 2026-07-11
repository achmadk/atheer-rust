use std::collections::HashMap;

/// Alias for a block index in the manager's pool.
pub type BlockId = usize;

/// Sentinel value meaning "no block allocated".
pub const NULL_BLOCK: BlockId = usize::MAX;

/// Number of tokens stored per block (default).
pub const DEFAULT_BLOCK_SIZE: usize = 16;

// ---------------------------------------------------------------------------
// BlockData
// ---------------------------------------------------------------------------

/// Holds the quantised KV data for a single block (N tokens).
struct BlockData {
    /// Serialised key data for the block's tokens.
    key_data: Vec<u8>,
    /// Serialised value data for the block's tokens.
    value_data: Vec<u8>,
    /// Number of tokens actually stored in this block (≤ block_size).
    token_count: usize,
    /// Reference count for copy-on-write sharing.
    ref_count: u32,
}

impl BlockData {
    fn new() -> Self {
        Self {
            key_data: Vec::new(),
            value_data: Vec::new(),
            token_count: 0,
            ref_count: 0,
        }
    }

    fn is_free(&self) -> bool {
        self.ref_count == 0 && self.token_count == 0
    }
}

// ---------------------------------------------------------------------------
// BlockTable
// ---------------------------------------------------------------------------

/// Maps `(layer_index, logical_token_position / block_size)` → `BlockId`.
///
/// Supports scatter-gather: the KV cache for a layer's sequence is split
/// across potentially non-contiguous physical blocks.
struct BlockTable {
    /// Inner mapping: `(layer, logical_block_index) → BlockId`.
    map: HashMap<(usize, usize), BlockId>,
}

impl BlockTable {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    fn insert(&mut self, layer: usize, logical_block: usize, block_id: BlockId) {
        self.map.insert((layer, logical_block), block_id);
    }

    fn get(&self, layer: usize, logical_block: usize) -> Option<BlockId> {
        self.map.get(&(layer, logical_block)).copied()
    }

    #[allow(dead_code)]
    fn remove(&mut self, layer: usize, logical_block: usize) -> Option<BlockId> {
        self.map.remove(&(layer, logical_block))
    }

    /// Remove all entries for a given layer.
    fn clear_layer(&mut self, layer: usize) {
        self.map.retain(|&(l, _), _| l != layer);
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

// ---------------------------------------------------------------------------
// BlockManager
// ---------------------------------------------------------------------------

/// Manages a fixed-size pool of KV cache blocks for paged attention.
///
/// Each block stores the quantized KV data for `block_size` tokens.
/// The block table maps `(layer, logical_position)` to a physical `BlockId`,
/// enabling non-contiguous cache storage (scatter-gather).
pub struct BlockManager {
    /// Pool of all blocks (free + used).
    blocks: Vec<BlockData>,
    /// Block size in tokens.
    block_size: usize,
    /// Number of consecutive blocks tracked per allocation.
    #[allow(dead_code)]
    num_layers: usize,
    /// Block table mapping logical positions to physical blocks.
    table: BlockTable,
    /// Number of allocated (non-free) blocks.
    allocated_count: usize,
}

impl BlockManager {
    /// Create a new block manager.
    ///
    /// * `total_blocks` — total number of blocks to pre-allocate in the pool.
    /// * `block_size` — number of tokens per block.
    /// * `num_layers` — number of KV cache layers in the model.
    ///
    /// Blocks start in the free pool.
    pub fn new(total_blocks: usize, block_size: usize, num_layers: usize) -> Self {
        let mut blocks = Vec::with_capacity(total_blocks);
        for _ in 0..total_blocks {
            blocks.push(BlockData::new());
        }
        Self {
            blocks,
            block_size,
            num_layers,
            table: BlockTable::new(),
            allocated_count: 0,
        }
    }

    /// Allocate a new block from the free pool.
    ///
    /// Returns a `BlockId` or `None` if the pool is exhausted.
    pub fn alloc_block(&mut self) -> Option<BlockId> {
        let id = self.blocks.iter().position(|b| b.is_free())?;
        self.blocks[id].ref_count = 1;
        self.allocated_count += 1;
        Some(id)
    }

    /// Allocate a block and immediately assign it to a logical position.
    pub fn alloc_at(&mut self, layer: usize, logical_block: usize) -> Option<BlockId> {
        let id = self.alloc_block()?;
        self.table.insert(layer, logical_block, id);
        Some(id)
    }

    /// Allocate a contiguous range of blocks for tokens `[start, start + count)`
    /// and map them into the block table.
    ///
    /// Returns the number of blocks allocated.
    pub fn alloc_span(&mut self, layer: usize, start_token: usize, token_count: usize) -> usize {
        let first_block = start_token / self.block_size;
        let last_block = (start_token + token_count - 1) / self.block_size;
        let mut allocated = 0;
        for logical in first_block..=last_block {
            if self.table.get(layer, logical).is_none() {
                if let Some(id) = self.alloc_block() {
                    self.table.insert(layer, logical, id);
                    allocated += 1;
                } else {
                    break;
                }
            }
        }
        allocated
    }

    /// Free a specific block by `BlockId`.
    ///
    /// Also removes any table entries pointing to this block.
    pub fn free_block(&mut self, block_id: BlockId) {
        if block_id >= self.blocks.len() || self.blocks[block_id].is_free() {
            return;
        }
        let data = &mut self.blocks[block_id];
        data.ref_count = 0;
        data.token_count = 0;
        data.key_data.clear();
        data.value_data.clear();
        self.allocated_count = self.allocated_count.saturating_sub(1);

        // Remove from table
        self.table.map.retain(|_, &mut v| v != block_id);
    }

    /// Free all blocks assigned to a given layer.
    pub fn free_layer(&mut self, layer: usize) {
        let block_ids: Vec<BlockId> = self
            .table
            .map
            .iter()
            .filter(|((l, _), _)| *l == layer)
            .map(|(_, &id)| id)
            .collect();
        for id in block_ids {
            self.free_block(id);
        }
        self.table.clear_layer(layer);
    }

    /// Free all blocks and reset the manager.
    pub fn reset(&mut self) {
        for data in &mut self.blocks {
            data.ref_count = 0;
            data.token_count = 0;
            data.key_data.clear();
            data.value_data.clear();
        }
        self.table.map.clear();
        self.allocated_count = 0;
    }

    /// Store quantized KV data into a block.
    ///
    /// Returns `false` if the block is not allocated.
    pub fn store_block(
        &mut self,
        block_id: BlockId,
        key_data: &[u8],
        value_data: &[u8],
        token_count: usize,
    ) -> bool {
        if block_id >= self.blocks.len() || self.blocks[block_id].is_free() {
            return false;
        }
        let block = &mut self.blocks[block_id];
        block.key_data = key_data.to_vec();
        block.value_data = value_data.to_vec();
        block.token_count = token_count.min(self.block_size);
        true
    }

    /// Retrieve quantized KV data from a block.
    pub fn load_block(&self, block_id: BlockId) -> Option<(&[u8], &[u8], usize)> {
        if block_id >= self.blocks.len() || self.blocks[block_id].is_free() {
            return None;
        }
        let block = &self.blocks[block_id];
        Some((&block.key_data, &block.value_data, block.token_count))
    }

    /// Resolve a logical position to a physical block and read its data.
    pub fn read_at(&self, layer: usize, logical_block: usize) -> Option<(&[u8], &[u8], usize)> {
        let id = self.table.get(layer, logical_block)?;
        self.load_block(id)
    }

    // -- Copy-on-write ------------------------------------------------------

    /// Share a block, incrementing its reference count.
    ///
    /// Returns `false` if the block doesn't exist or is free.
    pub fn share_block(&mut self, block_id: BlockId) -> bool {
        if block_id >= self.blocks.len() || self.blocks[block_id].is_free() {
            return false;
        }
        self.blocks[block_id].ref_count += 1;
        true
    }

    /// Copy-on-write: if a block is shared, allocate a new copy for writing.
    ///
    /// Returns the (possibly new) `BlockId` to use for writing, or `None` if
    /// allocation fails.
    pub fn cow_write(&mut self, block_id: BlockId) -> Option<BlockId> {
        if block_id >= self.blocks.len() || self.blocks[block_id].is_free() {
            return None;
        }

        if self.blocks[block_id].ref_count <= 1 {
            // No sharing → write in place.
            return Some(block_id);
        }

        // Shared → allocate new block and copy data.
        let (key_data, value_data, token_count) = {
            let src = &self.blocks[block_id];
            (
                src.key_data.clone(),
                src.value_data.clone(),
                src.token_count,
            )
        };

        let new_id = self.alloc_block()?;
        let dst = &mut self.blocks[new_id];
        dst.key_data = key_data;
        dst.value_data = value_data;
        dst.token_count = token_count;

        // Decrement original's ref count.
        self.blocks[block_id].ref_count -= 1;
        Some(new_id)
    }

    // -- Queries ------------------------------------------------------------

    /// Number of free blocks remaining.
    pub fn num_free_blocks(&self) -> usize {
        self.blocks.len() - self.allocated_count
    }

    /// Total capacity of the block pool.
    pub fn total_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Number of currently allocated blocks.
    pub fn allocated_blocks(&self) -> usize {
        self.allocated_count
    }

    /// Reference count of a specific block.
    pub fn ref_count(&self, block_id: BlockId) -> u32 {
        self.blocks.get(block_id).map(|b| b.ref_count).unwrap_or(0)
    }

    /// Block size in tokens.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of entries in the block table.
    pub fn table_entries(&self) -> usize {
        self.table.len()
    }

    /// Current fragmentation ratio (0.0 = no fragmentation, 1.0 = worst).
    ///
    /// Defined as `1 - (allocated_blocks / max_possible_contiguous)`.
    /// Higher values mean the pool is more fragmented.
    pub fn fragmentation_ratio(&self) -> f64 {
        if self.allocated_count == 0 || self.blocks.is_empty() {
            return 0.0;
        }
        let max_contiguous = self
            .blocks
            .split(|b| b.is_free())
            .map(|chunk| chunk.iter().filter(|b| !b.is_free()).count())
            .max()
            .unwrap_or(0) as f64;
        1.0 - (max_contiguous / self.allocated_count as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_manager() {
        let bm = BlockManager::new(64, DEFAULT_BLOCK_SIZE, 32);
        assert_eq!(bm.total_blocks(), 64);
        assert_eq!(bm.num_free_blocks(), 64);
        assert_eq!(bm.allocated_blocks(), 0);
    }

    #[test]
    fn test_alloc_free_cycle() {
        let mut bm = BlockManager::new(64, 16, 32);
        let id = bm.alloc_block().unwrap();
        assert_eq!(bm.allocated_blocks(), 1);
        assert_eq!(bm.num_free_blocks(), 63);
        assert!(bm.load_block(id).is_some());

        bm.free_block(id);
        assert_eq!(bm.allocated_blocks(), 0);
        assert_eq!(bm.num_free_blocks(), 64);
        assert!(bm.load_block(id).is_none());
    }

    #[test]
    fn test_alloc_span() {
        let mut bm = BlockManager::new(64, 16, 4);
        let allocated = bm.alloc_span(0, 0, 33);
        // tokens 0-31 → blocks 0, 1 (32 tokens, 2 blocks)
        // Wait: 33 tokens → block 0 (0-15), block 1 (16-31), block 2 (32) → 3 blocks
        assert_eq!(allocated, 3);
        assert_eq!(bm.allocated_blocks(), 3);

        // Read back
        let (_, _, count) = bm.read_at(0, 0).unwrap();
        assert_eq!(count, 0); // no data stored yet, token_count=0
    }

    #[test]
    fn test_store_and_load_block() {
        let mut bm = BlockManager::new(64, 16, 4);
        let id = bm.alloc_block().unwrap();

        let k = vec![1u8, 2, 3, 4];
        let v = vec![5u8, 6, 7, 8];
        assert!(bm.store_block(id, &k, &v, 2));

        let (lk, lv, count) = bm.load_block(id).unwrap();
        assert_eq!(lk, &[1, 2, 3, 4]);
        assert_eq!(lv, &[5, 6, 7, 8]);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_alloc_at() {
        let mut bm = BlockManager::new(64, 16, 4);
        let id = bm.alloc_at(1, 3).unwrap();
        assert_eq!(bm.table.get(1, 3), Some(id));
    }

    #[test]
    fn test_free_layer() {
        let mut bm = BlockManager::new(64, 16, 4);
        bm.alloc_at(0, 0);
        bm.alloc_at(0, 1);
        bm.alloc_at(1, 0);
        assert_eq!(bm.allocated_blocks(), 3);

        bm.free_layer(0);
        assert_eq!(bm.allocated_blocks(), 1);
        assert!(bm.table.get(0, 0).is_none());
        assert!(bm.table.get(1, 0).is_some());
    }

    #[test]
    fn test_reset() {
        let mut bm = BlockManager::new(64, 16, 4);
        bm.alloc_at(0, 0);
        bm.alloc_at(0, 1);
        assert_eq!(bm.allocated_blocks(), 2);

        bm.reset();
        assert_eq!(bm.allocated_blocks(), 0);
        assert_eq!(bm.num_free_blocks(), 64);
    }

    #[test]
    fn test_pool_exhaustion() {
        let mut bm = BlockManager::new(2, 16, 4);
        assert!(bm.alloc_block().is_some());
        assert!(bm.alloc_block().is_some());
        assert!(bm.alloc_block().is_none()); // pool exhausted
    }

    // -- Copy-on-write -------------------------------------------------------

    #[test]
    fn test_share_block() {
        let mut bm = BlockManager::new(64, 16, 4);
        let id = bm.alloc_block().unwrap();
        assert_eq!(bm.ref_count(id), 1);
        assert!(bm.share_block(id));
        assert_eq!(bm.ref_count(id), 2);
    }

    #[test]
    fn test_cow_no_copy_when_not_shared() {
        let mut bm = BlockManager::new(64, 16, 4);
        let id = bm.alloc_block().unwrap();
        bm.store_block(id, &[1, 2], &[3, 4], 1);
        let cow_id = bm.cow_write(id).unwrap();
        assert_eq!(cow_id, id); // same block, no copy
        assert_eq!(bm.allocated_blocks(), 1);
    }

    #[test]
    fn test_cow_copy_on_shared() {
        let mut bm = BlockManager::new(64, 16, 4);
        let id = bm.alloc_block().unwrap();
        bm.store_block(id, &[1, 2], &[3, 4], 1);
        bm.share_block(id); // ref_count = 2

        let cow_id = bm.cow_write(id).unwrap();
        assert_ne!(cow_id, id); // new block allocated
        assert_eq!(bm.allocated_blocks(), 2); // both exist
        assert_eq!(bm.ref_count(id), 1); // original's share removed

        // Data was copied
        let (k, v, _) = bm.load_block(cow_id).unwrap();
        assert_eq!(k, &[1, 2]);
        assert_eq!(v, &[3, 4]);
    }

    // -- Fragmentation -------------------------------------------------------

    #[test]
    fn test_fragmentation_zero_when_empty() {
        let bm = BlockManager::new(64, 16, 4);
        assert!((bm.fragmentation_ratio() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_fragmentation_increases_with_interleaved_frees() {
        let mut bm = BlockManager::new(8, 16, 4);
        let ids: Vec<_> = (0..8).map(|_| bm.alloc_block().unwrap()).collect();

        // Free every other block
        for i in (0..8).step_by(2) {
            bm.free_block(ids[i]);
        }
        // Allocated: 4, max contiguous: 1 (every block is isolated)
        // frag_ratio = 1 - (1/4) = 0.75
        let frag = bm.fragmentation_ratio();
        assert!((frag - 0.75).abs() < 0.01);
    }

    // -- Block table ---------------------------------------------------------

    #[test]
    fn test_table_entry_count() {
        let mut bm = BlockManager::new(64, 16, 4);
        bm.alloc_at(0, 0);
        bm.alloc_at(0, 1);
        bm.alloc_at(2, 5);
        assert_eq!(bm.table_entries(), 3);
    }

    #[test]
    fn test_free_removes_table_entry() {
        let mut bm = BlockManager::new(64, 16, 4);
        let id = bm.alloc_at(0, 0).unwrap();
        assert_eq!(bm.table_entries(), 1);
        bm.free_block(id);
        assert_eq!(bm.table_entries(), 0);
    }

    // -- Edge cases ----------------------------------------------------------

    #[test]
    fn test_free_invalid_block() {
        let mut bm = BlockManager::new(64, 16, 4);
        bm.free_block(NULL_BLOCK); // should not panic
        bm.free_block(999); // should not panic
    }

    #[test]
    fn test_load_unallocated_block() {
        let bm = BlockManager::new(64, 16, 4);
        assert!(bm.load_block(0).is_none()); // block 0 is free
    }

    #[test]
    fn test_store_unallocated_block() {
        let mut bm = BlockManager::new(64, 16, 4);
        assert!(!bm.store_block(0, &[1], &[2], 1));
    }

    #[test]
    fn test_read_at_no_allocation() {
        let bm = BlockManager::new(64, 16, 4);
        assert!(bm.read_at(0, 0).is_none());
    }
}
