from collections import OrderedDict, deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        prefix_cache_max_blocks: int | None = None,
    ):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        if prefix_cache_max_blocks is None:
            prefix_cache_max_blocks = num_blocks
        if prefix_cache_max_blocks < 0:
            raise ValueError("prefix_cache_max_blocks must be non-negative or None")
        self.prefix_cache_max_blocks = min(prefix_cache_max_blocks, num_blocks)
        # Only inactive hashed blocks are tracked here. Active blocks remain
        # protected by ref_count and are never eligible for eviction.
        self.prefix_cache_lru: OrderedDict[int, None] = OrderedDict()
        self.prefix_cache_evictions = 0

    @property
    def prefix_cache_blocks(self) -> int:
        return len(self.prefix_cache_lru)

    def _uncache_block(self, block_id: int, clear_hash: bool = False):
        self.prefix_cache_lru.pop(block_id, None)
        if clear_hash:
            block = self.blocks[block_id]
            if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
                del self.hash_to_block_id[block.hash]
            block.hash = -1
            block.token_ids = []

    def _cache_block(self, block_id: int):
        block = self.blocks[block_id]
        if block.ref_count != 0 or block.hash == -1:
            return
        self.prefix_cache_lru.pop(block_id, None)
        self.prefix_cache_lru[block_id] = None
        while len(self.prefix_cache_lru) > self.prefix_cache_max_blocks:
            evicted_id, _ = self.prefix_cache_lru.popitem(last=False)
            evicted = self.blocks[evicted_id]
            assert evicted.ref_count == 0
            if evicted.hash != -1 and self.hash_to_block_id.get(evicted.hash) == evicted_id:
                del self.hash_to_block_id[evicted.hash]
            evicted.hash = -1
            evicted.token_ids = []
            self.prefix_cache_evictions += 1

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self._uncache_block(block_id, clear_hash=True)
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                self._uncache_block(block_id)
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
                self._cache_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            if block.hash != -1 and block.hash != h and self.hash_to_block_id.get(block.hash) == block.block_id:
                del self.hash_to_block_id[block.hash]
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
            if block.ref_count == 0:
                self._cache_block(block.block_id)
