from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


def make_sequence(tokens: list[int]) -> Sequence:
    Sequence.block_size = 256
    return Sequence(tokens)


def test_reuses_complete_prefix_blocks_after_deallocation():
    manager = BlockManager(num_blocks=4, block_size=256)
    first = make_sequence(list(range(300)))
    assert manager.can_allocate(first) == 0
    manager.allocate(first, 0)
    first.num_scheduled_tokens = len(first)
    manager.hash_blocks(first)
    manager.deallocate(first)

    second = make_sequence(list(range(256)) + [999])
    assert manager.can_allocate(second) == 1
    manager.allocate(second, 1)
    assert second.num_cached_tokens == 256
    assert len(second.block_table) == 2


def test_deallocate_returns_unshared_blocks_to_free_pool():
    manager = BlockManager(num_blocks=3, block_size=256)
    sequence = make_sequence(list(range(256)))
    manager.allocate(sequence, 0)
    assert len(manager.free_block_ids) == 2
    manager.deallocate(sequence)
    assert len(manager.free_block_ids) == 3
    assert not manager.used_block_ids
    assert not sequence.block_table


def cache_sequence(manager: BlockManager, tokens: list[int]):
    sequence = make_sequence(tokens)
    assert manager.can_allocate(sequence) == 0
    manager.allocate(sequence, 0)
    sequence.num_scheduled_tokens = len(sequence)
    manager.hash_blocks(sequence)
    return sequence


def test_prefix_cache_capacity_evicts_oldest_inactive_block():
    manager = BlockManager(num_blocks=4, block_size=256, prefix_cache_max_blocks=1)
    first = cache_sequence(manager, list(range(300)))
    first_hash = manager.blocks[first.block_table[0]].hash
    manager.deallocate(first)

    second = cache_sequence(manager, list(range(1000, 1300)))
    second_hash = manager.blocks[second.block_table[0]].hash
    manager.deallocate(second)

    assert manager.prefix_cache_blocks == 1
    assert manager.prefix_cache_evictions == 1
    assert first_hash not in manager.hash_to_block_id
    assert manager.hash_to_block_id[second_hash] in manager.free_block_ids


def test_prefix_cache_capacity_does_not_evict_active_blocks():
    manager = BlockManager(num_blocks=4, block_size=256, prefix_cache_max_blocks=1)
    active = cache_sequence(manager, list(range(300)))
    active_hash = manager.blocks[active.block_table[0]].hash

    cached = cache_sequence(manager, list(range(1000, 1300)))
    cached_hash = manager.blocks[cached.block_table[0]].hash
    manager.deallocate(cached)

    assert manager.hash_to_block_id[active_hash] == active.block_table[0]
    assert cached_hash in manager.hash_to_block_id
    manager.deallocate(active)


def test_allocates_plain_free_blocks_before_cached_lru_blocks():
    manager = BlockManager(num_blocks=3, block_size=256, prefix_cache_max_blocks=3)
    cached = cache_sequence(manager, list(range(300)))
    cached_block_id = cached.block_table[0]
    manager.deallocate(cached)

    plain = make_sequence(list(range(1000, 1256)))
    assert manager.can_allocate(plain) == 0
    manager.allocate(plain, 0)

    assert plain.block_table[0] != cached_block_id
    assert cached_block_id in manager.hash_to_block_id
    manager.deallocate(plain)

    # Consume every ordinary free block while the cached block remains idle.
    plain_a = make_sequence(list(range(2000, 2256)))
    plain_b = make_sequence(list(range(3000, 3256)))
    manager.allocate(plain_a, 0)
    manager.allocate(plain_b, 0)
    assert manager.free_plain_blocks == 0

    fallback = make_sequence(list(range(4000, 4256)))
    manager.allocate(fallback, 0)
    assert fallback.block_table[0] == cached_block_id
    assert cached_block_id not in manager.hash_to_block_id


def test_free_queue_and_cache_invariants_hold_after_eviction():
    manager = BlockManager(num_blocks=3, block_size=256, prefix_cache_max_blocks=1)
    first = cache_sequence(manager, list(range(300)))
    manager.deallocate(first)
    assert manager.check_free_block_invariants()

    second = cache_sequence(manager, list(range(1000, 1300)))
    manager.deallocate(second)
    assert manager.check_free_block_invariants()
    assert manager.prefix_cache_blocks == 1

    third = make_sequence(list(range(2000, 2256)))
    manager.allocate(third, 0)
    assert manager.check_free_block_invariants()
