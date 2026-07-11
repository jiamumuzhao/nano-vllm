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
