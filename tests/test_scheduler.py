from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def make_scheduler(max_num_batched_tokens=6, max_num_seqs=2):
    Sequence.block_size = 4
    return Scheduler(SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=-1,
        num_kvcache_blocks=32,
        kvcache_block_size=4,
    ))


def test_single_waiting_sequence_uses_full_prefill_budget():
    scheduler = make_scheduler(max_num_batched_tokens=6, max_num_seqs=4)
    long_seq = Sequence(list(range(10)))

    scheduler.add(long_seq)
    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill
    assert scheduled == [long_seq]
    assert long_seq.num_scheduled_tokens == 6
    assert list(scheduler.waiting) == [long_seq]


def test_chunked_prefill_fairly_schedules_short_request_behind_long_prompt():
    scheduler = make_scheduler(max_num_batched_tokens=6, max_num_seqs=2)
    long_seq = Sequence(list(range(10)))
    short_seq = Sequence([100, 101])

    scheduler.add(long_seq)
    scheduler.add(short_seq)
    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill
    assert scheduled == [long_seq, short_seq]
    assert long_seq.num_scheduled_tokens == 3
    assert short_seq.num_scheduled_tokens == 2
    assert short_seq.status == SequenceStatus.RUNNING
    assert list(scheduler.running) == [short_seq]
    assert list(scheduler.waiting) == [long_seq]
