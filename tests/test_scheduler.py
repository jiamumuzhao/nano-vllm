from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler, WaitingQueue
from nanovllm.engine.sequence import Sequence, SequenceStatus


def make_scheduler(max_num_batched_tokens=6, max_num_seqs=2, scheduling_policy="fcfs"):
    Sequence.block_size = 4
    return Scheduler(SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        scheduling_policy=scheduling_policy,
        eos=-1,
        num_kvcache_blocks=32,
        kvcache_block_size=4,
        prefix_cache_max_blocks=-1,
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
    assert short_seq.status == SequenceStatus.DECODE
    assert list(scheduler.running) == [short_seq]
    assert list(scheduler.waiting) == [long_seq]


def test_preempted_sequence_returns_to_queued_state():
    scheduler = make_scheduler()
    seq = Sequence([1, 2])
    scheduler.block_manager.allocate(seq, 0)
    seq.status = SequenceStatus.DECODE

    scheduler.preempt(seq)

    assert seq.status == SequenceStatus.QUEUED
    assert seq.is_prefill
    assert not seq.block_table
    assert list(scheduler.waiting) == [seq]


def test_preemption_prefers_newer_request_with_larger_kv_footprint():
    scheduler = make_scheduler()
    older = Sequence([1, 2, 3, 4])
    newer = Sequence(list(range(12)))
    scheduler.block_manager.allocate(older, 0)
    scheduler.block_manager.allocate(newer, 0)
    scheduler.running.extend([older, newer])
    scheduler._sequence_meta[older.seq_id] = {
        "arrival_order": 0,
        "queued_step": 0,
        "admitted_step": 1,
        "preemption_count": 0,
    }
    scheduler._sequence_meta[newer.seq_id] = {
        "arrival_order": 1,
        "queued_step": 0,
        "admitted_step": 3,
        "preemption_count": 0,
    }

    assert scheduler._select_preemption_victim(older) is newer


def test_preemption_protects_a_previously_preempted_request():
    scheduler = make_scheduler()
    first = Sequence([1, 2, 3, 4])
    second = Sequence([5, 6, 7, 8])
    scheduler.running.extend([first, second])
    scheduler._sequence_meta[first.seq_id] = {
        "arrival_order": 0,
        "queued_step": 0,
        "admitted_step": 4,
        "preemption_count": 1,
    }
    scheduler._sequence_meta[second.seq_id] = {
        "arrival_order": 1,
        "queued_step": 0,
        "admitted_step": 4,
        "preemption_count": 0,
    }

    assert scheduler._select_preemption_victim(first) is second


def test_waiting_queue_compacts_stale_entries_after_cancellation_burst():
    queue = WaitingQueue()
    sequences = [Sequence([index]) for index in range(100)]
    for index, seq in enumerate(sequences):
        queue.push(seq, (0, index))
    for seq in sequences[:90]:
        queue.remove(seq)

    assert len(queue) == 10
    assert len(queue._heap) <= 2 * len(queue) + 64
    assert [seq.seq_id for seq in queue] == [seq.seq_id for seq in sequences[90:]]


def test_latency_policy_prefers_shorter_estimated_work():
    scheduler = make_scheduler(scheduling_policy="latency")
    long_seq = Sequence(list(range(20)))
    short_seq = Sequence([1, 2])
    scheduler.add(long_seq)
    scheduler.add(short_seq)

    assert scheduler._pop_waiting() is short_seq


def test_throughput_policy_prefers_smaller_kv_footprint():
    scheduler = make_scheduler(scheduling_policy="throughput")
    large_seq = Sequence(list(range(15)))
    small_seq = Sequence([1, 2, 3])
    scheduler.add(large_seq)
    scheduler.add(small_seq)

    assert scheduler._pop_waiting() is small_seq


def test_scheduler_snapshot_exposes_policy():
    scheduler = make_scheduler(scheduling_policy="latency")
    assert scheduler.get_metrics_snapshot()["scheduling_policy"] == "latency"


def test_preemption_cooldown_prefers_a_different_victim():
    scheduler = make_scheduler()
    scheduler._scheduler_step = 5
    recent = Sequence([1, 2, 3, 4])
    alternative = Sequence([5, 6, 7, 8])
    scheduler.running.extend([recent, alternative])
    scheduler._sequence_meta[recent.seq_id] = {
        "arrival_order": 0,
        "queued_step": 0,
        "admitted_step": 1,
        "preemption_count": 1,
        "last_preempt_step": 4,
    }
    scheduler._sequence_meta[alternative.seq_id] = {
        "arrival_order": 1,
        "queued_step": 0,
        "admitted_step": 1,
        "preemption_count": 0,
        "last_preempt_step": -100,
    }

    assert scheduler._select_preemption_victim(recent) is alternative


def test_preemption_records_recompute_tokens():
    scheduler = make_scheduler()
    seq = Sequence([1, 2, 3, 4])
    scheduler.block_manager.allocate(seq, 0)
    scheduler.preempt(seq)

    snapshot = scheduler.get_metrics_snapshot()
    assert snapshot["preemption_recompute_tokens"] == len(seq)
    assert snapshot["preemption_count"] == 1
