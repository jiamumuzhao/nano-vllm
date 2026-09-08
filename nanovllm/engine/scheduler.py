from collections import deque
import heapq

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class WaitingQueue:
    """Priority queue for waiting sequences with a stable inspection order."""

    def __init__(self):
        self._heap = []
        self._entries = {}

    def _maybe_compact(self):
        # Lazy removal keeps cancellation O(1). Rebuild only after stale
        # entries dominate the heap, preserving amortized O(log n) behavior.
        if len(self._heap) > 2 * len(self._entries) + 64:
            self._heap = [
                (key, seq_id)
                for seq_id, (key, _) in self._entries.items()
            ]
            heapq.heapify(self._heap)

    def push(self, seq: Sequence, key: tuple[int, ...]):
        self._entries[seq.seq_id] = (key, seq)
        heapq.heappush(self._heap, (key, seq.seq_id))
        self._maybe_compact()

    def pop(self) -> Sequence:
        while self._heap:
            key, seq_id = heapq.heappop(self._heap)
            entry = self._entries.get(seq_id)
            if entry is not None and entry[0] == key:
                del self._entries[seq_id]
                return entry[1]
        raise IndexError("pop from empty waiting queue")

    def remove(self, seq: Sequence):
        self._entries.pop(seq.seq_id, None)
        self._maybe_compact()

    def clear(self):
        self._heap.clear()
        self._entries.clear()

    def __bool__(self):
        return bool(self._entries)

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        entries = sorted(self._entries.values(), key=lambda item: (item[0], item[1].seq_id))
        return iter(seq for _, seq in entries)


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.scheduling_policy = getattr(config, "scheduling_policy", "fcfs")
        self.preemption_cooldown_steps = getattr(config, "preemption_cooldown_steps", 2)
        self.max_preemptions_per_step = getattr(config, "max_preemptions_per_step", 1)
        self.eos = config.eos
        if self.scheduling_policy not in ("fcfs", "throughput", "latency"):
            raise ValueError(
                "scheduling_policy must be one of: fcfs, throughput, latency"
            )
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            None if config.prefix_cache_max_blocks == -1 else config.prefix_cache_max_blocks,
        )
        self._use_fifo_waiting = self.scheduling_policy == "fcfs"
        self.waiting = deque() if self._use_fifo_waiting else WaitingQueue()
        self.running: deque[Sequence] = deque()
        self._sequences: dict[int, Sequence] = {}
        self._sequence_meta: dict[int, dict[str, int]] = {}
        self._arrival_counter = 0
        self._scheduler_step = 0
        # Read-only diagnostics for GPU E2E regression evidence. These do not
        # affect scheduling decisions or normal execution semantics.
        self.preemption_count = 0
        self.preemption_recompute_tokens = 0
        self.preemption_cooldown_skips = 0
        self._preemptions_this_step = 0
        self.preemption_events = []
        self.kv_blocks_peak_used = 0
        self.prefix_cache_requests = 0
        self.prefix_cache_hit_requests = 0
        self.prefix_cache_cached_tokens = 0
        self.prefix_cache_prompt_tokens = 0
        self._prefix_cache_accounted_seq_ids = set()
        self._prefix_cache_resolved_seq_ids = set()

    def _record_kv_usage(self):
        self.kv_blocks_peak_used = max(self.kv_blocks_peak_used, len(self.block_manager.used_block_ids))

    def get_metrics_snapshot(self) -> dict:
        """Return read-only scheduler/KV diagnostics without changing scheduling."""
        total = len(self.block_manager.blocks)
        used = len(self.block_manager.used_block_ids)
        prefix_cache_blocks = getattr(self.block_manager, "prefix_cache_blocks", 0)
        prefix_cache_max_blocks = getattr(self.block_manager, "prefix_cache_max_blocks", total)
        prefix_cache_evictions = getattr(self.block_manager, "prefix_cache_evictions", 0)
        requests = self.prefix_cache_requests
        prompt_tokens = self.prefix_cache_prompt_tokens
        return {
            "kv_blocks_total": total,
            "kv_blocks_used": used,
            "kv_blocks_free": len(getattr(self.block_manager, "free_block_ids", ())),
            "kv_plain_blocks_free": getattr(self.block_manager, "free_plain_blocks", 0),
            "kv_cached_blocks_free": getattr(self.block_manager, "free_cached_blocks", 0),
            "kv_blocks_peak_used": self.kv_blocks_peak_used,
            "kv_usage_peak_ratio": self.kv_blocks_peak_used / total if total else 0.0,
            "preemption_count": self.preemption_count,
            "preemption_recompute_tokens": getattr(self, "preemption_recompute_tokens", 0),
            "preemption_cooldown_skips": getattr(self, "preemption_cooldown_skips", 0),
            "preemption_cooldown_steps": getattr(self, "preemption_cooldown_steps", 0),
            "max_preemptions_per_step": getattr(self, "max_preemptions_per_step", 1),
            "prefix_cache_requests": requests,
            "prefix_cache_hit_requests": self.prefix_cache_hit_requests,
            "prefix_cache_cached_tokens": self.prefix_cache_cached_tokens,
            "prefix_cache_hit_rate": self.prefix_cache_hit_requests / requests if requests else 0.0,
            "prefix_cache_token_hit_rate": self.prefix_cache_cached_tokens / prompt_tokens if prompt_tokens else 0.0,
            "prefix_cache_blocks": prefix_cache_blocks,
            "prefix_cache_max_blocks": prefix_cache_max_blocks,
            "prefix_cache_usage_ratio": (
                prefix_cache_blocks / prefix_cache_max_blocks
                if prefix_cache_max_blocks else 0.0
            ),
            "prefix_cache_evictions": prefix_cache_evictions,
            "scheduling_policy": getattr(self, "scheduling_policy", "fcfs"),
        }

    def is_finished(self):
        return not self.waiting and not self.running

    def _meta(self, seq: Sequence) -> dict[str, int]:
        return self._sequence_meta.setdefault(seq.seq_id, {
            "arrival_order": self._arrival_counter,
            "queued_step": self._scheduler_step,
            "admitted_step": self._scheduler_step,
            "preemption_count": 0,
            "last_preempt_step": -10**9,
            "recompute_tokens": 0,
        })

    def _pop_waiting(self) -> Sequence:
        # Aging priority is maintained by the heap, so admission is O(log n)
        # instead of scanning every waiting sequence on each scheduler step.
        return self.waiting.popleft() if self._use_fifo_waiting else self.waiting.pop()

    def _waiting_key(self, seq: Sequence) -> tuple[int, ...]:
        meta = self._meta(seq)
        queued_step = meta["queued_step"]
        arrival_order = meta["arrival_order"]
        estimated_work = seq.num_tokens + seq.max_tokens
        if self.scheduling_policy == "latency":
            # Shortest estimated work first reduces mean queueing delay. The
            # age and arrival fields make ties deterministic.
            return (estimated_work, queued_step, arrival_order)
        if self.scheduling_policy == "throughput":
            estimated_blocks = (estimated_work + self.block_size - 1) // self.block_size
            # Prefer requests with a smaller KV footprint so more requests can
            # coexist in a batch. Work and age break ties deterministically.
            return (estimated_blocks, estimated_work, queued_step, arrival_order)
        return (queued_step, arrival_order)

    def _enqueue_waiting(self, seq: Sequence, refresh_age: bool = False):
        meta = self._meta(seq)
        if refresh_age:
            meta["queued_step"] = self._scheduler_step
        if self._use_fifo_waiting:
            self.waiting.append(seq)
        else:
            self.waiting.push(seq, self._waiting_key(seq))

    def _select_preemption_victim(self, current: Sequence) -> Sequence | None:
        candidates = list(self.running)
        if not candidates:
            return None
        eligible = [
            seq for seq in candidates
            if self._scheduler_step - self._meta(seq).get("last_preempt_step", -10**9)
            >= self.preemption_cooldown_steps
        ]
        if eligible:
            candidates = eligible
        else:
            self.preemption_cooldown_skips += len(candidates)
        # Prefer a request that has not already been preempted, is newer
        # (less recomputation progress to discard), and releases more blocks.
        # The preemption count is first to prevent repeatedly preempting the
        # same request; block footprint is a capacity-aware tie breaker.
        return max(
            candidates,
            key=lambda seq: (
                -self._meta(seq)["preemption_count"],
                -(
                    self._scheduler_step
                    - self._meta(seq)["admitted_step"]
                ),
                len(seq.block_table),
                -self._meta(seq)["arrival_order"],
            ),
        )

    def add(self, seq: Sequence):
        seq.status = SequenceStatus.QUEUED
        self._sequence_meta[seq.seq_id] = {
            "arrival_order": self._arrival_counter,
            "queued_step": self._scheduler_step,
            "admitted_step": self._scheduler_step,
            "preemption_count": 0,
        }
        self._arrival_counter += 1
        self._enqueue_waiting(seq)
        self._sequences[seq.seq_id] = seq

    def get_sequence(self, seq_id: int):
        return self._sequences.get(seq_id)

    def cancel(self, seq_id: int, reason: str = "cancelled") -> bool:
        seq = self._sequences.get(seq_id)
        if seq is None or seq.is_terminal:
            return seq is not None
        self.waiting.remove(seq)
        self.running = deque(item for item in self.running if item.seq_id != seq_id)
        if seq.block_table:
            self.block_manager.deallocate(seq)
            self._record_kv_usage()
        seq.status = SequenceStatus.CANCELLED
        seq.finish_reason = reason
        return True

    def fail(self, seq_id: int, reason: str = "error", error: str | None = None) -> bool:
        seq = self._sequences.get(seq_id)
        if seq is None or seq.is_terminal:
            return seq is not None
        self.waiting.remove(seq)
        self.running = deque(item for item in self.running if item.seq_id != seq_id)
        if seq.block_table:
            self.block_manager.deallocate(seq)
            self._record_kv_usage()
        seq.status = SequenceStatus.FAILED
        seq.finish_reason = reason
        seq.error = error
        return True

    def schedule(self) -> tuple[list[Sequence], bool]:
        self._scheduler_step += 1
        self._preemptions_this_step = 0
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        # Scan each waiting sequence at most once per scheduling step. This lets
        # long prompts make progress without monopolizing the full token budget.
        num_waiting = len(self.waiting)
        while self.waiting and num_waiting and len(scheduled_seqs) < self.max_num_seqs:
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            seq_slots = min(num_waiting, self.max_num_seqs - len(scheduled_seqs))
            token_budget = max(1, remaining // seq_slots)
            seq = self._pop_waiting()
            self._meta(seq)["admitted_step"] = self._scheduler_step
            num_waiting -= 1

            if not seq.block_table:
                if seq.seq_id not in self._prefix_cache_accounted_seq_ids:
                    self._prefix_cache_accounted_seq_ids.add(seq.seq_id)
                    self.prefix_cache_requests += 1
                    self.prefix_cache_prompt_tokens += seq.num_tokens
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    self._enqueue_waiting(seq, refresh_age=True)
                    continue
                # can_allocate() is the source of truth for prefix-cache
                # reuse.  Only count complete cached blocks and cap tokens by
                # the actual prompt length.
                prompt_tokens = seq.num_tokens
                cached_tokens = min(num_cached_blocks * self.block_size, prompt_tokens)
                if seq.seq_id not in self._prefix_cache_resolved_seq_ids:
                    self._prefix_cache_resolved_seq_ids.add(seq.seq_id)
                    self.prefix_cache_cached_tokens += cached_tokens
                    if num_cached_blocks > 0:
                        self.prefix_cache_hit_requests += 1
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                self.block_manager.allocate(seq, num_cached_blocks)
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            seq.status = SequenceStatus.PREFILL

            seq.num_scheduled_tokens = min(num_tokens, token_budget)
            num_batched_tokens += seq.num_scheduled_tokens
            scheduled_seqs.append(seq)

            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.DECODE
                self.running.append(seq)
            else:
                self._enqueue_waiting(seq, refresh_age=True)

        if scheduled_seqs:
            self._record_kv_usage()
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self._preemptions_this_step >= self.max_preemptions_per_step:
                    self.preempt(seq)
                    break
                victim = self._select_preemption_victim(seq)
                if victim is not None:
                    self.running.remove(victim)
                    self.preempt(victim)
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                seq.status = SequenceStatus.DECODE
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        self._record_kv_usage()
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        meta = self._meta(seq)
        meta["preemption_count"] += 1
        meta["last_preempt_step"] = self._scheduler_step
        meta["recompute_tokens"] = meta.get("recompute_tokens", 0) + len(seq)
        meta["queued_step"] = self._scheduler_step
        self.preemption_count += 1
        self.preemption_recompute_tokens += len(seq)
        self._preemptions_this_step += 1
        self.preemption_events.append({
            "seq_id": seq.seq_id,
            "num_tokens": len(seq),
            "kv_blocks": len(seq.block_table),
            "preemption_count": meta["preemption_count"],
            "reason": "kv_capacity",
        })
        seq.status = SequenceStatus.QUEUED
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self._record_kv_usage()
        self._enqueue_waiting(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        token_events = []
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            # if(is_prefill):
            #     print(f"Prefill - Sequence {seq.seq_id}: Block Num={len(seq.block_table)}, Total={seq.num_tokens}")
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                seq.status = SequenceStatus.PREFILL
                continue
            seq.append_token(token_id)
            finished = (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens
            token_events.append((seq.seq_id, token_id, finished))
            if finished:
                # print(f"Sequence {seq.seq_id} finished with reason: {'EOS' if token_id == self.eos else 'max_tokens'}")
                seq.status = SequenceStatus.FINISHED
                seq.finish_reason = "stop"
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
            else:
                seq.status = SequenceStatus.DECODE
        return token_events
