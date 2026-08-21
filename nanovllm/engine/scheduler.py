from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            None if config.prefix_cache_max_blocks == -1 else config.prefix_cache_max_blocks,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self._sequences: dict[int, Sequence] = {}
        # Read-only diagnostics for GPU E2E regression evidence. These do not
        # affect scheduling decisions or normal execution semantics.
        self.preemption_count = 0
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
        requests = self.prefix_cache_requests
        prompt_tokens = self.prefix_cache_prompt_tokens
        return {
            "kv_blocks_total": total,
            "kv_blocks_used": used,
            "kv_blocks_peak_used": self.kv_blocks_peak_used,
            "kv_usage_peak_ratio": self.kv_blocks_peak_used / total if total else 0.0,
            "preemption_count": self.preemption_count,
            "prefix_cache_requests": requests,
            "prefix_cache_hit_requests": self.prefix_cache_hit_requests,
            "prefix_cache_cached_tokens": self.prefix_cache_cached_tokens,
            "prefix_cache_hit_rate": self.prefix_cache_hit_requests / requests if requests else 0.0,
            "prefix_cache_token_hit_rate": self.prefix_cache_cached_tokens / prompt_tokens if prompt_tokens else 0.0,
            "prefix_cache_blocks": self.block_manager.prefix_cache_blocks,
            "prefix_cache_max_blocks": self.block_manager.prefix_cache_max_blocks,
            "prefix_cache_usage_ratio": (
                self.block_manager.prefix_cache_blocks / self.block_manager.prefix_cache_max_blocks
                if self.block_manager.prefix_cache_max_blocks else 0.0
            ),
            "prefix_cache_evictions": self.block_manager.prefix_cache_evictions,
        }

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        seq.status = SequenceStatus.QUEUED
        self.waiting.append(seq)
        self._sequences[seq.seq_id] = seq

    def get_sequence(self, seq_id: int):
        return self._sequences.get(seq_id)

    def cancel(self, seq_id: int, reason: str = "cancelled") -> bool:
        seq = self._sequences.get(seq_id)
        if seq is None or seq.is_terminal:
            return seq is not None
        self.waiting = deque(item for item in self.waiting if item.seq_id != seq_id)
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
        self.waiting = deque(item for item in self.waiting if item.seq_id != seq_id)
        self.running = deque(item for item in self.running if item.seq_id != seq_id)
        if seq.block_table:
            self.block_manager.deallocate(seq)
            self._record_kv_usage()
        seq.status = SequenceStatus.FAILED
        seq.finish_reason = reason
        seq.error = error
        return True

    def schedule(self) -> tuple[list[Sequence], bool]:
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
            seq = self.waiting.popleft()
            num_waiting -= 1

            if not seq.block_table:
                if seq.seq_id not in self._prefix_cache_accounted_seq_ids:
                    self._prefix_cache_accounted_seq_ids.add(seq.seq_id)
                    self.prefix_cache_requests += 1
                    self.prefix_cache_prompt_tokens += seq.num_tokens
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    self.waiting.append(seq)
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
                self._record_kv_usage()
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
                self.waiting.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                seq.status = SequenceStatus.DECODE
                self.block_manager.may_append(seq)
                self._record_kv_usage()
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        self.preemption_count += 1
        self.preemption_events.append({"seq_id": seq.seq_id, "num_tokens": len(seq), "reason": "kv_capacity"})
        seq.status = SequenceStatus.PREFILL
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self._record_kv_usage()
        self.waiting.appendleft(seq)

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
                self._record_kv_usage()
                self.running.remove(seq)
            else:
                seq.status = SequenceStatus.DECODE
        return token_events
