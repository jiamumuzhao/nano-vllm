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
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

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
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    self.waiting.append(seq)
                    continue
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                self.block_manager.allocate(seq, num_cached_blocks)
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            seq.num_scheduled_tokens = min(num_tokens, token_budget)
            num_batched_tokens += seq.num_scheduled_tokens
            scheduled_seqs.append(seq)

            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
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
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
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
                continue
            seq.append_token(token_id)
            finished = (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens
            token_events.append((seq.seq_id, token_id, finished))
            if finished:
                # print(f"Sequence {seq.seq_id} finished with reason: {'EOS' if token_id == self.eos else 'max_tokens'}")
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        return token_events
