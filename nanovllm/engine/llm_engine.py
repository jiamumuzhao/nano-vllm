import atexit
import os
import uuid
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from multiprocessing.shared_memory import SharedMemory


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        config.shared_memory_name = f"nanovllm_{os.getpid()}_{uuid.uuid4().hex}"
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        self.ready_events = []
        self._exited = False
        self.shutdown_diagnostic = None
        ctx = mp.get_context("spawn")
        try:
            for i in range(1, config.tensor_parallel_size):
                event = ctx.Event()
                ready_event = ctx.Event()
                process = ctx.Process(target=ModelRunner, args=(config, i, event, ready_event))
                process.start()
                self.ps.append(process)
                self.events.append(event)
                self.ready_events.append(ready_event)
            self.model_runner = ModelRunner(config, 0, self.events, self.ready_events, self.ps)
            self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
            config.eos = self.tokenizer.eos_token_id
            self.scheduler = Scheduler(config)
            self._atexit_callback = self.exit
            atexit.register(self._atexit_callback)
        except BaseException:
            # ModelRunner initializes NCCL before model/KV setup.  If setup
            # fails, tear down the parent group and all spawned workers before
            # exposing the original exception to the caller.
            self._destroy_parent_group()
            self._unlink_shared_memory()
            self._reap_workers(config.tp_shutdown_timeout)
            self.ps.clear()
            raise

    def _destroy_parent_group(self):
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    def _unlink_shared_memory(self):
        name = getattr(getattr(self, "config", None), "shared_memory_name", None)
        if not name:
            return
        try:
            shm = SharedMemory(name=name)
        except FileNotFoundError:
            return
        try:
            shm.unlink()
        finally:
            shm.close()

    def _worker_diagnostics(self):
        return [
            {"rank": i + 1, "pid": p.pid, "is_alive": p.is_alive(), "exitcode": p.exitcode,
             "ready": self.ready_events[i].is_set() if i < len(self.ready_events) else False}
            for i, p in enumerate(self.ps)
        ]

    def _reap_workers(self, timeout):
        for p in self.ps:
            p.join(timeout=timeout)
        stuck = [p for p in self.ps if p.is_alive()]
        if stuck:
            for p in stuck:
                p.terminate()
            for p in stuck:
                p.join(timeout=timeout)
        return stuck

    def exit(self):
        if self._exited:
            return
        self._exited = True
        callback = getattr(self, "_atexit_callback", None)
        if callback is not None:
            atexit.unregister(callback)
            self._atexit_callback = None
        failure = None
        try:
            if hasattr(self, "model_runner"):
                self.model_runner.call("exit")
        except BaseException as exc:
            failure = exc
        finally:
            # Engine shutdown must release scheduler-owned KV block references
            # even when a runner call failed. This is cleanup only; it does not
            # change scheduling or inference behavior.
            if hasattr(self, "scheduler"):
                manager = self.scheduler.block_manager
                for seq in (*self.scheduler.waiting, *self.scheduler.running):
                    if seq.block_table:
                        manager.deallocate(seq)
                self.scheduler.waiting.clear()
                self.scheduler.running.clear()
            if hasattr(self, "model_runner"):
                del self.model_runner
            stuck = self._reap_workers(self.config.tp_shutdown_timeout)
            self._destroy_parent_group()
            self._unlink_shared_memory()
            if stuck:
                self.shutdown_diagnostic = (
                    f"TP worker shutdown timed out after {self.config.tp_shutdown_timeout}s: "
                    f"workers={self._worker_diagnostics()}, endpoint={self.config.distributed_init_method}"
                )
            elif failure is not None:
                self.shutdown_diagnostic = f"TP shutdown failed: {type(failure).__name__}: {failure}"
        if self.shutdown_diagnostic:
            raise RuntimeError(self.shutdown_diagnostic) from failure

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        outputs, num_tokens, _ = self.step_with_events()
        return outputs, num_tokens

    def step_with_events(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        token_events = self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens, token_events

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        total_prefill_time = total_decode_time = 0.
        total_prefill_tokens = total_decode_tokens = 0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            elapsed = perf_counter() - t
            if num_tokens > 0:
                total_prefill_time += elapsed
                total_prefill_tokens += num_tokens
            else:
                total_decode_time += elapsed
                total_decode_tokens += -num_tokens
            prefill_speed = total_prefill_tokens / total_prefill_time if total_prefill_time else 0.
            decode_speed = total_decode_tokens / total_decode_time if total_decode_time else 0.
            pbar.set_postfix({
                "Prefill": f"{int(prefill_speed)}tok/s",
                "Decode": f"{int(decode_speed)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        print(f"\n{'='*40}")
        print(f"  Prefill: {total_prefill_tokens:>6} tokens in {total_prefill_time:.2f}s  ({total_prefill_tokens/total_prefill_time:.1f} tok/s)")
        print(f"  Decode:  {total_decode_tokens:>6} tokens in {total_decode_time:.2f}s  ({total_decode_tokens/total_decode_time:.1f} tok/s)")
        print(f"{'='*40}")
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
