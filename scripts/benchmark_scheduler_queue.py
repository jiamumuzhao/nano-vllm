#!/usr/bin/env python3
"""CPU-only benchmark for the scheduler waiting-queue hot path."""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import deque

from nanovllm.engine.scheduler import WaitingQueue
from nanovllm.engine.sequence import Sequence


def percentile(values, ratio):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * ratio))
    return ordered[index]


def benchmark_heap(size, operations, samples):
    sequences = [Sequence([index]) for index in range(size)]
    timings = []
    for _ in range(samples):
        queue = WaitingQueue()
        for index, seq in enumerate(sequences):
            queue.push(seq, (0, index))
        start = time.perf_counter_ns()
        for index in range(operations):
            seq = queue.pop()
            queue.push(seq, (index // size + 1, seq.seq_id))
        elapsed = time.perf_counter_ns() - start
        timings.append(elapsed / operations / 1000)
    return timings


def benchmark_scan(size, operations, samples):
    sequences = [Sequence([index]) for index in range(size)]
    timings = []
    for _ in range(samples):
        queue = deque(sequences)
        start = time.perf_counter_ns()
        for _ in range(operations):
            seq = min(queue, key=lambda item: item.seq_id)
            queue.remove(seq)
            queue.append(seq)
        elapsed = time.perf_counter_ns() - start
        timings.append(elapsed / operations / 1000)
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 128, 512, 2048])
    parser.add_argument("--operations", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()

    print("queue_size,implementation,p50_us,p95_us,mean_us,ops_per_sec")
    for size in args.sizes:
        for name, runner in (("heap", benchmark_heap), ("scan", benchmark_scan)):
            values = runner(size, args.operations, args.samples)
            mean_us = statistics.mean(values)
            print(
                f"{size},{name},{percentile(values, 0.50):.3f},"
                f"{percentile(values, 0.95):.3f},{mean_us:.3f},"
                f"{1_000_000 / mean_us:.1f}"
            )


if __name__ == "__main__":
    main()
