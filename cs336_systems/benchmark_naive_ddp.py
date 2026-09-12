"""Run with: uv run python -m cs336_systems.benchmark_naive_ddp"""

import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import NaiveDDP


def naive_ddp_benchmarking(model, optimizer, inputs, targets, warmup_steps=5, steps=10):
    """Time full training steps and gradient communication, then print rank averages."""
    if warmup_steps < 0 or steps < 1:
        raise ValueError("warmup_steps must be nonnegative and steps must be positive")

    def synchronize():
        if inputs.is_cuda:
            torch.cuda.synchronize(inputs.device)

    model.train()
    timings = []
    for iteration in range(warmup_steps + steps):
        dist.barrier()  # Align ranks outside the timed region.
        synchronize()
        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        synchronize()  # Finish backward before measuring communication.

        communication_start = time.perf_counter()
        model.finish_gradient_synchronization()
        synchronize()
        communication_end = time.perf_counter()

        optimizer.step()
        synchronize()
        end = time.perf_counter()

        if iteration >= warmup_steps:
            timings.append((end - start, communication_end - communication_start))
        del logits, loss

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, timings)
    if dist.get_rank() == 0:
        all_timings = [timing for rank_timings in gathered for timing in rank_timings]
        step_times = [step for step, communication in all_timings]
        communication_times = [communication for step, communication in all_timings]
        mean_step = statistics.mean(step_times)
        mean_communication = statistics.mean(communication_times)
        print(f"Mean step: {mean_step * 1000:.3f} ms")
        print(f"Mean gradient communication: {mean_communication * 1000:.3f} ms")
        print(f"Communication fraction: {mean_communication / mean_step:.2%}")


def _worker(rank, world_size):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method="tcp://127.0.0.1:29501", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(42)
        model = BasicsTransformerLM(
            vocab_size=10_000,
            context_length=512,
            d_model=2560,
            d_ff=10240,
            num_layers=32,
            num_heads=32,
        ).to(device=device, dtype=torch.float32)
        model = NaiveDDP(model)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        # Global batch of four, split across ranks. Reuse it for every step.
        generator = torch.Generator().manual_seed(43)
        tokens = torch.randint(10_000, (4, 513), generator=generator)
        local_tokens = tokens.chunk(world_size, dim=0)[rank].to(device)
        inputs = local_tokens[:, :-1].contiguous()
        targets = local_tokens[:, 1:].contiguous()
        naive_ddp_benchmarking(model, optimizer, inputs, targets)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This benchmark requires two allocated GPUs.")
    print("XL, FP32, 2 GPUs, global batch 4, context 512; 5 warmup + 10 measured steps")
    mp.spawn(_worker, args=(2,), nprocs=2, join=True)
