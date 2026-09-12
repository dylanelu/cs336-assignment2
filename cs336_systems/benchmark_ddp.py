"""Run with: uv run python -m cs336_systems.benchmark_ddp"""

import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

# Uncomment exactly one implementation to benchmark.
# from cs336_systems.ddp import NaiveDDP as DDP
# from cs336_systems.ddp import FlattenedDDP as DDP
from cs336_systems.ddp import OverlappedDDP as DDP


def ddp_benchmarking(model, optimizer, inputs, targets, warmup_steps=5, steps=10):
    """Time full steps and post-backward synchronization; print mean and std."""
    if warmup_steps < 0 or steps < 1:
        raise ValueError("warmup_steps must be nonnegative and steps must be positive")

    def synchronize():
        if inputs.is_cuda:
            torch.cuda.synchronize(inputs.device)

    model.train()
    if inputs.is_cuda:
        sync_start = torch.cuda.Event(enable_timing=True)
        sync_end = torch.cuda.Event(enable_timing=True)
    timings = []
    for iteration in range(warmup_steps + steps):
        dist.barrier()  # Align ranks outside the timed region.
        synchronize()
        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()

        # Events time the GPU stream without blocking between backward and sync.
        # This includes packing/copying. With overlap, it measures only the tail
        # after backward, not communication already overlapped with computation.
        if inputs.is_cuda:
            sync_start.record(torch.cuda.current_stream(inputs.device))
        else:
            sync_start_time = time.perf_counter()
        model.finish_gradient_synchronization()
        if inputs.is_cuda:
            sync_end.record(torch.cuda.current_stream(inputs.device))
        else:
            sync_seconds = time.perf_counter() - sync_start_time

        optimizer.step()
        synchronize()
        end = time.perf_counter()

        if iteration >= warmup_steps:
            if inputs.is_cuda:
                sync_seconds = sync_start.elapsed_time(sync_end) / 1000
            timings.append((end - start, sync_seconds))
        del logits, loss

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, timings)
    if dist.get_rank() == 0:
        # Average ranks for each iteration, then measure variability across steps.
        step_times = [statistics.mean(t[0] for t in ranks) for ranks in zip(*gathered)]
        sync_times = [statistics.mean(t[1] for t in ranks) for ranks in zip(*gathered)]
        mean_step = statistics.mean(step_times)
        mean_sync = statistics.mean(sync_times)
        std_step = statistics.stdev(step_times) if steps > 1 else 0.0
        std_sync = statistics.stdev(sync_times) if steps > 1 else 0.0
        print(f"Rank-averaged timings over {steps} steps (mean +/- std):")
        print(f"Full step (wall time): {mean_step * 1000:.3f} +/- {std_step * 1000:.3f} ms")
        print(f"Post-backward gradient sync: {mean_sync * 1000:.3f} +/- {std_sync * 1000:.3f} ms")
        print(f"Post-backward sync / full step: {mean_sync / mean_step:.2%}")
        print("Sync includes packing/copying; with overlap, only the post-backward tail is measured.")


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
        model = DDP(model)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        # Global batch of four, split across ranks. Reuse it for every step.
        generator = torch.Generator().manual_seed(43)
        tokens = torch.randint(10_000, (4, 513), generator=generator)
        local_tokens = tokens.chunk(world_size, dim=0)[rank].to(device)
        inputs = local_tokens[:, :-1].contiguous()
        targets = local_tokens[:, 1:].contiguous()
        ddp_benchmarking(model, optimizer, inputs, targets)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This benchmark requires two allocated GPUs.")
    print(f"{DDP.__name__}: XL, FP32, 2 GPUs, global batch 4, context 512; 5 warmup + 10 measured steps")
    mp.spawn(_worker, args=(2,), nprocs=2, join=True)
