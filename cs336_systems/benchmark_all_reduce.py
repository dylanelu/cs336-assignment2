import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

device = "cuda"


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def distributed_demo(rank, world_size, cnt):
    setup(rank, world_size)

    try:
        data = torch.rand(cnt, dtype=torch.float32, device=device)

        # Warmup
        for _ in range(5):
            dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        size_mb = data.numel() * data.element_size() / 1_000_000
        print(f"Rank {rank}: at {size_mb:g} MB, elapsed time: {elapsed:.6f} s")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 4

    # Float32 uses 4 bytes per element. Sizes below are in decimal bytes.
    for size_bytes in (1_000_000, 10_000_000, 100_000_000, 1_000_000_000):
        cnt = size_bytes // 4
        mp.spawn(fn=distributed_demo, args=(world_size, cnt), nprocs=world_size, join=True)
