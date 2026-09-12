## Setup

2 x B200

## Results

### Naive DDP

Full step (wall time): 588.416 +/- 0.159 ms
Post-backward gradient sync: 33.009 +/- 0.122 ms
Post-backward sync / full step: 5.61%

### Flattened DDP

Flattend grad and param tensors so that we only need one broadcast/all_reduce per init or update

Full step (wall time): 588.208 +/- 0.444 ms
Post-backward gradient sync: 32.685 +/- 0.154 ms
Post-backward sync / full step: 5.56%

### Overlapped DDP

We async all_reduce the gradient as soon as it's ready, allowing us to overlap communication with computation (backwards pass)

Full step (wall time): 564.572 +/- 0.311 ms
Post-backward gradient sync: 0.241 +/- 0.004 ms
Post-backward sync / full step: 0.04%