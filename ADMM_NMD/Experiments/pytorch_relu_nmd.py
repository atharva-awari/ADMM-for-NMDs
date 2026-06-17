"""
pytorch_relu_nmd.py
PyTorch/Adam baseline for ReLU-NMD with Frobenius loss.

Loads the same data matrix X and initializations (W0_scaled, H0) saved
by Figure4_extended.m to ensure an identical starting point across all
three methods (ADMM, CD, PyTorch/Adam).

Usage:
  1. Run Figure5.m first (saves inits to figures/inits/)
  2. python pytorch_relu_nmd.py

Outputs:
  - figures/PyTorch_time_CBCL.txt   (tikz-compatible: x1 y1)
  - figures/PyTorch_time_MNIST.txt
  - figures/table_PyTorch_CBCL.txt  (LaTeX table row)
  - figures/table_PyTorch_MNIST.txt
  - Console summary with mean +/- std over 10 runs
"""

import os
import time
import numpy as np
import torch
import scipy.io as sio

# ====================================================================
#  CONFIGURATION (must match Figure4_extended.m)
# ====================================================================
NRUNS       = 10
TIME_BUDGET = 10.0    # seconds, same as ADMM/CD
LR          = 1e-2    # Adam learning rate
DEVICE      = 'cpu'

DATASETS = {
    'CBCL':  {'rank': 10},
    'MNIST': {'rank': 32},
}

os.makedirs('figures', exist_ok=True)

# ====================================================================
#  PyTorch ReLU-NMD with Adam
# ====================================================================
def pytorch_relu_nmd(X_np, W0_np, H0_np, time_budget):
    """
    Solve  min_{W,H}  ||X - max(0, WH)||_F / ||X||_F
    using PyTorch + Adam optimizer.

    Args:
        X_np:    data matrix (m x n), numpy
        W0_np:   initial W (m x r), numpy (already scaled)
        H0_np:   initial H (r x n), numpy
        time_budget: max wall-clock time in seconds

    Returns:
        errors:  list of relative errors at recorded time points
        times:   list of elapsed times at recorded points
        final_err: final relative error
        n_iters: total number of Adam iterations performed
    """
    X_norm = np.linalg.norm(X_np, 'fro')

    # Convert to torch tensors
    X_t = torch.tensor(X_np, dtype=torch.float64, device=DEVICE)
    W   = torch.tensor(W0_np, dtype=torch.float64, device=DEVICE,
                       requires_grad=True)
    H   = torch.tensor(H0_np, dtype=torch.float64, device=DEVICE,
                       requires_grad=True)
    X_norm_t = torch.tensor(X_norm, dtype=torch.float64, device=DEVICE)

    optimizer = torch.optim.Adam([W, H], lr=LR)

    errors = []
    times  = []
    start  = time.time()

    # Record initial error (before any update)
    with torch.no_grad():
        T0 = torch.relu(W @ H)
        init_err = (torch.norm(X_t - T0, 'fro') / X_norm_t).item()
    errors.append(init_err)
    times.append(0.0)

    n_iters = 0
    next_record = 0.05  # record every 0.05 seconds

    while True:
        elapsed = time.time() - start
        if elapsed >= time_budget:
            break

        optimizer.zero_grad()
        T = torch.relu(W @ H)
        loss = torch.norm(X_t - T, 'fro') / X_norm_t
        loss.backward()
        optimizer.step()
        n_iters += 1

        # Record at regular time intervals (avoids huge output)
        if elapsed >= next_record:
            errors.append(loss.item())
            times.append(elapsed)
            next_record += 0.05

    # Record final error
    with torch.no_grad():
        T = torch.relu(W @ H)
        final_err = (torch.norm(X_t - T, 'fro') / X_norm_t).item()
    errors.append(final_err)
    times.append(time.time() - start)

    return np.array(errors), np.array(times), final_err, n_iters


# ====================================================================
#  MAIN
# ====================================================================
for name, info in DATASETS.items():
    r = info['rank']
    print(f"\n{'='*60}")
    print(f"  Dataset: {name}  |  Rank: {r}  |  Runs: {NRUNS}")
    print(f"{'='*60}")

    # Load data matrix saved by MATLAB
    X_mat = sio.loadmat(f'figures/inits/X_{name}.mat')
    X = X_mat['X'].astype(np.float64)
    m, n = X.shape
    print(f"  Matrix size: {m} x {n}")

    final_errors = []
    all_errors   = []
    all_times    = []
    all_iters    = []

    for run in range(1, NRUNS + 1):
        print(f"\n--- Run {run}/{NRUNS} ---")

        # Load initialization saved by MATLAB (identical to ADMM/CD)
        init_file = f'figures/inits/init_{name}_run{run}.mat'
        if not os.path.exists(init_file):
            print(f"  ERROR: {init_file} not found. Run Figure4_extended.m first.")
            exit(1)

        init_data = sio.loadmat(init_file)
        W0 = init_data['W0_scaled'].astype(np.float64)
        H0 = init_data['H0'].astype(np.float64)

        errors, times, final_err, n_iters = pytorch_relu_nmd(
            X, W0, H0, TIME_BUDGET
        )

        final_errors.append(final_err)
        all_errors.append(errors)
        all_times.append(times)
        all_iters.append(n_iters)

        print(f"  PyTorch/Adam: final err = {final_err:.4f}, "
              f"time = {times[-1]:.2f} s, iters = {n_iters}")

    # Summary
    final_errors = np.array(final_errors)
    all_iters    = np.array(all_iters)

    print(f"\n{'-'*60}")
    print(f"  SUMMARY: {name} (rank {r}, {NRUNS} runs)")
    print(f"{'-'*60}")
    print(f"  PyTorch/Adam | {final_errors.mean():.4f} +/- "
          f"{final_errors.std():.4f}  | {all_iters.mean():.0f} avg iters")
    print(f"{'-'*60}")

    # Select median run for plotting
    sorted_idx = np.argsort(final_errors)
    med_idx    = sorted_idx[NRUNS // 2]
    err_plot   = all_errors[med_idx]
    time_plot  = all_times[med_idx]

    # Export for tikz (time-based)
    out_file = f"figures/PyTorch_time_{name}.txt"
    with open(out_file, 'w') as f:
        f.write("x1 y1\n")
        for t, e in zip(time_plot, err_plot):
            f.write(f"{t:.6e} {e:.6e}\n")
    print(f"  Saved: {out_file}")

    # Export LaTeX table row
    table_file = f"figures/table_PyTorch_{name}.txt"
    with open(table_file, 'w') as f:
        f.write(f"% LaTeX table row for {name} (rank {r}, {NRUNS} runs)\n")
        f.write(f"PyTorch/Adam & ${final_errors.mean():.4f} "
                f"\\pm {final_errors.std():.4f}$ \\\\\n")
    print(f"  Saved: {table_file}")

print("\n*** All PyTorch experiments complete. ***")
