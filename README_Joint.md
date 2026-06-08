# Planetary-Lander Trajectory Generation — MI-AE + CVAE

Generative modeling of Planetary-lander descent trajectories. The repository contains two model families:

1. **Joint MI-AE + CVAE** — a Mutual-Information AutoEncoder that disentangles the
   *conditioning* (controls + initial state) into a **domain-private** latent and a
   **domain-shared** latent across a real and a simulated dataset, trained jointly with a
   **Conditional VAE** that generates per-step state increments (deltas).
2. **Plain CVAE** — a standalone Conditional VAE baseline that conditions directly on the
   raw initial state and controls, with no domain disentanglement.

Each model has a **training** script (produces a checkpoint) and a **testing** script
(samples trajectories and exports `.mat` files for downstream MATLAB analysis).

---

## Data format

Each trajectory is a single CSV that flattens to a fixed-length vector. The layout
constants are shared across all scripts:

| Quantity | Symbol | Value |
|----------|--------|-------|
| State dimension | `n_states` | 9 |
| Control dimension | `n_controls` | 3 |
| Horizon (timesteps) | `T` | 100 |
| Flattened states | `state_flat_dim` | 900 |
| Flattened controls | `control_flat_dim` | 300 |
| Total per trajectory | `total_flat_dim` | 1200 |

A row is read as `[states (9×100) | controls (3×100)]`. From each trajectory the scripts
derive:

- `s0` — the initial state (first timestep), shape `[N, 9]`
- `s_rest` — states for timesteps 1…T-1
- `deltas` — first differences `s_{t+1} − s_t`, shape `[N, 9×(T-1)] = [N, 891]`
- `controls` — the full control sequence

**The models predict deltas, not absolute states.** Trajectories are reconstructed at test
time by taking the cumulative sum of predicted deltas from the known `s0`.

## Conditioning and normalization

Conditioning is `c = [u_normalized, s0_normalized]`, giving `cond_dim_raw = 300 + 9 = 309`.

Normalization uses per-channel mean/std computed from the training data, with a standard-
deviation floor (`STD_FLOOR = 1e-3`) that replaces degenerate (near-constant) channels'
std with 1.0 to avoid divide-by-zero. Deltas and `s_rest` use **per-timestep** statistics.
The real-domain stats are saved into the checkpoint so the test scripts can de-normalize
generated deltas back into physical units.

---

## Model 1 — Joint MI-AE + CVAE

### MI-AE (disentanglement)

Two **deterministic** encoders map the conditioning `c` to two latents:

- `encoder1 → z1` (private / domain-unique), `z1_dim = 32`
- `encoder2 → zS` (shared across domains), `zS_dim = 32`

Domain-specific decoders reconstruct the conditioning:

- `decoderA([z1_A, zS_A]) → c_A`
- `decoderB([z1_B, zS_B]) → c_B`

Disentanglement is enforced with **distance-based** auxiliary losses rather than an
adversarial discriminator:

| Loss | Purpose | Weight |
|------|---------|--------|
| `L_share` | pull `mean/std` of `zS_A` and `zS_B` together (moment matching) | `lambda_share = 1.0` |
| `L_sep`   | push `mean(z1_A)` and `mean(z1_B)` apart (hinge with `sep_margin = 4.0`) | `lambda_sep = 1.0` |
| `L_orth`  | within-domain orthogonality of `z1` vs `zS` | `lambda_orth = 0.5` |
| `L_zS_norm` | small L2 on `zS` | `lambda_zS_norm = 1e-3` |
| `mi_pen`  | Gaussian MI penalty between `z1_A` and `z1_B`, using an EMA-blended covariance | `beta_mi = 0.1` |

### CVAE (generation)

A standard CVAE over the **normalized deltas**:

- `cvae_encoder(deltas, c) → (mu, logvar)`, latent `zx_dim = 32`, `logvar` clamped to [−10, 10]
- `cvae_decoder([zx, cond_z]) → deltas`, where `cond_z = [z1_A, zS_A]` from the MI-AE

CVAE losses: summed-MSE reconstruction, KL to a standard normal (annealed via
`kl_weight`, `kl_max_cvae = 0.05`), plus a **boundary loss** on the first delta
(`boundary_weight = 10.0`) to anchor the start of the trajectory.

### Training details

- Single `Adam` optimizer (`lr = 1e-3`) over all six modules.
- Total loss = MI-AE losses + CVAE losses, backpropagated jointly.
- Gradient clipping at `max_norm = 5.0`.
- `epochs = 1000`, `batch_size = 32`. The smaller domain (A) is tiled to match the larger
  (B) each epoch; both are independently shuffled.
- `hidden_mi = 128`, `hidden_cv = 324`.

### Diagnostics printed at the end

- MI-AE reconstruction error on `A_cond` and `B_cond`.
- **Cross reconstruction**: rebuild `A_cond` using `zS` taken from B — a lower
  cross/correct ratio indicates the shared latent is genuinely aligned across domains.
- Squared-distance separation checks for `z1` (should exceed the margin) and `zS`
  (should be ≈ 0), plus mean cosine similarity between private and shared latents.
- CVAE posterior- and prior-sample reconstruction MSE on the real deltas.

An 8-panel loss plot is shown and a checkpoint is saved containing all module weights,
hyperparameters, layout constants, and the per-domain normalization statistics.

---

## Model 2 — Plain CVAE (baseline)

A single CVAE with no domain disentanglement:

- `Encoder([s0, deltas, controls]) → (mu, logvar)`
- `Decoder([z, s0, controls]) → deltas`
- `latent_dim = 32`, `hidden_dim = 324`, `epochs = 1000`, `batch_size = 32`.
- Losses: summed-MSE reconstruction + annealed KL (`kl_max = 0.05`, `anneal_till = 300`)
  + first-step boundary loss (`boundary_weight = 10.0`).
- Trains on the first `n_train = 40` real trajectories and saves to `cvae.pth`.

---

## Testing / inference

Both test scripts share the same evaluation protocol, driven by `GROUP_STARTS` (a list of
starting file indices) and `GROUP_SIZE = 10`:

1. Load a group of held-out real trajectories (indices well outside the training range).
2. Build the query conditioning from the group's shared control sequence and a chosen
   initial state.
   - **Joint model**: encode the query into `z1` and `zS`, then draw `N_SAMPLES = 1000`
     latents `zx ~ N(0, I)` and decode deltas.
   - **Plain CVAE**: draw `z ~ N(0, I)` and decode deltas conditioned on `s0` and controls.
3. De-normalize deltas and integrate via `cumsum` from `s0` to obtain full trajectories.

### Physical parameters (`PARAMS_A`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `m` | 500.0 | mass |
| `g` | 3.728 | Mars gravity |
| `l` | 10.0  | length scale |
| `c` | 0.2   | drag / damping coefficient |


## Requirements

- Python 3.9+
- `torch`
- `numpy`
- `pandas`
- `scipy` (for `savemat`)
- `matplotlib` (training diagnostics plot)

```bash
pip install torch numpy pandas scipy matplotlib
```

A CUDA GPU is used automatically if available; otherwise the scripts fall back to CPU.
A fixed seed (`42` for training, `0` for testing) is set for reproducibility.
