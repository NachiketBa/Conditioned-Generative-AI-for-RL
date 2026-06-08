# Trajectory Prediction with RNN informed CVAE and MI-CVAE

Two trained models, two testing scripts, one goal: roll out plausible state trajectories from a learned latent space conditioned on state and control input.

---

## What's in this repo

| File | Description |
|---|---|
| `CVRNN_teacher_forced.py` | Trains the RNN informed CVAE on real data |
| `MIVRNN_teacher_forced.py` | Trains the RNN informed MI-CVAE with shared/unique latent structure |
| `CVRNN_teacher_forced_test.py` | Loads decoder and rolls out trajectory samples |
| `MIVRNN_teacher_forced_test.py` | Loads decoder specific to real data and rolls out trajectory samples |

---

## Models

### RNN informed CVAE (`stepwise_cvae.pth`)

A conditional variational autoencoder trained to predict next states step by step. The encoder compresses full trajectories (state + control) into a latent code `z`, and the decoder unrolls predictions autoregressively using that code.

- State dim: 9, Control dim: 3
- Latent dim: 32, Hidden dim: 64
- KL annealing over first 300 epochs, max weight 0.05
- Trained on N real-noise trajectories

## RNN informed MI-CVAE (`stepwise_mivae.pth`)

A two-encoder CVAE trained on real (`A`) and simulated (`B`) trajectories simultaneously. The setup splits the latent space into two parts:

- `z1` — domain-specific (different prior per dataset)
- `z2` — shared across domains

After a warmup period, mutual information between `z1_A` and `z1_B` is maximized to push the two domains apart in that subspace. Reconstruction happens through separate decoders for each domain.

- z1 dim: 32, z2 dim: 32, Hidden dim: 324
- MI warmup: 50 epochs, beta (MI weight): 20.0
- N real + Ns simulated trajectories

---

## Running the test scripts

Both scripts load a checkpoint, pull one test trajectory from disk, and plot several sampled rollouts against ground truth.

## Output

Each script generates a 2D position plot (`X` vs `Y`):
- Black line = ground truth trajectory
- Dashed colored lines = 5 samples drawn from the prior

The spread of sampled trajectories reflects model uncertainty. Tight clustering = low variance in the learned latent space; wide spread = higher uncertainty or a looser prior fit.

## Notes

- The MI-CVAE test only uses `decoderA` (real noise decoder).
- Normalization stats are saved in the checkpoint, so no need to recompute them at test time.
- Both decoders are autoregressive: prediction errors accumulate over long horizons. Short rollouts will be tighter than long ones.
