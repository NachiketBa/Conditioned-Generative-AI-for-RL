# =============================================================================
# CVAE test script

# =============================================================================

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import savemat

# -----------------------------------------------------------------------------

CKPT_PATH = Path("cvae.pth")
A_path = "C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Documents/Desktop/Journal_2025_work/cvae_process_real_noise"
MAT_SAVE_DIR = Path(
    "C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Documents/Desktop/Journal_2025_work/matlab_export_cvae_40_1000"
)

GROUP_STARTS = [1050, 1250, 1450, 1650, 1850,
                2050, 2250, 2450, 2650, 2850]

GROUP_SIZE  = 10
N_SAMPLES   = 1000
SEED        = 0

X_INDEX = 1
Y_INDEX = 2
DT = 0.05
PARAMS_A = {"m": 500.0, "g": 3.728, "l": 10.0, "c": 0.2}
# ---------------------------------------------------------------------------- <<<

# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

checkpoint = torch.load(str(CKPT_PATH), map_location=device)

state_dim        = checkpoint["state_dim"]
control_dim      = checkpoint["control_dim"]
state_flat_dim   = checkpoint["state_flat_dim"]
state_rest_dim   = checkpoint["state_rest_dim"]
delta_flat_dim   = checkpoint["delta_flat_dim"]
control_flat_dim = checkpoint["control_flat_dim"]
latent_dim       = checkpoint["latent_dim"]
hidden_dim       = checkpoint["hidden_dim"]
T                = checkpoint["T"]

s0_mean    = checkpoint["s0_mean"].to(device)
s0_std     = checkpoint["s0_std"].to(device)
delta_mean = checkpoint["delta_mean"].to(device)
delta_std  = checkpoint["delta_std"].to(device)
ctrl_mean  = checkpoint["ctrl_mean"].to(device)
ctrl_std   = checkpoint["ctrl_std"].to(device)


class Decoder(nn.Module):
    def __init__(self, latent_dim, state_dim, delta_flat_dim, control_flat_dim, hidden_dim):
        super().__init__()
        in_dim = latent_dim + state_dim + control_flat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, delta_flat_dim),
        )

    def forward(self, z, s0, u):
        return self.net(torch.cat([z, s0, u], dim=1))


decoder = Decoder(latent_dim, state_dim, delta_flat_dim,
                  control_flat_dim, hidden_dim).to(device)
decoder.load_state_dict(checkpoint["decoder_state_dict"])
decoder.eval()


def load_traj_group(folder_path, file_pattern, expected_dim, start_index, group_size=10):
    csv_files = sorted(glob.glob(os.path.join(folder_path, file_pattern)))
    if start_index + group_size > len(csv_files):
        raise IndexError(f"Need files [{start_index}, {start_index + group_size}); "
                         f"only {len(csv_files)} present.")
    rows = []
    for i in range(start_index, start_index + group_size):
        df = pd.read_csv(csv_files[i], header=None).astype(float)
        arr = df.values.reshape(-1)
        if arr.shape[0] != expected_dim:
            raise ValueError(f"Expected {expected_dim} entries, got {arr.shape[0]} in {csv_files[i]}")
        rows.append(torch.tensor(arr, dtype=torch.float32))
    return torch.stack(rows)


total_flat_dim = state_flat_dim + control_flat_dim
# -----------------------------------------------------------------------------
MAT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
removed = 0
for old_mat in MAT_SAVE_DIR.glob("*.mat"):
    try:
        old_mat.unlink()
        removed += 1
    except OSError as e:
        print(f"  could not remove {old_mat.name}: {e}")
print(f"Cleared {removed} stale .mat file(s) from {MAT_SAVE_DIR}")
print(f"Writing {len(GROUP_STARTS)} group(s).")

# -----------------------------------------------------------------------------
for g, group_start in enumerate(GROUP_STARTS):
    end = group_start + GROUP_SIZE

    flat_group = load_traj_group(A_path, "*.csv", total_flat_dim,
                                 start_index=group_start, group_size=GROUP_SIZE)

    true_states_flat   = flat_group[:, :state_flat_dim]
    true_controls_flat = flat_group[:, state_flat_dim:]

    shared_ctrl_flat = true_controls_flat[0:1]
    u_norm = (shared_ctrl_flat.to(device) - ctrl_mean) / ctrl_std

    true_states = true_states_flat.reshape(GROUP_SIZE, state_dim, T)

    s0_true = true_states[0, :, 0].unsqueeze(0)
    s0_norm = (s0_true.to(device) - s0_mean) / s0_std

    pred_xy_list = []
    pred_full_list = []
    for i in range(N_SAMPLES):
        with torch.no_grad():
            z = torch.randn(1, latent_dim).to(device)
            deltas_pred_norm = decoder(z, s0_norm, u_norm)
            deltas_pred = deltas_pred_norm * delta_std + delta_mean
            deltas_pred_3d = deltas_pred.reshape(1, state_dim, T - 1)
            s_rest_3d = s0_true.to(device).unsqueeze(-1) + torch.cumsum(deltas_pred_3d, dim=2)
            s_full_3d = torch.cat([s0_true.to(device).unsqueeze(-1), s_rest_3d], dim=2)
            pred_traj = s_full_3d[0].cpu()
            pred_xy_list.append(pred_traj[[X_INDEX, Y_INDEX], :].numpy())
            pred_full_list.append(pred_traj.numpy())

    pred_xy_np  = np.stack(pred_xy_list, axis=0).astype(np.float64)
    truth_xy_np = true_states[:, [X_INDEX, Y_INDEX], :].numpy().astype(np.float64)
    s0_xy_np = np.array([[s0_true[0, X_INDEX].item(),
                          s0_true[0, Y_INDEX].item()]], dtype=np.float64)
    pred_full_np  = np.stack(pred_full_list, axis=0).astype(np.float64)   # N x state_dim x T
    truth_full_np = true_states.numpy().astype(np.float64)               # G x state_dim x T
    controls_np   = shared_ctrl_flat.reshape(control_dim, T).cpu().numpy().astype(np.float64)  # control_dim x T
    # ------------------------------------------------------------------------ <<<

    pred_xy_mean = pred_xy_np.mean(axis=0)
    pred_xy_std  = pred_xy_np.std(axis=0)
    endpoint_err = float(np.linalg.norm(pred_xy_mean[:, -1]
                                        - truth_xy_np[:, :, -1].mean(axis=0)))
    err_xy   = truth_xy_np - pred_xy_mean[None, :, :]
    within   = (np.abs(err_xy) <= pred_xy_std[None, :, :]).all(axis=1)
    coverage = float(within.mean())

    mat_dict = {
        "truth_xy":     truth_xy_np,
        "pred_xy":      pred_xy_np,
        "s0_xy":        s0_xy_np,
        "endpoint_err": endpoint_err,
        "coverage":     coverage,
        "file_start":   int(group_start),
        "file_end":     int(end - 1),
        "x_index":      int(X_INDEX),
        "y_index":      int(Y_INDEX),
        # >>> ADDED for physics index ----------------------------------------
        "pred_full":    pred_full_np,    # N x state_dim x T  (all channels)
        "truth_full":   truth_full_np,   # G x state_dim x T  (all channels)
        "controls":     controls_np,     # control_dim x T    (frozen, physical)
        "state_dim":    int(state_dim),
        "control_dim":  int(control_dim),
        "dt":           float(DT),
        "params_m":     PARAMS_A["m"],
        "params_g":     PARAMS_A["g"],
        "params_l":     PARAMS_A["l"],
        "params_c":     PARAMS_A["c"],
        # -------------------------------------------------------------------- <<<
    }

    out_mat = MAT_SAVE_DIR / f"group_{g:03d}_files_{group_start:04d}-{end - 1:04d}.mat"
    savemat(str(out_mat), mat_dict, do_compression=True)
    print(f"[Group {g}] files {group_start}-{end - 1}  "
          f"endpoint_err = {endpoint_err:.4f}  coverage = {coverage*100:.1f}%  "
          f"-> {out_mat.name}")

print(f"\nWrote {len(GROUP_STARTS)} group(s). Run aggregate_indices.m / "
      f"aggregate_physics_indices.m in MATLAB.")