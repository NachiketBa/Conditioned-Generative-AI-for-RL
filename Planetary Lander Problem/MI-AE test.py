# =============================================================================
# Joint MI-AE + CVAE test script
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
# USER CONFIG
# -----------------------------------------------------------------------------
CKPT_PATH = Path(
    "C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Documents/Desktop/Journal_2025_work/Mars_lander_MIVAE_CVAE_joint/mivae_cvae_joint_distance_1000.pt"
)
DATA_FOLDER = Path(
    "C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Documents/Desktop/Journal_2025_work/cvae_process_real_noise"
)
MAT_SAVE_DIR = Path(
    "C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Documents/Desktop/Journal_2025_work/Mars_lander_MIVAE_CVAE_joint/matlab_export_distance_1000_1000_det"
)

N_TRAIN_A = 1
X_INDEX = 1
Y_INDEX = 2

GROUP_SIZE = 10
GROUP_QUERY_INDEX = 0


GROUP_STARTS = [1050, 1250, 1450, 1650, 1850,
                2050, 2250, 2450, 2650, 2850]

N_SAMPLES = 1000
SEED = 0
DT = 0.05

# Real-world vehicle parameters (Param_A)
PARAMS_A = {"m": 500.0, "g": 3.728, "l": 10.0, "c": 0.2}
# ----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class CondEncoder(nn.Module):
    def __init__(self, in_dim, z_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.fc_z = nn.Linear(hidden_dim, z_dim)

    def forward(self, c):
        h = self.net(c)
        return self.fc_z(h)


class CVAEDecoder(nn.Module):
    def __init__(self, z_dim, cond_z_dim, out_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + cond_z_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z, cond_z):
        return self.net(torch.cat([z, cond_z], dim=1))


# -----------------------------------------------------------------------------
# Load checkpoint
# -----------------------------------------------------------------------------
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

n_states         = ckpt["n_states"]
T                = ckpt["T"]
state_flat_dim   = ckpt["state_flat_dim"]
control_flat_dim = ckpt["control_flat_dim"]
total_flat_dim   = state_flat_dim + control_flat_dim
delta_flat_dim   = ckpt["delta_flat_dim"]
z1_dim           = ckpt["z1_dim"]
zS_dim           = ckpt["zS_dim"]
zx_dim           = ckpt["zx_dim"]
hidden_mi        = ckpt["hidden_mi"]
hidden_cv        = ckpt["hidden_cv"]
cond_dim_raw     = ckpt["cond_dim_raw"]

control_dim = control_flat_dim // T
# ----------------------------------------------------------------------------

A_u_mean     = ckpt["A_u_mean"].to(device)
A_u_std      = ckpt["A_u_std"].to(device)
A_ic_mean    = ckpt["A_ic_mean"].to(device)
A_ic_std     = ckpt["A_ic_std"].to(device)
A_delta_mean = ckpt["A_delta_mean"].to(device)
A_delta_std  = ckpt["A_delta_std"].to(device)

encoder1     = CondEncoder(cond_dim_raw, z1_dim, hidden_mi).to(device)
encoder2     = CondEncoder(cond_dim_raw, zS_dim, hidden_mi).to(device)
cvae_decoder = CVAEDecoder(zx_dim, z1_dim + zS_dim, delta_flat_dim, hidden_cv).to(device)

encoder1.load_state_dict(ckpt["encoder1_state_dict"])
encoder2.load_state_dict(ckpt["encoder2_state_dict"])
cvae_decoder.load_state_dict(ckpt["cvae_decoder_state_dict"])
encoder1.eval(); encoder2.eval(); cvae_decoder.eval()


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def load_traj_group(folder_path, expected_dim, start_index, group_size):
    csv_files = sorted(glob.glob(os.path.join(str(folder_path), "*.csv")))
    if start_index + group_size > len(csv_files):
        raise IndexError(f"Need files [{start_index}, {start_index + group_size}); "
                         f"only {len(csv_files)} present.")
    rows = []
    for i in range(start_index, start_index + group_size):
        arr = pd.read_csv(csv_files[i], header=None).astype(float).values.reshape(-1)
        if arr.shape[0] != expected_dim:
            raise ValueError(f"{csv_files[i]}: got {arr.shape[0]}, expected {expected_dim}")
        rows.append(torch.tensor(arr, dtype=torch.float32))
    return torch.stack(rows)


# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
for g, start in enumerate(GROUP_STARTS):
    if start < N_TRAIN_A:
        raise ValueError(f"start {start} is inside training range [0, {N_TRAIN_A}).")
    end = start + GROUP_SIZE

    flat_group = load_traj_group(DATA_FOLDER, total_flat_dim,
                                 start_index=start, group_size=GROUP_SIZE)
    group_states_flat   = flat_group[:, :state_flat_dim].to(device)
    group_controls_flat = flat_group[:, state_flat_dim:].to(device)

    group_states_3d = group_states_flat.view(GROUP_SIZE, n_states, T)
    shared_u_flat   = group_controls_flat[0:1]

    s0_query = group_states_3d[GROUP_QUERY_INDEX, :, 0].unsqueeze(0)

    u_n  = (shared_u_flat - A_u_mean)  / A_u_std
    s0_n = (s0_query      - A_ic_mean) / A_ic_std
    c_query = torch.cat([u_n, s0_n], dim=1)

    with torch.no_grad():
        z1   = encoder1(c_query)
        zshr = encoder2(c_query)
        z1_rep   = z1.expand(N_SAMPLES, -1).contiguous()
        zshr_rep = zshr.expand(N_SAMPLES, -1).contiguous()
        cond_z   = torch.cat([z1_rep, zshr_rep], dim=1)
        z_x      = torch.randn(N_SAMPLES, zx_dim, device=device)

        deltas_hat_n = cvae_decoder(z_x, cond_z)
        deltas_hat   = deltas_hat_n * A_delta_std + A_delta_mean

        deltas_hat_3d = deltas_hat.view(N_SAMPLES, n_states, T - 1)
        s0_for_integ  = s0_query.expand(N_SAMPLES, -1).unsqueeze(-1)
        srest_hat_3d  = s0_for_integ + torch.cumsum(deltas_hat_3d, dim=2)
        pred_full_3d  = torch.cat([s0_for_integ, srest_hat_3d], dim=2)

    pred_xy  = pred_full_3d[:, [X_INDEX, Y_INDEX], :]
    truth_xy = group_states_3d[:, [X_INDEX, Y_INDEX], :]

    pred_xy_mean = pred_xy.mean(dim=0)
    pred_xy_std  = pred_xy.std(dim=0)
    endpoint_err = (pred_xy_mean[:, -1] - truth_xy[:, :, -1].mean(dim=0)).norm().item()
    err_xy = truth_xy - pred_xy_mean.unsqueeze(0)
    coverage = (err_xy.abs() <= pred_xy_std.unsqueeze(0)).all(dim=1).float().mean().item()

    s0_xy_np = np.array([[s0_query[0, X_INDEX].item(),
                          s0_query[0, Y_INDEX].item()]], dtype=np.float64)
    pred_full_np  = pred_full_3d.cpu().numpy().astype(np.float64)        # N x n_states x T
    truth_full_np = group_states_3d.cpu().numpy().astype(np.float64)     # G x n_states x T
    controls_np   = shared_u_flat.reshape(control_dim, T).cpu().numpy().astype(np.float64)  # control_dim x T
    # ------------------------------------------------------------------------ <<<

    mat_dict = {
        "truth_xy":     truth_xy.cpu().numpy().astype(np.float64),
        "pred_xy":      pred_xy.cpu().numpy().astype(np.float64),
        "s0_xy":        s0_xy_np,
        "endpoint_err": float(endpoint_err),
        "coverage":     float(coverage),
        "file_start":   int(start),
        "file_end":     int(end - 1),
        "x_index":      int(X_INDEX),
        "y_index":      int(Y_INDEX),
        "pred_full":    pred_full_np,
        "truth_full":   truth_full_np,
        "controls":     controls_np,
        "state_dim":    int(n_states),
        "control_dim":  int(control_dim),
        "dt":           float(DT),
        "params_m":     PARAMS_A["m"],
        "params_g":     PARAMS_A["g"],
        "params_l":     PARAMS_A["l"],
        "params_c":     PARAMS_A["c"],
        # -------------------------------------------------------------------- <<<
    }

    out_mat = MAT_SAVE_DIR / f"group_{g:03d}_files_{start:04d}-{end - 1:04d}.mat"
    savemat(str(out_mat), mat_dict, do_compression=True)
    print(f"[Group {g}] files {start}-{end - 1}  "
          f"endpoint_err = {endpoint_err:.4f}  coverage = {coverage*100:.1f}%  "
          f"-> {out_mat.name}")

print(f"\nWrote {len(GROUP_STARTS)} group(s). Run aggregate_indices.m / "
      f"aggregate_physics_indices.m in MATLAB.")
