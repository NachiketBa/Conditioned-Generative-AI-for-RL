# =============================================================================
# Joint training of MI-AE (on conditioning c = [u, s_0]) + CVAE (on real states)

import os
import glob
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Device and seed
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# Data paths
# -----------------------------------------------------------------------------
A_path = Path(
    "C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Documents/Desktop/Journal_2025_work/cvae_process_real_noise"
)
B_path = Path(
    "C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Documents/Desktop/Journal_2025_work/cvae_process_sim_noise"
)

# -----------------------------------------------------------------------------
# Layout constants
# -----------------------------------------------------------------------------
n_states   = 9
n_controls = 3
T          = 100

state_flat_dim   = n_states * T
state_rest_dim   = n_states * (T - 1)
delta_flat_dim   = n_states * (T - 1)
control_flat_dim = n_controls * T
total_flat_dim   = state_flat_dim + control_flat_dim

cond_dim_raw = control_flat_dim + n_states
STD_FLOOR = 1e-3

# -----------------------------------------------------------------------------
# Load CSVs
# -----------------------------------------------------------------------------
def load_flat_csvs(folder_path, expected_dim, pattern="*.csv"):
    csv_files = sorted(glob.glob(os.path.join(str(folder_path), pattern)))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSVs found in {folder_path}")
    rows = []
    for f in csv_files:
        arr = pd.read_csv(f, header=None).astype(float).values.reshape(-1)
        if arr.shape[0] != expected_dim:
            raise ValueError(f"{f}: got {arr.shape[0]}, expected {expected_dim}")
        rows.append(torch.tensor(arr, dtype=torch.float32))
    return torch.stack(rows)


print("Loading real (A) ...")
A_flat = load_flat_csvs(A_path, total_flat_dim)
print(f"  A: {A_flat.shape}")
print("Loading sim (B) ...")
B_flat = load_flat_csvs(B_path, total_flat_dim)
print(f"  B: {B_flat.shape}")

A_flat = A_flat[:1000]
B_flat = B_flat[:2000]

# -----------------------------------------------------------------------------
# Split each row into (s_0, s_rest, deltas, controls)
# -----------------------------------------------------------------------------
def split_traj(X):
    N = X.size(0)
    states = X[:, :state_flat_dim]
    controls = X[:, state_flat_dim:state_flat_dim + control_flat_dim]
    states_3d = states.reshape(N, n_states, T)
    s0 = states_3d[:, :, 0]
    s_rest_3d = states_3d[:, :, 1:]
    s_rest = s_rest_3d.reshape(N, state_rest_dim)
    deltas_3d = states_3d[:, :, 1:] - states_3d[:, :, :-1]
    deltas = deltas_3d.reshape(N, delta_flat_dim)
    return s0, s_rest, deltas, controls


A_s0, A_srest, A_deltas, A_u = split_traj(A_flat)
B_s0, B_srest, B_deltas, B_u = split_traj(B_flat)

# -----------------------------------------------------------------------------
# Normalization (with std clamp for degenerate channels)
# -----------------------------------------------------------------------------
def stats_clamped(t):
    mean = t.mean(dim=0, keepdim=True)
    std = t.std(dim=0, keepdim=True)
    std = torch.where(std < STD_FLOOR, torch.ones_like(std), std)
    return mean, std


A_s0_mean, A_s0_std = stats_clamped(A_s0)

A_deltas_3d = A_deltas.reshape(A_deltas.size(0), n_states, T - 1)
A_delta_mean_3d = A_deltas_3d.mean(dim=0, keepdim=True)
A_delta_std_3d  = A_deltas_3d.std(dim=0, keepdim=True)
A_delta_std_3d  = torch.where(A_delta_std_3d < STD_FLOOR,
                              torch.ones_like(A_delta_std_3d), A_delta_std_3d)
A_delta_mean = A_delta_mean_3d.reshape(1, delta_flat_dim)
A_delta_std  = A_delta_std_3d.reshape(1, delta_flat_dim)

A_srest_3d = A_srest.reshape(A_srest.size(0), n_states, T - 1)
A_srest_mean_3d = A_srest_3d.mean(dim=0, keepdim=True)
A_srest_std_3d  = A_srest_3d.std(dim=0, keepdim=True)
A_srest_std_3d  = torch.where(A_srest_std_3d < STD_FLOOR,
                              torch.ones_like(A_srest_std_3d), A_srest_std_3d)
A_srest_mean = A_srest_mean_3d.reshape(1, state_rest_dim)
A_srest_std  = A_srest_std_3d.reshape(1, state_rest_dim)

A_u_mean,  A_u_std  = stats_clamped(A_u)
A_ic_mean, A_ic_std = stats_clamped(A_s0)
B_u_mean,  B_u_std  = stats_clamped(B_u)
B_ic_mean, B_ic_std = stats_clamped(B_s0)

A_deltas_n  = (A_deltas - A_delta_mean) / A_delta_std

A_u_n        = (A_u  - A_u_mean)  / A_u_std
A_s0_n_mivae = (A_s0 - A_ic_mean) / A_ic_std
B_u_n        = (B_u  - B_u_mean)  / B_u_std
B_s0_n_mivae = (B_s0 - B_ic_mean) / B_ic_std

A_cond = torch.cat([A_u_n, A_s0_n_mivae], dim=1)
B_cond = torch.cat([B_u_n, B_s0_n_mivae], dim=1)
assert A_cond.size(1) == cond_dim_raw

print(f"A_cond: {A_cond.shape}, B_cond: {B_cond.shape}")
print(f"A_deltas: {A_deltas_n.shape}  (CVAE target: normalized deltas)")

A_cond     = A_cond.to(device)
B_cond     = B_cond.to(device)
A_deltas_n = A_deltas_n.to(device)

N_A = A_cond.size(0)
N_B = B_cond.size(0)

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
z1_dim     = 32
zS_dim     = 32
hidden_mi  = 128

zx_dim     = 32
hidden_cv  = 324

# Distance-based disentanglement weights
lambda_share   = 1.0     # pull zS_A and zS_B together (moment matching)
lambda_sep     = 1.0     # push z1_A and z1_B apart (hinge on squared dist)
sep_margin     = 4.0     # margin for the hinge; ~= squared distance target
lambda_orth    = 0.5     # within-domain z1 vs zS orthogonality
lambda_zS_norm = 1e-3    # tiny L2 on zS

beta_mi        = 0.1     # EMA-MI penalty weight (kept)
lambda_kl_cvae = 1.0
kl_max_cvae    = 0.05

boundary_weight = 10.0

mi_warmup_epochs    = 50
cross_warmup_epochs = 20

epochs     = 1000
batch_size = 32

# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
class CondEncoder(nn.Module):
    def __init__(self, in_dim, z_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fc_z = nn.Linear(hidden_dim, z_dim)

    def forward(self, c):
        h = self.net(c)
        return self.fc_z(h)


class CondDecoder(nn.Module):
    def __init__(self, z_in_dim, out_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class CVAEEncoder(nn.Module):
    def __init__(self, delta_dim, raw_cond_dim, z_dim, hidden_dim):
        super().__init__()
        in_dim = delta_dim + raw_cond_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, deltas, raw_cond):
        h = torch.cat([deltas, raw_cond], dim=1)
        h = self.net(h)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)
        return mu, logvar


class CVAEDecoder(nn.Module):
    def __init__(self, z_dim, cond_z_dim, out_dim, hidden_dim):
        super().__init__()
        in_dim = z_dim + cond_z_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z, cond_z):
        return self.net(torch.cat([z, cond_z], dim=1))


def reparam(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# -----------------------------------------------------------------------------
# Distance-based disentanglement losses
# -----------------------------------------------------------------------------
def shared_alignment_loss(zS_A, zS_B):
    mean_A = zS_A.mean(dim=0)
    mean_B = zS_B.mean(dim=0)
    std_A  = zS_A.std(dim=0)
    std_B  = zS_B.std(dim=0)
    return ((mean_A - mean_B) ** 2).sum() + ((std_A - std_B) ** 2).sum()


def unique_separation_loss(z1_A, z1_B, margin):
    diff = z1_A.mean(dim=0) - z1_B.mean(dim=0)
    sq_dist = (diff ** 2).sum()
    return F.relu(margin - sq_dist)


def orthogonality_loss(z_private, z_shared):
    dot = (z_private * z_shared).sum(dim=1)
    return (dot ** 2).mean()


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
encoder1   = CondEncoder(cond_dim_raw, z1_dim, hidden_mi).to(device)   # deterministic
encoder2   = CondEncoder(cond_dim_raw, zS_dim, hidden_mi).to(device)   # deterministic
decoderA   = CondDecoder(z1_dim + zS_dim, cond_dim_raw, hidden_mi).to(device)
decoderB   = CondDecoder(z1_dim + zS_dim, cond_dim_raw, hidden_mi).to(device)

cvae_cond_dim = z1_dim + zS_dim
cvae_encoder  = CVAEEncoder(delta_flat_dim, cond_dim_raw, zx_dim, hidden_cv).to(device)
cvae_decoder  = CVAEDecoder(zx_dim, cvae_cond_dim, delta_flat_dim, hidden_cv).to(device)

all_params = (
    list(encoder1.parameters())
    + list(encoder2.parameters())
    + list(decoderA.parameters())
    + list(decoderB.parameters())
    + list(cvae_encoder.parameters())
    + list(cvae_decoder.parameters())
)
optimizer = optim.Adam(all_params, lr=1e-3)

# -----------------------------------------------------------------------------
ema_decay = 0.99
ema_mean  = None
ema_cov   = None


def update_ema(new_mean, new_cov):
    global ema_mean, ema_cov
    if ema_mean is None:
        ema_mean = new_mean.detach()
        ema_cov  = new_cov.detach()
    else:
        ema_mean = ema_decay * ema_mean + (1 - ema_decay) * new_mean.detach()
        ema_cov  = ema_decay * ema_cov  + (1 - ema_decay) * new_cov.detach()


def mi_loss(z1_A, z1_B):
    batch = torch.cat([z1_A, z1_B], dim=1)
    mean  = batch.mean(dim=0)
    centered = batch - mean
    cov = (centered.T @ centered) / max(batch.size(0) - 1, 1)

    update_ema(mean, cov)
    cov_blend = 0.5 * cov + 0.5 * ema_cov

    d = z1_A.size(1)
    cov_joint = cov_blend + 1e-4 * torch.eye(2 * d, device=z1_A.device)
    cov_AA = cov_joint[:d, :d]
    cov_BB = cov_joint[d:, d:]

    _, ld_j = torch.slogdet(cov_joint)
    _, ld_A = torch.slogdet(cov_AA)
    _, ld_B = torch.slogdet(cov_BB)
    mi = 0.5 * (ld_A + ld_B - ld_j)
    return F.relu(mi)


def kl_standard_normal(mu, logvar):
    return 0.5 * (-logvar + logvar.exp() + mu.pow(2) - 1.0).sum(dim=1)


def kl_weight(epoch, total, k_max):
    return min(k_max, (epoch / max(total, 1)) * k_max)


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
max_len     = max(N_A, N_B)
n_repeat_A  = max_len // N_A + 1
n_repeat_B  = max_len // N_B + 1
num_batches = max_len // batch_size
print(f"\nEpoch length: {num_batches} batches  (max_len={max_len})")
print(f"A is tiled {n_repeat_A}x then sliced to {max_len}")

hist = {
    "total": [], "mi_recA": [], "mi_recB": [],
    "share": [], "sep": [], "orth": [], "zS_norm": [],
    "mi_mi": [],
    "cvae_rec": [], "cvae_kl": [], "cvae_boundary": [],
}

for epoch in range(epochs):
    w_kl_cvae = kl_weight(epoch, total=300, k_max=kl_max_cvae)

    A_cond_tiled   = A_cond.repeat(n_repeat_A, 1)[:max_len]
    A_deltas_tiled = A_deltas_n.repeat(n_repeat_A, 1)[:max_len]
    B_cond_tiled   = B_cond.repeat(n_repeat_B, 1)[:max_len]

    idx_A = torch.randperm(max_len, device=device)
    idx_B = torch.randperm(max_len, device=device)

    x_A_cond   = A_cond_tiled[idx_A]
    x_A_deltas = A_deltas_tiled[idx_A]
    x_B_cond   = B_cond_tiled[idx_B]

    sums = {k: 0.0 for k in hist}

    encoder1.train(); encoder2.train()
    decoderA.train(); decoderB.train()
    cvae_encoder.train(); cvae_decoder.train()

    for b in range(num_batches):
        sl = slice(b * batch_size, (b + 1) * batch_size)

        c_A      = x_A_cond[sl]
        deltas_A = x_A_deltas[sl]
        c_B      = x_B_cond[sl]

        bs = c_A.size(0)

        # ============ MI forward (DETERMINISTIC encoders) ============
        z1_A = encoder1(c_A)
        z1_B = encoder1(c_B)

        c_cat = torch.cat([c_A, c_B], dim=0)
        z2 = encoder2(c_cat)
        z_shared_A = z2[:bs]
        z_shared_B = z2[bs:]

        c_hat_A = decoderA(torch.cat([z1_A, z_shared_A], dim=1))
        c_hat_B = decoderB(torch.cat([z1_B, z_shared_B], dim=1))
        mi_rec_A = F.mse_loss(c_hat_A, c_A, reduction="sum")
        mi_rec_B = F.mse_loss(c_hat_B, c_B, reduction="sum")

        # ===== Distance-based disentanglement =====
        L_share   = shared_alignment_loss(z_shared_A, z_shared_B)
        L_sep     = unique_separation_loss(z1_A, z1_B, margin=sep_margin)
        L_orth    = orthogonality_loss(z1_A, z_shared_A) \
                  + orthogonality_loss(z1_B, z_shared_B)
        L_zS_norm = (z_shared_A.pow(2).mean() + z_shared_B.pow(2).mean()) * 0.5

        mi_pen = mi_loss(z1_A, z1_B)

        # ============ CVAE forward ============
        cond_z_for_cvae = torch.cat([z1_A, z_shared_A], dim=1)

        mu_x, logvar_x = cvae_encoder(deltas_A, c_A)
        z_x = reparam(mu_x, logvar_x)
        deltas_hat = cvae_decoder(z_x, cond_z_for_cvae)

        cvae_rec = F.mse_loss(deltas_hat, deltas_A, reduction="sum")

        deltas_hat_3d = deltas_hat.reshape(bs, n_states, T - 1)
        deltas_A_3d   = deltas_A.reshape(bs, n_states, T - 1)
        cvae_boundary = F.mse_loss(deltas_hat_3d[:, :, 0],
                                   deltas_A_3d[:, :, 0],
                                   reduction="sum")

        cvae_kl = kl_standard_normal(mu_x, logvar_x).sum() / zx_dim

        # ============ Total loss============
        loss_mi_total = (
            mi_rec_A + mi_rec_B
            + lambda_share   * L_share
            + lambda_sep     * L_sep
            + lambda_orth    * L_orth
            + lambda_zS_norm * L_zS_norm
            + beta_mi        * mi_pen
        )
        loss_cvae_total = cvae_rec + cvae_kl + boundary_weight * cvae_boundary
        loss = loss_mi_total + loss_cvae_total

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
        optimizer.step()

        sums["total"]         += loss.item()
        sums["mi_recA"]       += mi_rec_A.item()
        sums["mi_recB"]       += mi_rec_B.item()
        sums["share"]         += L_share.item()
        sums["sep"]           += L_sep.item()
        sums["orth"]          += L_orth.item()
        sums["zS_norm"]       += L_zS_norm.item()
        sums["mi_mi"]         += mi_pen.item()
        sums["cvae_rec"]      += cvae_rec.item()
        sums["cvae_kl"]       += cvae_kl.item()
        sums["cvae_boundary"] += cvae_boundary.item()

    for k in sums:
        hist[k].append(sums[k] / num_batches)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Ep {epoch+1:4d}/{epochs} | "
            f"Total {hist['total'][-1]:.4f} | "
            f"recA {hist['mi_recA'][-1]:.4f} recB {hist['mi_recB'][-1]:.4f} | "
            f"share {hist['share'][-1]:.4f} sep {hist['sep'][-1]:.4f} "
            f"orth {hist['orth'][-1]:.4f} zS {hist['zS_norm'][-1]:.4f} | "
            f"mi {hist['mi_mi'][-1]:.4f} | "
            f"CVAE[rec {hist['cvae_rec'][-1]:.5f} bdy {hist['cvae_boundary'][-1]:.5f} "
            f"kl {hist['cvae_kl'][-1]:.4f}]"
        )


# -----------------------------------------------------------------------------
# Post-training diagnostics
# -----------------------------------------------------------------------------
encoder1.eval(); encoder2.eval()
decoderA.eval(); decoderB.eval()
cvae_encoder.eval(); cvae_decoder.eval()

with torch.no_grad():
    z1_A_all = encoder1(A_cond)
    z2_A_all = encoder2(A_cond)
    c_hat_A_all  = decoderA(torch.cat([z1_A_all, z2_A_all], dim=1))
    mi_err_A     = F.mse_loss(c_hat_A_all, A_cond).item()

    z1_B_all = encoder1(B_cond)
    z2_B_all = encoder2(B_cond)
    c_hat_B_all  = decoderB(torch.cat([z1_B_all, z2_B_all], dim=1))
    mi_err_B     = F.mse_loss(c_hat_B_all, B_cond).item()

    sub = torch.randperm(N_B, device=device)[:N_A]
    z2_B_sub = encoder2(B_cond[sub])
    c_hat_A_cross_all = decoderA(torch.cat([z1_A_all, z2_B_sub], dim=1))
    mi_err_cross      = F.mse_loss(c_hat_A_cross_all, A_cond).item()

    # Distance-based separation check (on deterministic latents)
    diff_mean_z1 = (z1_A_all.mean(dim=0) - z1_B_all.mean(dim=0))
    sq_dist_z1   = (diff_mean_z1 ** 2).sum().item()
    diff_mean_zS = (z2_A_all.mean(dim=0) - z2_B_all.mean(dim=0))
    sq_dist_zS   = (diff_mean_zS ** 2).sum().item()

    print(f"\n[Distance-based disentanglement check]")
    print(f"  || mean(z1_A) - mean(z1_B) ||^2 = {sq_dist_z1:.4f}   "
          f"(target >= margin = {sep_margin})")
    print(f"  || mean(zS_A) - mean(zS_B) ||^2 = {sq_dist_zS:.4f}   "
          f"(target ~ 0)")
    print(f"  mean cosine(z1_A, zS_A): "
          f"{F.cosine_similarity(z1_A_all, z2_A_all, dim=1).mean().item():+.3f}")
    print(f"  mean cosine(z1_B, zS_B): "
          f"{F.cosine_similarity(z1_B_all, z2_B_all, dim=1).mean().item():+.3f}")

    cond_z_all = torch.cat([z1_A_all, z2_A_all], dim=1)
    mu_x_all, _ = cvae_encoder(A_deltas_n, A_cond)
    deltas_post = cvae_decoder(mu_x_all, cond_z_all)
    cvae_err_post = F.mse_loss(deltas_post, A_deltas_n).item()

    z_prior = torch.randn(N_A, zx_dim, device=device)
    deltas_prior = cvae_decoder(z_prior, cond_z_all)
    cvae_err_prior = F.mse_loss(deltas_prior, A_deltas_n).item()

print("\n========== Diagnostics ==========")
print(f"[MI-AE]  decoderA recon of A_cond:                 {mi_err_A:.4f}")
print(f"[MI-AE]  decoderB recon of B_cond:                 {mi_err_B:.4f}")
print(f"[MI-AE]  cross (z_shared from B) recon of A_cond:  {mi_err_cross:.4f}")
print(f"[MI-AE]  cross/correct ratio (lower => z_shared aligned across domains): "
      f"{mi_err_cross / max(mi_err_A, 1e-8):.2f}x")
print(f"[CVAE]   posterior recon MSE on A DELTAS (norm):   {cvae_err_post:.5f}")
print(f"[CVAE]   prior-sample recon MSE on A DELTAS (norm):{cvae_err_prior:.5f}")

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes[0, 0].plot(hist["total"]);          axes[0, 0].set_title("Total loss");           axes[0, 0].grid(True)
axes[0, 1].plot(hist["mi_recA"]);        axes[0, 1].set_title("MI-AE recA");           axes[0, 1].grid(True)
axes[0, 2].plot(hist["mi_recB"]);        axes[0, 2].set_title("MI-AE recB");           axes[0, 2].grid(True)
axes[0, 3].plot(hist["cvae_boundary"]);  axes[0, 3].set_title("CVAE boundary");        axes[0, 3].grid(True)
axes[1, 0].plot(hist["share"]);          axes[1, 0].set_title("L_share (pull zS)");    axes[1, 0].grid(True)
axes[1, 1].plot(hist["sep"]);            axes[1, 1].set_title("L_sep (push z1)");      axes[1, 1].grid(True)
axes[1, 2].plot(hist["orth"]);           axes[1, 2].set_title("L_orth (z1 vs zS)");    axes[1, 2].grid(True)
axes[1, 3].plot(hist["cvae_rec"]);       axes[1, 3].set_title("CVAE recon (delta)");   axes[1, 3].grid(True)
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# Save checkpoint
# -----------------------------------------------------------------------------
out_folder = Path(
    "C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Documents/Desktop/Journal_2025_work/Mars_lander_MIVAE_CVAE_joint"
)
out_folder.mkdir(parents=True, exist_ok=True)

checkpoint = {
    # MI (deterministic) encoders + decoders
    "encoder1_state_dict": encoder1.state_dict(),
    "encoder2_state_dict": encoder2.state_dict(),
    "decoderA_state_dict": decoderA.state_dict(),
    "decoderB_state_dict": decoderB.state_dict(),
    "z1_dim": z1_dim, "zS_dim": zS_dim, "hidden_mi": hidden_mi,
    "cond_dim_raw": cond_dim_raw,
    "mi_encoder_mode": "deterministic",

    # Distance-based disentanglement hyperparams
    "disentangle_mode": "distance_based",
    "lambda_share": lambda_share,
    "lambda_sep":   lambda_sep,
    "sep_margin":   sep_margin,
    "lambda_orth":  lambda_orth,
    "lambda_zS_norm": lambda_zS_norm,

    # CVAE
    "cvae_encoder_state_dict": cvae_encoder.state_dict(),
    "cvae_decoder_state_dict": cvae_decoder.state_dict(),
    "zx_dim": zx_dim, "hidden_cv": hidden_cv,
    "state_rest_dim": state_rest_dim,
    "delta_flat_dim": delta_flat_dim,
    "prediction_mode": "delta",

    # Layout
    "n_states": n_states, "n_controls": n_controls, "T": T,
    "state_flat_dim": state_flat_dim,
    "control_flat_dim": control_flat_dim,

    # CVAE (real-A) state stats
    "A_s0_mean": A_s0_mean.cpu(), "A_s0_std": A_s0_std.cpu(),
    "A_srest_mean": A_srest_mean.cpu(), "A_srest_std": A_srest_std.cpu(),
    "A_delta_mean": A_delta_mean.cpu(), "A_delta_std": A_delta_std.cpu(),

    # MI conditioning stats (per-domain)
    "A_u_mean":  A_u_mean.cpu(),  "A_u_std":  A_u_std.cpu(),
    "A_ic_mean": A_ic_mean.cpu(), "A_ic_std": A_ic_std.cpu(),
    "B_u_mean":  B_u_mean.cpu(),  "B_u_std":  B_u_std.cpu(),
    "B_ic_mean": B_ic_mean.cpu(), "B_ic_std": B_ic_std.cpu(),

    "hist": hist,
}
ckpt_path = out_folder / "mivae_cvae_joint_distance_1000.pt"
torch.save(checkpoint, ckpt_path)
print(f"\nSaved checkpoint to {ckpt_path}")