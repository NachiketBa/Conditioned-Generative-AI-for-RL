# This script implements the CVAE with DELTA prediction.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import glob
import os
import pandas as pd

# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
print(f"Using device: {device}")


# -------------------------------
# Load data
# -------------------------------
data_path = "C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Documents/Desktop/Journal_2025_work/cvae_process_real_noise"
file_pattern = "*.csv"

T = 100
state_dim = 9
control_dim = 3
state_flat_dim = state_dim * T                   # 900
state_rest_dim = state_dim * (T - 1)             # 891
delta_flat_dim = state_dim * (T - 1)             # 891 — same shape as state_rest
control_flat_dim = control_dim * T               # 300
total_flat_dim = state_flat_dim + control_flat_dim  # 1200


def load_flat_csvs(folder_path, pattern, expected_dim):
    csv_files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No files found matching '{pattern}' in {folder_path}")
    rows = []
    for file in csv_files:
        df = pd.read_csv(file, header=None).astype(float)
        arr = df.values.reshape(-1)
        if arr.shape[0] != expected_dim:
            raise ValueError(
                f"File {file} has {arr.shape[0]} entries, expected {expected_dim}"
            )
        rows.append(torch.tensor(arr, dtype=torch.float32))
    return torch.stack(rows)


flat_data = load_flat_csvs(data_path, file_pattern, total_flat_dim)
print("Loaded flattened data shape [N, 12*T]:", flat_data.shape)

N = flat_data.shape[0]
states = flat_data[:, :state_flat_dim]
controls = flat_data[:, state_flat_dim:]

states_3d = states.reshape(N, state_dim, T)              # [N, 9, T]
s0 = states_3d[:, :, 0]                                  # [N, 9]
deltas_3d = states_3d[:, :, 1:] - states_3d[:, :, :-1]   # [N, 9, T-1]
deltas = deltas_3d.reshape(N, delta_flat_dim)            # [N, 9*(T-1)]

# -------------------------------
# Train split
# -------------------------------
n_train = 40
s0_train       = s0[:n_train]
deltas_train   = deltas[:n_train]
controls_train = controls[:n_train]

# -------------------------------
# Normalization
# -------------------------------
STD_FLOOR = 1e-3

s0_mean = s0_train.mean(dim=0, keepdim=True)
s0_std  = s0_train.std(dim=0, keepdim=True)
s0_std  = torch.where(s0_std < STD_FLOOR, torch.ones_like(s0_std), s0_std)

# Per-timestep stats for DELTAS
deltas_3d_train = deltas_train.reshape(n_train, state_dim, T - 1)
delta_mean_3d = deltas_3d_train.mean(dim=0, keepdim=True)
delta_std_3d  = deltas_3d_train.std(dim=0, keepdim=True)
delta_std_3d  = torch.where(delta_std_3d < STD_FLOOR,
                            torch.ones_like(delta_std_3d), delta_std_3d)
delta_mean = delta_mean_3d.reshape(1, delta_flat_dim)
delta_std  = delta_std_3d.reshape(1, delta_flat_dim)

ctrl_mean = controls_train.mean(dim=0, keepdim=True)
ctrl_std  = controls_train.std(dim=0, keepdim=True)
ctrl_std  = torch.where(ctrl_std < STD_FLOOR, torch.ones_like(ctrl_std), ctrl_std)

# Normalize
s0_train_n       = (s0_train       - s0_mean)    / s0_std
deltas_train_n   = (deltas_train   - delta_mean) / delta_std
controls_train_n = (controls_train - ctrl_mean)  / ctrl_std

dataset = TensorDataset(s0_train_n, deltas_train_n, controls_train_n)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# -------------------------------
# Model
# -------------------------------
latent_dim = 32
hidden_dim = 324


class Encoder(nn.Module):
    def __init__(self, state_dim, delta_flat_dim, control_flat_dim, latent_dim, hidden_dim):
        super().__init__()
        in_dim = state_dim + delta_flat_dim + control_flat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, s0, deltas, u):
        x = torch.cat([s0, deltas, u], dim=1)
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, state_dim, delta_flat_dim, control_flat_dim, hidden_dim):
        super().__init__()
        in_dim = latent_dim + state_dim + control_flat_dim
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
            nn.Linear(hidden_dim, delta_flat_dim),
        )

    def forward(self, z, s0, u):
        return self.net(torch.cat([z, s0, u], dim=1))


def reparameterize(mu, logvar):
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


encoder = Encoder(state_dim, delta_flat_dim, control_flat_dim, latent_dim, hidden_dim).to(device)
decoder = Decoder(latent_dim, state_dim, delta_flat_dim, control_flat_dim, hidden_dim).to(device)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
)

epochs = 1000
anneal_till = 300
kl_max = 0.05
boundary_weight = 10.0


def kl_weight(epoch, total, k_max):
    return min(k_max, (epoch / total) * k_max)


for epoch in range(epochs):
    tot_recon = tot_kl = tot_boundary = 0.0
    w = kl_weight(epoch, anneal_till, kl_max)

    encoder.train()
    decoder.train()

    for s0_batch, deltas_batch, u_batch in loader:
        s0_batch     = s0_batch.to(device)
        deltas_batch = deltas_batch.to(device)
        u_batch      = u_batch.to(device)
        optimizer.zero_grad()

        mu, logvar = encoder(s0_batch, deltas_batch, u_batch)
        z = reparameterize(mu, logvar)
        deltas_pred = decoder(z, s0_batch, u_batch)  # [B, 9*(T-1)]
        recon_loss = F.mse_loss(deltas_pred, deltas_batch, reduction='sum')
        B = deltas_pred.shape[0]
        deltas_pred_3d = deltas_pred.reshape(B, state_dim, T - 1)
        deltas_true_3d = deltas_batch.reshape(B, state_dim, T - 1)
        boundary_loss = F.mse_loss(deltas_pred_3d[:, :, 0],
                                   deltas_true_3d[:, :, 0],
                                   reduction='sum')

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss + w * kl_loss + boundary_weight * boundary_loss

        loss.backward()
        optimizer.step()

        tot_recon    += recon_loss.item()
        tot_kl       += kl_loss.item()
        tot_boundary += boundary_loss.item()

    if (epoch + 1) % 20 == 0:
        n = len(loader)
        print(f"Epoch {epoch + 1:4d} | KL w: {w:.4f} | "
              f"Recon: {tot_recon / n:.5f} | "
              f"KL: {tot_kl / n:.5f} | "
              f"Boundary: {tot_boundary / n:.5f}")

# -------------------------------
# Save
# -------------------------------
torch.save({
    "encoder_state_dict": encoder.state_dict(),
    "decoder_state_dict": decoder.state_dict(),
    "s0_mean": s0_mean, "s0_std": s0_std,
    "delta_mean": delta_mean, "delta_std": delta_std,  # deltas, not srest
    "ctrl_mean": ctrl_mean, "ctrl_std": ctrl_std,
    "state_dim": state_dim, "control_dim": control_dim,
    "state_flat_dim": state_flat_dim,
    "state_rest_dim": state_rest_dim,
    "delta_flat_dim": delta_flat_dim,
    "control_flat_dim": control_flat_dim,
    "latent_dim": latent_dim, "hidden_dim": hidden_dim, "T": T,
    "prediction_mode": "delta",
}, "cvae.pth")