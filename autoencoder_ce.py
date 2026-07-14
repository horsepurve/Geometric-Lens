"""
MLP Autoencoder for 2D visualization of mesh-structured high-dimensional data.

Key features:
  - Encoder: 2000 -> ... -> 2  |  Decoder: symmetric 2 -> ... -> 2000
  - Grid-based interpolation sampling (fresh synthetic points each epoch)
  - Three losses:
      L_recon    : MSE(x, Decoder(Encoder(x)))
      L_neighbor : preserve pairwise distances from original space in latent space
      L_ce       : CrossEntropy( classifier(Decoder(z)), argmax(classifier(x)) )
                   ground-truth labels = argmax of frozen classifier on original x
                   predicted logits    = frozen classifier applied to x_hat = Decoder(z)
  - Global kNN graph rebuilt each epoch (for L_neighbor only)
  - Frozen linear classifier (10 000 classes)

smoke test:
python /mnt/user-data/outputs/mesh_autoencoder.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AEConfig:
    # ── Data layout ───────────────────────────────────────────────────────────
    input_dim:        int   = 2000   # dimensionality of each point
    num_rows:         int   = 5      # mesh rows
    num_cols:         int   = 32     # mesh columns  (num_rows x num_cols = N)

    # ── Grid sampling ─────────────────────────────────────────────────────────
    samples_per_cell: int   = 4      # synthetic points per 2x2 cell per epoch
    dirichlet_alpha:  float = 2.0    # >1 -> weights near centroid (mostly inside simplex)
    extrap_std:       float = 0.05   # Gaussian noise on weights -> mild extrapolation

    # ── Architecture ──────────────────────────────────────────────────────────
    hidden_dims:  list = field(default_factory=lambda: [512, 128, 32])
    norm:         str  = "layernorm"   # "layernorm" | "batchnorm" | "none"
    activation:   str  = "gelu"        # "gelu" | "relu"

    # ── kNN (for L_neighbor only) ─────────────────────────────────────────────
    k_neighbors:  int  = 10

    # ── Loss weights ──────────────────────────────────────────────────────────
    lambda_recon:    float = 1.0
    lambda_neighbor: float = 1.0
    lambda_ce:       float = 1.0

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int   = 32
    lr:         float = 1e-3
    epochs:     int   = 200
    device:     str   = "cuda" if torch.cuda.is_available() else "cpu"
    seed:       int   = 42

    # ── Visualisation ─────────────────────────────────────────────────────────
    plot_every: int   = 50     # scatter plot every N epochs (0 = only at end)
    figsize:    tuple = (6, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def _make_block(in_dim: int, out_dim: int, norm: str, act: str) -> nn.Sequential:
    layers = [nn.Linear(in_dim, out_dim)]
    if norm == "layernorm":
        layers.append(nn.LayerNorm(out_dim))
    elif norm == "batchnorm":
        layers.append(nn.BatchNorm1d(out_dim))
    layers.append(nn.GELU() if act == "gelu" else nn.ReLU())
    return nn.Sequential(*layers)


class MeshAutoencoder(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder: input_dim -> hidden_dims -> 2
        enc_dims = [cfg.input_dim] + cfg.hidden_dims + [2]
        enc_blocks = []
        for i in range(len(enc_dims) - 2):
            enc_blocks.append(_make_block(enc_dims[i], enc_dims[i+1], cfg.norm, cfg.activation))
        enc_blocks.append(nn.Linear(enc_dims[-2], enc_dims[-1]))   # final: no norm/act
        self.encoder = nn.Sequential(*enc_blocks)

        # Decoder: symmetric  2 -> hidden_dims reversed -> input_dim
        dec_dims = list(reversed(enc_dims))
        dec_blocks = []
        for i in range(len(dec_dims) - 2):
            dec_blocks.append(_make_block(dec_dims[i], dec_dims[i+1], cfg.norm, cfg.activation))
        dec_blocks.append(nn.Linear(dec_dims[-2], dec_dims[-1]))   # final: no norm/act
        self.decoder = nn.Sequential(*dec_blocks)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z     = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


# ─────────────────────────────────────────────────────────────────────────────
# Frozen classifier
# ─────────────────────────────────────────────────────────────────────────────

class FrozenClassifier(nn.Module):
    """
    Linear classifier (weight, bias) that is always frozen.

    forward(x)  -> raw logits  (N, num_classes)   <- differentiable w.r.t. x
    labels(x)   -> argmax      (N,)  int64         <- hard labels, no gradient
    """
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight.float())   # (C, D)
        self.register_buffer("bias",   bias.float())     # (C,)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits (N, C). Gradient flows through x, not through weight/bias."""
        return x @ self.weight.T + self.bias

    @torch.no_grad()
    def labels(self, x: torch.Tensor) -> torch.Tensor:
        """Hard class labels (N,) int64. No gradient."""
        return self.forward(x).argmax(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Grid sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_grid_points(
    data: dict,
    num_rows: int,
    num_cols: int,
    samples_per_cell: int,
    dirichlet_alpha: float = 2.0,
    extrap_std: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    For every 2x2 cell in the mesh, generate `samples_per_cell` synthetic points
    via noisy Dirichlet interpolation over the 4 corners:

        weights ~ Dirichlet(alpha) + N(0, extrap_std^2),  then renormalise
        alpha > 1  -> cluster near centroid (mostly inside simplex)
        extrap_std -> mild extrapolation outside simplex

    Returns (N_synthetic, input_dim) float32 array.
    """
    if rng is None:
        rng = np.random.default_rng()

    synthetic = []
    for i in range(num_rows - 1):
        for j in range(num_cols - 1):
            corners = [
                data.get((i,   j)),
                data.get((i+1, j)),
                data.get((i,   j+1)),
                data.get((i+1, j+1)),
            ]
            if any(c is None for c in corners):
                continue
            C = np.stack([np.asarray(c, dtype=np.float32) for c in corners])  # (4, D)

            w = rng.dirichlet(np.full(4, dirichlet_alpha), size=samples_per_cell)  # (k, 4)
            w = w + rng.normal(0, extrap_std, size=w.shape)
            w = w / w.sum(axis=1, keepdims=True)   # renormalise; negatives -> extrapolation

            synthetic.append(w @ C)   # (k, D)

    if not synthetic:
        D = np.asarray(next(iter(data.values()))).shape[-1]
        return np.empty((0, D), dtype=np.float32)
    return np.concatenate(synthetic, axis=0)


def build_epoch_dataset(
    data: dict,
    num_rows: int,
    num_cols: int,
    samples_per_cell: int,
    dirichlet_alpha: float,
    extrap_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Original points (row-major) ++ fresh synthetic interpolations.
    Returns float32 array of shape (N_orig + N_synthetic, input_dim).
    """
    orig = np.stack(
        [np.asarray(data[(i, j)], dtype=np.float32)
         for i in range(num_rows) for j in range(num_cols)
         if (i, j) in data],
        axis=0,
    )
    if samples_per_cell == 0:
        return orig 
    synth = sample_grid_points(
        data, num_rows, num_cols,
        samples_per_cell, dirichlet_alpha, extrap_std, rng,
    )
    if synth.shape[0] == 0:
        return orig
    return np.concatenate([orig, synth], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# kNN graph
# ─────────────────────────────────────────────────────────────────────────────

def build_knn_graph(X: np.ndarray, k: int):
    """
    Returns:
        indices   (N, k) -- kNN indices (self excluded)
        distances (N, k) -- Euclidean distances in original space
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nbrs.fit(X)
    dists, idxs = nbrs.kneighbors(X)
    return idxs[:, 1:], dists[:, 1:]


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────

def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)


def neighborhood_loss(
    z_batch:       torch.Tensor,   # (B, 2)  latent for batch points
    batch_indices: torch.Tensor,   # (B,)    global indices (CPU)
    knn_indices:   np.ndarray,     # (N, k)  global kNN index table
    knn_dists:     np.ndarray,     # (N, k)  original-space distances
    z_all_det:     torch.Tensor,   # (N, 2)  full latent, detached
    device:        str,
) -> torch.Tensor:
    """
    For each batch point i and each of its k neighbours j:
        L = mean_ij ( ||z_i - z_j||  -  d_ij )^2

    Encourages latent distances to mirror original-space distances.
    z_j is looked up from z_all_det (detached), so gradient only flows through z_i.
    """
    total = torch.tensor(0.0, device=device)
    count = 0
    for local_idx, global_idx in enumerate(batch_indices.tolist()):
        nbr_idx = knn_indices[global_idx]                                    # (k,)
        d_orig  = torch.tensor(knn_dists[global_idx], dtype=torch.float32,
                               device=device)                                # (k,)
        zi = z_batch[local_idx].unsqueeze(0)                                # (1, 2)
        zj = z_all_det[nbr_idx]                                             # (k, 2)

        # dist_latent = torch.norm(zi - zj, dim=-1)                           # (k,)
        # eps inside sqrt avoids zero-division nan gradients when zi == zj
        diff = zi - zj                                                       # (k, 2)
        dist_latent = (diff * diff).sum(-1).clamp(min=1e-8).sqrt()          # (k,)
        
        total = total + ((dist_latent - d_orig) ** 2).mean()
        count += 1

    return total / max(count, 1)


def classification_loss(
    x_hat:      torch.Tensor,        # (B, D)  reconstructed -- differentiable
    labels:     torch.Tensor,        # (B,)    hard labels from classifier(x), int64
    classifier: FrozenClassifier,
) -> torch.Tensor:
    """
    CrossEntropy( classifier(x_hat), labels )

    Full pipeline for each batch point:
        x -> z = Encoder(x) -> x_hat = Decoder(z) -> logits = W @ x_hat + b
    Labels = argmax( classifier(x) ) computed once per epoch with no_grad.
    Gradient flows x_hat -> logits -> CE; classifier weights are frozen buffers.
    """
    logits = classifier(x_hat)          # (B, C) -- differentiable w.r.t. x_hat
    return F.cross_entropy(logits, labels)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data:               dict,
    classifier_weight:  torch.Tensor,   # (num_classes, input_dim)
    classifier_bias:    torch.Tensor,   # (num_classes,)
    cfg:                AEConfig,
) -> tuple:
    """
    Train the autoencoder. Returns (model, history).
    history: list of dicts with keys epoch, loss_recon, loss_neighbor, loss_ce, loss_total.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng    = np.random.default_rng(cfg.seed)
    device = cfg.device

    model      = MeshAutoencoder(cfg).to(device)
    classifier = FrozenClassifier(classifier_weight, classifier_bias).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = []

    for epoch in range(1, cfg.epochs + 1):

        # ── 1. Build augmented dataset for this epoch ─────────────────────────
        X_np  = build_epoch_dataset(
            data, cfg.num_rows, cfg.num_cols,
            cfg.samples_per_cell, cfg.dirichlet_alpha, cfg.extrap_std, rng,
        )                                                        # (N_aug, D)
        N_aug = X_np.shape[0]
        X_all = torch.tensor(X_np, dtype=torch.float32, device=device)

        # ── 2. Global kNN graph in original space (for L_neighbor) ────────────
        k_eff = min(cfg.k_neighbors, N_aug - 1)
        # knn_idx, knn_dist = build_knn_graph(X_np, k_eff)        # (N_aug, k_eff)

        # ── 3. Hard labels: argmax( classifier(x) ) for all augmented points ──
        #    These are the ground-truth targets for the CE loss.
        #    Computed once per epoch, no gradient needed.
        with torch.no_grad():
            labels_all = classifier.labels(X_all)                # (N_aug,) int64

        # ── 4. Mini-batch loop ────────────────────────────────────────────────
        perm    = torch.randperm(N_aug, device=device)
        batches = perm.split(cfg.batch_size)

        epoch_losses = {"recon": 0., "neighbor": 0., "ce": 0., "total": 0.}
        n_batches = 0

        model.train()
        for batch_idx_tensor in batches:
            x_batch   = X_all[batch_idx_tensor]            # (B, D)
            lbl_batch = labels_all[batch_idx_tensor]       # (B,) int64

            z_batch, x_hat = model(x_batch)                # (B, 2), (B, D)

            # Fresh full-dataset latents for neighbour look-ups (detached,
            # no gradient — neighbours are reference points, not optimised here)
            with torch.no_grad():
                z_all_det = model.encode(X_all).detach()   # (N_aug, 2)

            l_recon    = reconstruction_loss(x_batch, x_hat)
            # l_neighbor = neighborhood_loss(
            #     z_batch, batch_idx_tensor,
            #     knn_idx, knn_dist, z_all_det, device,
            # )
            l_ce = classification_loss(x_hat, lbl_batch, classifier)

            loss = (cfg.lambda_recon    * l_recon +
                    # cfg.lambda_neighbor * l_neighbor +
                    cfg.lambda_ce       * l_ce)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses["recon"]    += l_recon.item()
            # epoch_losses["neighbor"] += l_neighbor.item()
            epoch_losses["ce"]       += l_ce.item()
            epoch_losses["total"]    += loss.item()
            n_batches += 1

        for k_ in epoch_losses:
            epoch_losses[k_] /= max(n_batches, 1)

        history.append({
            "epoch":         epoch,
            "loss_recon":    epoch_losses["recon"],
            # "loss_neighbor": epoch_losses["neighbor"],
            "loss_ce":       epoch_losses["ce"],
            "loss_total":    epoch_losses["total"],
        })

        if epoch % max(cfg.epochs // 10, 1) == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{cfg.epochs}  |  "
                f"total={epoch_losses['total']:.4f}  "
                f"recon={epoch_losses['recon']:.4f}  "
                # f"nbr={epoch_losses['neighbor']:.4f}  "
                f"ce={epoch_losses['ce']:.4f}"
            )

        # Intermediate visualisation (on full augmented set)
        if cfg.plot_every > 0 and epoch % cfg.plot_every == 0 and epoch < cfg.epochs:
            _plot_embedding(
                model, X_all, labels_all, epoch, cfg, device,
                title=f"Epoch {epoch} (augmented, N={N_aug})",
            )

    # ── Final visualisation on ORIGINAL points only ───────────────────────────
    X_orig_np = np.stack(
        [np.asarray(data[(i, j)], dtype=np.float32)
         for i in range(cfg.num_rows) for j in range(cfg.num_cols)
         if (i, j) in data],
        axis=0,
    )
    X_orig = torch.tensor(X_orig_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        labels_orig = classifier.labels(X_orig)
    _plot_embedding(
        model, X_orig, labels_orig, cfg.epochs, cfg, device,
        title="Final embedding -- original points only",
    )

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _plot_embedding(
    model:   MeshAutoencoder,
    X:       torch.Tensor,    # (N, D) on device
    labels:  torch.Tensor,    # (N,)   int64
    epoch:   int,
    cfg:     AEConfig,
    device:  str,
    title:   str = "",
):
    model.eval()
    with torch.no_grad():
        z = model.encode(X).cpu().numpy()        # (N, 2)
    cls = labels.cpu().numpy()                   # (N,)
    model.train()

    _, ranks = np.unique(cls, return_inverse=True) # remap
    # n_cls = int(cls.max()) + 1
    n_cls = int(ranks.max()) + 1
    cmap  = (plt.colormaps.get_cmap("tab20").resampled(n_cls)
             if n_cls <= 20
             else plt.colormaps.get_cmap("nipy_spectral").resampled(n_cls))

    fig, ax = plt.subplots(figsize=cfg.figsize)
    sc = ax.scatter(z[:, 0], z[:, 1], c=ranks, cmap=cmap,
                    vmin=0, vmax=n_cls - 1, s=30, alpha=0.85, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Class (argmax of frozen classifier)")
    ax.set_title(title or f"2D embedding -- epoch {epoch}")
    ax.set_xlabel("z1");  ax.set_ylabel("z2")
    plt.tight_layout();   plt.show()


def plot_loss_curves(history: list):
    epochs    = [h["epoch"]         for h in history]
    totals    = [h["loss_total"]    for h in history]
    recons    = [h["loss_recon"]    for h in history]
    # neighbors = [h["loss_neighbor"] for h in history]
    ces       = [h["loss_ce"]       for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, totals, linewidth=2, label="total")
    axes[0].set_title("Total loss");  axes[0].set_xlabel("Epoch");  axes[0].legend()

    axes[1].plot(epochs, recons,    label="reconstruction")
    # axes[1].plot(epochs, neighbors, label="neighborhood")
    axes[1].plot(epochs, ces,       label="cross-entropy")
    axes[1].set_title("Individual losses");  axes[1].set_xlabel("Epoch");  axes[1].legend()

    plt.tight_layout();  plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Utility: encode arbitrary points after training
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_points(
    model:  MeshAutoencoder,
    points: np.ndarray,          # (N, input_dim)
    device: str = "cpu",
) -> np.ndarray:
    """Returns (N, 2) numpy array of latent coordinates."""
    model.eval()
    x = torch.tensor(points, dtype=torch.float32, device=device)
    return model.encode(x).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _make_dummy_data(num_rows=5, num_cols=8, input_dim=64, num_classes=20, seed=0):
    rng = np.random.default_rng(seed)
    data   = {(i, j): rng.standard_normal(input_dim).astype(np.float32)
              for i in range(num_rows) for j in range(num_cols)}
    weight = torch.randn(num_classes, input_dim) * 0.01
    bias   = torch.zeros(num_classes)
    return data, weight, bias


if __name__ == "__main__":
    print("Running smoke test ...")
    _data, _w, _b = _make_dummy_data()
    _cfg = AEConfig(
        input_dim=64, num_rows=5, num_cols=8,
        hidden_dims=[32, 8],
        epochs=10, batch_size=8,
        samples_per_cell=2, k_neighbors=5,
        plot_every=0, device="cpu",
    )
    _model, _hist = train(_data, _w, _b, _cfg)
    print("Smoke test passed.")
