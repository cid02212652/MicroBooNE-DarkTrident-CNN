
"""
sparse_net_binary.py

Event-level classifier using SparseSSNet-style sparse U-ResNet (SparseConvNet)
adapted for *event-level* labels.

- Input: dense images (B,1,H,W) from your existing MPID_Dataset
  (see mpid_data_binary.MPID_Dataset) which returns (H,W) and is reshaped in the trainer. 
- Output: logits (B, num_classes) compatible with BCEWithLogitsLoss used in your trainers. 

Dependencies:
- sparseconvnet (FacebookResearch SparseConvNet). If not installed, this file raises a
  helpful ImportError when you instantiate the model.

Why pooling is "different" from ResNet's global average pooling:
- ResNet global avg pool averages over *all* HxW locations.
- SparseConvNet operates on active sites only (nonzero pixels).
  Our "mean" pooling averages over active sites (a sparse analogue of GAP).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


def dense_to_point_cloud(x: torch.Tensor, threshold: float = 0.0):
    """
    Convert dense (B,H,W) or (B,1,H,W) to SparseConvNet (coords, feats).

    Returns:
      coords: (N,3) int64 on CPU, ordered [y, x, batch]
      feats : (N,1) float32 on same device as x
    """
    if x.ndim == 3:
        x = x.unsqueeze(1)  # (B,1,H,W)
    if x.ndim != 4 or x.size(1) != 1:
        raise ValueError(f"Expected (B,1,H,W) or (B,H,W), got {tuple(x.shape)}")

    # Find active pixels. idx columns: [b, c(=0), y, x]
    idx = (x > threshold).nonzero(as_tuple=False)

    if idx.numel() == 0:
        coords = torch.zeros((0, 3), device="cpu", dtype=torch.long)
        feats  = torch.zeros((0, 1), device=x.device, dtype=torch.float32)
        return coords, feats

    b  = idx[:, 0]
    y  = idx[:, 2]
    xx = idx[:, 3]

    # Features stay on the same device as x (GPU if x is GPU)
    feats = x[b, 0, y, xx].reshape(-1, 1).to(dtype=torch.float32)

    # Coords MUST be Long on CPU for SparseConvNet stability (esp. with GPU features)
    coords = torch.stack([y, xx, b], dim=1).to(device="cpu", dtype=torch.long)

    # Optional safety during debugging:
    H = x.size(2)
    W = x.size(3)
    assert coords[:,0].min() >= 0 and coords[:,0].max() < H
    assert coords[:,1].min() >= 0 and coords[:,1].max() < W

    return coords, feats


class MPID(nn.Module):
    """
    Drop-in model class name "MPID" to match the rest of the codebase,
    but implemented as SparseSSNet-style sparse UResNet + event pooling.

    Parameters map roughly to SparseSSNet flags: 
      - spatial_size: -ss
      - data_dim: -dd (2 for your 512x512 images)
      - uresnet_num_strides: -uns
      - uresnet_filters: -uf
    """

    def __init__(
        self,
        dropout: float = 0.0,
        num_classes: int = 2,
        spatial_size: int = 512,
        data_dim: int = 2,
        uresnet_num_strides: int = 5,
        uresnet_filters: int = 16,
        reps: int = 2,
        pool: Literal["mean", "max", "meanmax"] = "meanmax",
        dense_threshold: float = 0.0,
    ):
        super().__init__()

        try:
            import sparseconvnet as scn
        except Exception as e:
            raise ImportError(
                "This model requires 'sparseconvnet' (FacebookResearch SparseConvNet).\n"
                "SparseSSNet is built on SparseConvNet; install/build it first.\n"
                "Repo: https://github.com/facebookresearch/SparseConvNet\n"
            ) from e

        self.num_classes = int(num_classes)
        self.pool = pool
        self.dense_threshold = float(dense_threshold)
        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()

        # Following the structure used in ranitay/SparseSSNet uresnet_sparse.py 
        dimension = int(data_dim)
        m = int(uresnet_filters)
        nPlanes = [i * m for i in range(1, int(uresnet_num_strides) + 1)]
        kernel_size = 2  # matches their code 

        self.sparse_model = (
            scn.Sequential()
            .add(scn.InputLayer(dimension, int(spatial_size), mode=3))
            .add(scn.SubmanifoldConvolution(dimension, 1, m, 3, False))
            .add(scn.UNet(dimension, int(reps), nPlanes, residual_blocks=True, downsample=[kernel_size, 2]))
            .add(scn.BatchNormReLU(m))
            .add(scn.OutputLayer(dimension))
        )

        head_in = m * (2 if pool == "meanmax" else 1)
        self.head = nn.Linear(head_in, self.num_classes)

    @staticmethod
    def _mean_pool(point_feat: torch.Tensor, batch_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
        # point_feat: (N, C), batch_ids: (N,)
        out = torch.zeros((batch_size, point_feat.size(1)), device=point_feat.device, dtype=point_feat.dtype)
        counts = torch.zeros((batch_size, 1), device=point_feat.device, dtype=point_feat.dtype)
        out.index_add_(0, batch_ids, point_feat)
        ones = torch.ones((point_feat.size(0), 1), device=point_feat.device, dtype=point_feat.dtype)
        counts.index_add_(0, batch_ids, ones)
        return out / torch.clamp(counts, min=1.0)

    @staticmethod
    def _max_pool(point_feat: torch.Tensor, batch_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
        C = point_feat.size(1)
        out = torch.full((batch_size, C), -1e9, device=point_feat.device, dtype=point_feat.dtype)
        for b in batch_ids.unique():
            bi = int(b.item())
            sel = (batch_ids == b)
            out[bi] = point_feat[sel].max(dim=0).values
        # any completely empty events (shouldn't happen if B inferred from input) -> zeros
        out = torch.where(out < -1e8, torch.zeros_like(out), out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: dense image batch (B,1,H,W) or (B,H,W) (we add channel).
        returns: logits (B, num_classes)
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Expected (B,1,H,W) or (B,H,W), got {tuple(x.shape)}")

        B = int(x.size(0))
        coords, feats = dense_to_point_cloud(x, threshold=self.dense_threshold)

        if coords.numel() == 0:
            return torch.zeros((B, self.num_classes), device=x.device, dtype=torch.float32)
        
        point_feat = self.sparse_model((coords.int(), feats))  # coords int32 is fine
        batch_ids = coords[:, -1]

        if self.pool == "mean":
            pooled = self._mean_pool(point_feat, batch_ids, B)
        elif self.pool == "max":
            pooled = self._max_pool(point_feat, batch_ids, B)
        elif self.pool == "meanmax":
            pooled = torch.cat(
                [self._mean_pool(point_feat, batch_ids, B), self._max_pool(point_feat, batch_ids, B)],
                dim=1,
            )
        else:
            raise ValueError(f"Unknown pool={self.pool}")

        pooled = self.dropout(pooled)
        logits = self.head(pooled)
        return logits
