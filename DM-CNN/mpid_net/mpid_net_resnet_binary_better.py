###
### RESNET NET VERSION FOR BINARY CLASSIFICATION
###
### - Default normalization: BatchNorm (standard torchvision ResNet)
### - Optional: GroupNorm via norm="gn"
### - Input: supports single-channel detector images (in_channels=1)
### - Output: num_classes logits (default 2) for BCEWithLogitsLoss + one-hot labels
###

import torch
import torch.nn as nn

try:
    from torchvision import models
except ImportError as e:
    raise ImportError(
        "mpid_net_resnet_binary.py requires torchvision. "
        "Install torchvision or implement a custom ResNet."
    ) from e


class MPID(nn.Module):
    """
    MPID-style wrapper around torchvision ResNet for your binary setup.
    Keeps constructor pattern similar to mpid_net_binary.py.
    """
    def __init__(
        self,
        dropout=0.5,
        num_classes=2,
        eps=1e-05,            # kept for interface compatibility
        running_stats=False,  # kept for interface compatibility
        arch="resnet18",
        in_channels=1,
        pretrained=False,
        norm="bn",            # "bn" (default) or "gn"
        gn_groups=32,
    ):
        super(MPID, self).__init__()

        # Choose normalization layer for ResNet blocks
        norm = str(norm).lower()
        if norm not in ("bn", "gn"):
            raise ValueError("norm must be 'bn' or 'gn'")

        if norm == "bn":
            norm_layer = nn.BatchNorm2d
        else:
            # GroupNorm factory (ResNet expects norm_layer(num_channels) -> module)
            def norm_layer(num_channels):
                # You can lower gn_groups if you ever use a backbone with channels not divisible by 32
                return nn.GroupNorm(gn_groups, num_channels)

        # Build backbone with the chosen norm layer
        if arch == "resnet18":
            backbone = self._build_resnet18(norm_layer=norm_layer, pretrained=pretrained)
        elif arch == "resnet34":
            backbone = self._build_resnet34(norm_layer=norm_layer, pretrained=pretrained)
        else:
            raise ValueError("arch must be 'resnet18' or 'resnet34'")

        # --- Adapt first conv to accept in_channels (your case: 1) ---
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # If pretrained weights exist and in_channels != 3, initialize conv1 sensibly.
        # Note: torchvision pretrained weights are typically for RGB ImageNet.
        if pretrained:
            with torch.no_grad():
                if in_channels == 1:
                    # average RGB weights -> 1 channel
                    backbone.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
                elif in_channels == 2:
                    # take first 2 channels from RGB
                    backbone.conv1.weight[:, :2, :, :] = old_conv.weight[:, :2, :, :]
                elif in_channels == 3:
                    backbone.conv1.weight[:] = old_conv.weight
                elif in_channels > 3:
                    # copy RGB then fill remaining channels with mean RGB
                    backbone.conv1.weight[:, :3, :, :] = old_conv.weight
                    mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)
                    for c in range(3, in_channels):
                        backbone.conv1.weight[:, c:c+1, :, :] = mean_rgb

        # --- Replace classifier head: output num_classes logits ---
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, num_classes),
        )

        self.net = backbone

        # eps/running_stats are intentionally unused here (BN/GN handle their own internals).
        # They exist only to keep the constructor signature close to mpid_net_binary.py.

    def forward(self, x):
        return self.net(x)

    def _build_resnet18(self, norm_layer, pretrained=False):
        # Handle both new and older torchvision APIs
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            return models.resnet18(weights=weights, norm_layer=norm_layer)
        except Exception:
            return models.resnet18(pretrained=pretrained, norm_layer=norm_layer)

    def _build_resnet34(self, norm_layer, pretrained=False):
        try:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            return models.resnet34(weights=weights, norm_layer=norm_layer)
        except Exception:
            return models.resnet34(pretrained=pretrained, norm_layer=norm_layer)



if __name__ == '__main__':
    # Minimal sanity check: forward pass shape
    mpid = MPID(
        arch="resnet18",
        in_channels=1,
        num_classes=2,
        dropout=0.5,
        pretrained=False,
        norm="bn",   # change to "gn" to test GroupNorm
        gn_groups=32,
    )
    print(mpid)
    print("mpid.training, ", mpid.training)

    x = torch.ones((2, 1, 512, 512))
    y = mpid(x)
    print("output shape:", y.shape)  # should be [2, 2]
