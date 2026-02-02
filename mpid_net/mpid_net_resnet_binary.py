###
### Note 2019-style: using GroupNorm to avoid BatchNorm inference issues.
### (Keeping similar style to mpid_net_binary.py)
###

# RESNET NET VERSION FOR BINARY CLASSIFICATION

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
    # eps/running_stats kept for interface compatibility with mpid_net_binary.py
    def __init__(
        self,
        dropout=0.5,
        num_classes=2,
        eps=1e-05,
        running_stats=False,
        arch="resnet18",
        in_channels=1,
        pretrained=False,
    ):
        super(MPID, self).__init__()

        # GroupNorm factory (ResNet expects norm_layer(num_channels) -> module)
        # Use 32 groups (standard for GN ResNets); channels are divisible for resnet18/34 (64/128/256/512).
        def gn(num_channels):
            return nn.GroupNorm(32, num_channels)

        # --- Build backbone ---
        # Handle different torchvision versions (weights API vs pretrained=)
        backbone = None
        if arch == "resnet18":
            backbone = self._build_resnet18(norm_layer=gn, pretrained=pretrained)
        elif arch == "resnet34":
            backbone = self._build_resnet34(norm_layer=gn, pretrained=pretrained)
        else:
            raise ValueError("arch must be 'resnet18' or 'resnet34'")

        # --- Adapt first conv to single-channel (or arbitrary in_channels) ---
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # If pretrained weights exist and in_channels != 3, initialize conv1 sensibly
        if pretrained:
            with torch.no_grad():
                if in_channels == 1:
                    # average RGB weights -> 1 channel
                    backbone.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
                elif in_channels > 3:
                    # copy RGB then repeat remaining channels from mean
                    backbone.conv1.weight[:, :3, :, :] = old_conv.weight
                    mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)
                    for c in range(3, in_channels):
                        backbone.conv1.weight[:, c:c+1, :, :] = mean_rgb
                else:
                    # in_channels == 2 or 3
                    backbone.conv1.weight[:, :in_channels, :, :] = old_conv.weight[:, :in_channels, :, :]

        # --- Replace classifier head to output 2 logits ---
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, num_classes),
        )

        self.net = backbone

    def forward(self, x):
        return self.net(x)

    def _build_resnet18(self, norm_layer, pretrained=False):
        # torchvision new API uses weights=..., older API uses pretrained=...
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
    x = torch.ones([512, 512])
    mpid = MPID(arch="resnet18", in_channels=1, pretrained=False)
    print(mpid)
    print("mpid.training, ", mpid.training)
    print(mpid(x.view((-1, 1, 512, 512))))
