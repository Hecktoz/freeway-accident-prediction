"""Shared PyTorch model architecture definitions.

Imported by inference.py (for loading checkpoints) and by the training scripts
(unet_matrix.py / unet_feature_fusion.py import the classes they need from here
rather than defining them inline).

All geometry constants that appear in both training and inference live here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

SPEED_MAX: float = 140.0
PAD_TARGET: int = 64
FREE_FLOW_KMH: float = 80.0

ALL_FF_METHODS = ("direct_concat", "mid_inject", "attention")
ALL_BACKBONES = ("simple", "residual_attention")


@dataclass
class WindowConfig:
    """Geometric parameters shared by training and inference."""

    upstream_km: int = 25
    downstream_km: int = 10
    time_before_min: int = 30
    time_after_min: int = 145
    spatial_resolution_km: int = 1
    resolution_min: int = 5

    @property
    def n_space(self) -> int:
        return (self.upstream_km + self.downstream_km) // self.spatial_resolution_km + 1

    @property
    def n_time(self) -> int:
        return (self.time_before_min + self.time_after_min) // self.resolution_min + 1

    @property
    def n_before(self) -> int:
        return self.time_before_min // self.resolution_min

    @property
    def n_after(self) -> int:
        return self.n_time - self.n_before

    @property
    def pad_h(self) -> int:
        return (PAD_TARGET - self.n_space) // 2

    @property
    def pad_w(self) -> int:
        return (PAD_TARGET - self.n_time) // 2


# ---------------------------------------------------------------------------
# Matrix-only U-Net (unet_matrix architecture)
# ---------------------------------------------------------------------------


def base_in_channels(input_type: str) -> int:
    """Canonical U-Net input has exactly three channels.

    The retained input contract is ``vd+hist`` with observed VD speed,
    observed-VD validity, and nearest-space-filled causal historical speed.
    The former fourth historical-validity channel is intentionally unsupported.
    """
    if input_type != "vd+hist":
        raise ValueError(
            "U-Net input is locked to 'vd+hist' (3 channels: VD speed, "
            "VD validity, causal historical speed)"
        )
    return 3


def _check_backbone(backbone: str) -> None:
    if backbone not in ALL_BACKBONES:
        raise ValueError(f"Unsupported backbone: {backbone}. Choose from {ALL_BACKBONES}")


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        activation: nn.Module | None = None,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            activation if activation is not None else nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    """Two-convolution residual block used by the larger UNet v2 backbone."""

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(self.drop(y))
        return self.act(y + self.skip(x))


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel gate for the bottleneck."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class AttentionGate(nn.Module):
    """Skip-connection gate conditioned on the decoder feature map."""

    def __init__(self, skip_ch: int, gate_ch: int, inter_ch: int) -> None:
        super().__init__()
        self.skip_proj = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.gate_proj = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_ch, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return skip * self.psi(self.skip_proj(skip) + self.gate_proj(gate))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.gate = AttentionGate(skip_ch=skip_ch, gate_ch=out_ch, inter_ch=max(out_ch // 2, 16))
        self.fuse = nn.Sequential(
            ResidualBlock(out_ch + skip_ch, out_ch, dropout=dropout),
            ResidualBlock(out_ch, out_ch, dropout=dropout),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = self.gate(skip, x)
        return self.fuse(torch.cat([x, skip], dim=1))


class ResidualAttentionCore(nn.Module):
    """Residual double-conv UNet core with gated skip connections."""

    bottleneck_channels = 512

    def __init__(self, in_ch: int, *, dropout: float = 0.10) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            ResidualBlock(in_ch, 64, stride=2),
            ResidualBlock(64, 64),
        )
        self.enc2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )
        self.enc3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, dropout=dropout),
            ResidualBlock(256, 256, dropout=dropout),
        )
        self.enc4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2, dropout=dropout),
            ResidualBlock(512, 512, dropout=dropout),
        )
        self.bottleneck = nn.Sequential(
            ResidualBlock(512, 512, dropout=dropout),
            SEBlock(512),
            ResidualBlock(512, 512, dropout=dropout),
        )
        self.dec4 = DecoderBlock(512, 256, 256, dropout=dropout)
        self.dec3 = DecoderBlock(256, 128, 128, dropout=dropout * 0.5)
        self.dec2 = DecoderBlock(128, 64, 64, dropout=dropout * 0.5)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )
        self.dec1 = ResidualBlock(64, 64)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return e1, e2, e3, e4

    def decode(
        self,
        bottleneck: torch.Tensor,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
    ) -> torch.Tensor:
        d4 = self.dec4(bottleneck, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        return self.dec1(self.up1(d2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1, e2, e3, e4 = self.encode(x)
        b = self.bottleneck(e4)
        return self.decode(b, e1, e2, e3), b


class SimpleUNet(nn.Module):
    """Four-layer encoder-decoder U-Net; no incident features.

    Input:  (B, 3, PAD_TARGET, PAD_TARGET)  — VD speed, VD validity,
            nearest-space-filled causal historical speed
    Output: (B, n_space, n_after)           — sigmoid-normalized predicted speed
    """

    def __init__(
        self,
        cfg: WindowConfig,
        input_type: str = "vd+hist",
        in_channels: int | None = None,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_type = input_type
        canonical_in_ch = base_in_channels(input_type)
        in_ch = canonical_in_ch if in_channels is None else int(in_channels)
        if in_ch != canonical_in_ch:
            raise ValueError(
                f"U-Net input is locked to {canonical_in_ch} channels, got {in_ch}"
            )
        if base_channels < 8:
            raise ValueError(f"base_channels must be >= 8, got {base_channels}")
        c1, c2, c3, c4 = (base_channels * factor for factor in (1, 2, 4, 8))
        self.base_channels = int(base_channels)

        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, c1, 4, stride=2, padding=1), nn.BatchNorm2d(c1), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(c1, c2, 4, stride=2, padding=1), nn.BatchNorm2d(c2), nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(nn.Conv2d(c2, c3, 4, stride=2, padding=1), nn.BatchNorm2d(c3), nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(nn.Conv2d(c3, c4, 4, stride=2, padding=1), nn.BatchNorm2d(c4), nn.LeakyReLU(0.2))

        self.dec4 = nn.Sequential(nn.ConvTranspose2d(c4, c3, 4, stride=2, padding=1), nn.BatchNorm2d(c3), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(c3 * 2, c2, 4, stride=2, padding=1), nn.BatchNorm2d(c2), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(c2 * 2, c1, 4, stride=2, padding=1), nn.BatchNorm2d(c1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(c1 * 2, c1, 4, stride=2, padding=1), nn.BatchNorm2d(c1), nn.ReLU())

        self.final = nn.Sequential(nn.Conv2d(c1, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = torch.cat([self.dec4(e4), e3], dim=1)
        d3 = torch.cat([self.dec3(d4), e2], dim=1)
        d2 = torch.cat([self.dec2(d3), e1], dim=1)
        d1 = self.dec1(d2)

        out = self.final(d1)
        out = out[:, :, cfg.pad_h : cfg.pad_h + cfg.n_space, cfg.pad_w : cfg.pad_w + cfg.n_time]
        out = out.squeeze(1)
        return out[:, :, cfg.n_before:]


class ResidualAttentionUNet(nn.Module):
    """Larger matrix-only UNet v2 backbone.

    Compared with `SimpleUNet`, this uses residual double-conv stages, SE-gated
    bottleneck features, attention-gated skip connections, and light dropout.
    """

    def __init__(
        self,
        cfg: WindowConfig,
        input_type: str = "vd+hist",
        dropout: float = 0.10,
        in_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_type = input_type
        canonical_in_ch = base_in_channels(input_type)
        in_ch = canonical_in_ch if in_channels is None else int(in_channels)
        if in_ch != canonical_in_ch:
            raise ValueError(
                f"U-Net input is locked to {canonical_in_ch} channels, got {in_ch}"
            )
        self.core = ResidualAttentionCore(in_ch, dropout=dropout)
        self.final = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        dec, _ = self.core(x)
        out = self.final(dec)
        out = out[:, :, cfg.pad_h : cfg.pad_h + cfg.n_space, cfg.pad_w : cfg.pad_w + cfg.n_time]
        out = out.squeeze(1)
        return out[:, :, cfg.n_before:]


def build_matrix_unet(
    cfg: WindowConfig,
    *,
    input_type: str = "vd+hist",
    backbone: str = "simple",
    in_channels: int | None = None,
    simple_base_channels: int = 64,
) -> nn.Module:
    """Instantiate a matrix-only UNet by backbone name."""
    _check_backbone(backbone)
    if backbone == "simple":
        return SimpleUNet(
            cfg=cfg,
            input_type=input_type,
            in_channels=in_channels,
            base_channels=simple_base_channels,
        )
    if simple_base_channels != 64:
        raise ValueError("simple_base_channels only applies to backbone='simple'")
    return ResidualAttentionUNet(
        cfg=cfg,
        input_type=input_type,
        in_channels=in_channels,
    )


# ---------------------------------------------------------------------------
# Feature-fusion U-Net (unet_feature_fusion architecture)
# ---------------------------------------------------------------------------


class FeatureFusionUNet(nn.Module):
    """U-Net augmented with incident-feature injection.

    Three injection methods:
    - direct_concat: broadcast feature embedding as extra input channels
    - mid_inject:    add feature embedding to bottleneck feature map
    - attention:     gate bottleneck channels with sigmoid feature embedding

    Input:  x=(B, 3, PAD_TARGET, PAD_TARGET), feat=(B, feature_dim)
    Output: (B, n_space, n_after) sigmoid-normalized predicted speed, plus
            optional decay_curve=(B, n_after) for TIID head (None when use_tiid=False)
    """

    def __init__(
        self,
        feature_dim: int,
        method: str,
        use_tiid: bool,
        cfg: WindowConfig,
        input_type: str = "vd+hist",
        backbone: str = "simple",
    ) -> None:
        super().__init__()
        if method not in ALL_FF_METHODS:
            raise ValueError(f"Unsupported method: {method}. Choose from {ALL_FF_METHODS}")
        _check_backbone(backbone)
        self.method = method
        self.use_tiid = use_tiid
        self.cfg = cfg
        self.input_type = input_type
        self.backbone = backbone
        base_ch = base_in_channels(input_type)

        if method == "direct_concat":
            self.feature_to_input = nn.Sequential(
                nn.Linear(feature_dim, 32), nn.ReLU(), nn.Linear(32, 8), nn.Tanh()
            )
            in_ch = base_ch + 8
        else:
            in_ch = base_ch

        if method == "mid_inject":
            self.feature_to_bottleneck = nn.Linear(feature_dim, 512)
        if method == "attention":
            self.feature_to_gate = nn.Sequential(nn.Linear(feature_dim, 512), nn.Sigmoid())

        if backbone == "simple":
            self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
            self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
            self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
            self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))

            self.dec4 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
            self.dec3 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
            self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
            self.dec1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        else:
            self.core = ResidualAttentionCore(in_ch)

        self.final = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1), nn.Sigmoid())

        if self.use_tiid:
            self.decay_head = nn.Sequential(
                nn.Linear(512 + feature_dim, 256), nn.ReLU(), nn.Linear(256, cfg.n_after)
            )
            self.tiid_scale = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))

    def forward(self, x: torch.Tensor, feat: torch.Tensor):
        cfg = self.cfg

        if self.method == "direct_concat":
            feat_ch = self.feature_to_input(feat).unsqueeze(-1).unsqueeze(-1)
            feat_map = feat_ch.expand(-1, -1, x.shape[2], x.shape[3])
            x_in = torch.cat([x, feat_map], dim=1)
        else:
            x_in = x

        if self.backbone == "simple":
            e1 = self.enc1(x_in)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)

            if self.method == "mid_inject":
                e4 = e4 + self.feature_to_bottleneck(feat).unsqueeze(-1).unsqueeze(-1)
            if self.method == "attention":
                e4 = e4 * self.feature_to_gate(feat).unsqueeze(-1).unsqueeze(-1)

            d4 = torch.cat([self.dec4(e4), e3], dim=1)
            d3 = torch.cat([self.dec3(d4), e2], dim=1)
            d2 = torch.cat([self.dec2(d3), e1], dim=1)
            d1 = self.dec1(d2)
            pooled_source = e4
        else:
            e1, e2, e3, e4 = self.core.encode(x_in)
            if self.method == "mid_inject":
                e4 = e4 + self.feature_to_bottleneck(feat).unsqueeze(-1).unsqueeze(-1)
            if self.method == "attention":
                e4 = e4 * self.feature_to_gate(feat).unsqueeze(-1).unsqueeze(-1)
            bottleneck = self.core.bottleneck(e4)
            d1 = self.core.decode(bottleneck, e1, e2, e3)
            pooled_source = bottleneck

        out = self.final(d1)
        out = out[:, :, cfg.pad_h : cfg.pad_h + cfg.n_space, cfg.pad_w : cfg.pad_w + cfg.n_time]
        out = out.squeeze(1)
        base_pred = out[:, :, cfg.n_before:]

        if not self.use_tiid:
            return base_pred, None

        pooled = F.adaptive_avg_pool2d(pooled_source, 1).flatten(1)
        decay_curve = torch.sigmoid(self.decay_head(torch.cat([pooled, feat], dim=1)))
        decay_curve = torch.cummin(decay_curve, dim=1).values

        context = x[:, 0, cfg.pad_h : cfg.pad_h + cfg.n_space, cfg.pad_w : cfg.pad_w + cfg.n_before]
        baseline = context.mean(dim=(1, 2))
        penalty = decay_curve * baseline.unsqueeze(1) * torch.clamp(self.tiid_scale, min=0.0)
        penalty = penalty.unsqueeze(1).expand(-1, cfg.n_space, -1)
        pred = torch.clamp(base_pred - penalty, 0.0, 1.0)
        return pred, decay_curve


def build_feature_fusion_unet(
    feature_dim: int,
    method: str,
    use_tiid: bool,
    cfg: WindowConfig,
    *,
    input_type: str = "vd+hist",
    backbone: str = "simple",
) -> FeatureFusionUNet:
    """Instantiate a feature-fusion UNet by backbone name."""
    return FeatureFusionUNet(
        feature_dim=feature_dim,
        method=method,
        use_tiid=use_tiid,
        cfg=cfg,
        input_type=input_type,
        backbone=backbone,
    )
