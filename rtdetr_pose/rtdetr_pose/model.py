try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - optional dependency for scaffolding
    from types import SimpleNamespace

    torch = None
    nn = SimpleNamespace(Module=object)
    F = SimpleNamespace()


def rot6d_to_matrix(x):
    if torch is None:
        raise RuntimeError("torch is required for rot6d_to_matrix")
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = nn.functional.normalize(a1, dim=-1)
    b2 = nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def _sincos_1d(pos, dim, device, dtype):
    half_dim = dim // 2
    omega = torch.arange(half_dim, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / half_dim))
    out = pos[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


class SinePositionEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        if hidden_dim % 4 != 0:
            raise ValueError("hidden_dim must be divisible by 4 for 2D sin/cos")
        self.hidden_dim = hidden_dim

    def forward(self, height, width, device, dtype):
        xs = torch.arange(width, device=device, dtype=dtype)
        ys = torch.arange(height, device=device, dtype=dtype)
        grid_x = xs.repeat(height)
        grid_y = ys.repeat_interleave(width)
        dim_half = self.hidden_dim // 2
        pos_y = _sincos_1d(grid_y, dim_half, device, dtype)
        pos_x = _sincos_1d(grid_x, dim_half, device, dtype)
        pos = torch.cat([pos_y, pos_x], dim=1)
        return pos.unsqueeze(0)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=0.5, shortcut=True):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvNormAct(in_channels, hidden, kernel_size=1)
        self.conv2 = ConvNormAct(hidden, out_channels, kernel_size=3, padding=1)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.shortcut:
            y = y + x
        return y


class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1, expansion=0.5):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvNormAct(in_channels, hidden, kernel_size=1)
        self.conv2 = ConvNormAct(in_channels, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            *[Bottleneck(hidden, hidden, expansion=1.0) for _ in range(num_blocks)]
        )
        self.conv3 = ConvNormAct(hidden * 2, out_channels, kernel_size=1)

    def forward(self, x):
        y1 = self.blocks(self.conv1(x))
        y2 = self.conv2(x)
        return self.conv3(torch.cat([y1, y2], dim=1))


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=5):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
        self.conv2 = ConvNormAct(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class CSPResNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        stem_channels=32,
        stage_channels=(64, 128, 256),
        stage_blocks=(1, 2, 2),
        use_sppf=True,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, stem_channels, kernel_size=3, stride=2, padding=1),
            ConvNormAct(stem_channels, stem_channels, kernel_size=3, padding=1),
            ConvNormAct(stem_channels, stem_channels * 2, kernel_size=3, padding=1),
        )
        in_ch = stem_channels * 2
        stages = []
        for out_ch, num_blocks in zip(stage_channels, stage_blocks):
            stages.append(
                nn.Sequential(
                    ConvNormAct(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    CSPBlock(out_ch, out_ch, num_blocks=num_blocks),
                )
            )
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.sppf = SPPF(stage_channels[-1], stage_channels[-1]) if use_sppf else None

    def forward(self, x):
        x = self.stem(x)
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        if self.sppf is not None:
            outputs[-1] = self.sppf(outputs[-1])
        return outputs


class FPNPAN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lateral = nn.ModuleList(
            [ConvNormAct(ch, out_channels, kernel_size=1) for ch in in_channels]
        )
        self.fpn_convs = nn.ModuleList(
            [ConvNormAct(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels]
        )
        self.downsample = nn.ModuleList(
            [ConvNormAct(out_channels, out_channels, kernel_size=3, stride=2, padding=1) for _ in in_channels[1:]]
        )
        self.pan_convs = nn.ModuleList(
            [ConvNormAct(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels[1:]]
        )

    def forward(self, features):
        feats = [lat(feat) for lat, feat in zip(self.lateral, features)]
        p5 = feats[-1]
        p4 = feats[-2] + F.interpolate(p5, size=feats[-2].shape[-2:], mode="nearest")
        p3 = feats[-3] + F.interpolate(p4, size=feats[-3].shape[-2:], mode="nearest")
        p3 = self.fpn_convs[0](p3)
        p4 = self.fpn_convs[1](p4)
        p5 = self.fpn_convs[2](p5)
        n4 = self.pan_convs[0](p4 + self.downsample[0](p3))
        n5 = self.pan_convs[1](p5 + self.downsample[1](n4))
        return [p3, n4, n5]


class HybridEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=1, nhead=8, dim_feedforward=None):
        super().__init__()
        self.fpn = FPNPAN(in_channels=in_channels, out_channels=hidden_dim)
        self.num_layers = num_layers
        dim_feedforward = dim_feedforward or hidden_dim * 4
        if num_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=0.0,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            self.encoder = None

    def forward(self, features, pos_embed):
        features = self.fpn(features)
        memory = torch.cat([feat.flatten(2).permute(0, 2, 1) for feat in features], dim=1)
        pos = torch.cat(pos_embed, dim=1)
        memory = memory + pos
        if self.encoder is not None:
            memory = self.encoder(memory)
        return memory, features


class RTDETRDecoder(nn.Module):
    def __init__(self, hidden_dim, num_layers=3, nhead=8, dim_feedforward=None):
        super().__init__()
        dim_feedforward = dim_feedforward or hidden_dim * 4
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, tgt, memory):
        return self.decoder(tgt, memory)


class HeadFast(nn.Module):
    def __init__(self, hidden_dim, num_classes, use_uncertainty=False):
        super().__init__()
        self.cls = nn.Linear(hidden_dim, num_classes)
        self.box = nn.Linear(hidden_dim, 4)
        self.log_z = nn.Linear(hidden_dim, 1)
        self.rot6d = nn.Linear(hidden_dim, 6)
        self.use_uncertainty = use_uncertainty
        if use_uncertainty:
            self.log_sigma_z = nn.Linear(hidden_dim, 1)
            self.log_sigma_rot = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = {
            "logits": self.cls(x),
            "bbox": self.box(x),
            "log_z": self.log_z(x),
            "rot6d": self.rot6d(x),
        }
        if self.use_uncertainty:
            out["log_sigma_z"] = self.log_sigma_z(x)
            out["log_sigma_rot"] = self.log_sigma_rot(x)
        return out


class CenterOffsetHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.offset = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        return self.offset(x)


class GlobalKHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.delta = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        return self.delta(x)


class RTDETRPose(nn.Module):
    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        use_uncertainty=False,
        stem_channels=32,
        backbone_channels=(64, 128, 256),
        stage_blocks=(1, 2, 2),
        num_encoder_layers=1,
        num_decoder_layers=3,
        nhead=8,
        encoder_dim_feedforward=None,
        decoder_dim_feedforward=None,
    ):
        super().__init__()
        self.backbone = CSPResNet(
            stem_channels=stem_channels,
            stage_channels=backbone_channels,
            stage_blocks=stage_blocks,
        )
        self.position = SinePositionEmbedding(hidden_dim)
        self.encoder = HybridEncoder(
            in_channels=backbone_channels,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            nhead=nhead,
            dim_feedforward=encoder_dim_feedforward,
        )
        self.decoder = RTDETRDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            nhead=nhead,
            dim_feedforward=decoder_dim_feedforward,
        )
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.head = HeadFast(hidden_dim, num_classes, use_uncertainty=use_uncertainty)
        self.offset_head = CenterOffsetHead(hidden_dim)
        self.k_head = GlobalKHead(hidden_dim)

    def forward(self, x):
        if torch is None:
            raise RuntimeError("torch is required for RTDETRPose")
        batch = x.shape[0]
        feats = self.backbone(x)
        pos_embed = []
        for feat in feats:
            _, _, height, width = feat.shape
            pos = self.position(height, width, feat.device, feat.dtype).repeat(batch, 1, 1)
            pos_embed.append(pos)
        memory, _ = self.encoder(feats, pos_embed)
        queries = self.query_embed.weight.unsqueeze(0).repeat(batch, 1, 1)
        tgt = torch.zeros_like(queries) + queries
        dec = self.decoder(tgt, memory)
        out = self.head(dec)
        out["offsets"] = self.offset_head(dec)
        out["k_delta"] = self.k_head(dec.mean(dim=1))
        return out
