try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency for scaffolding
    torch = None
    nn = None


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
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)
        dim_half = self.hidden_dim // 2
        pos_y = _sincos_1d(grid_y, dim_half, device, dtype)
        pos_x = _sincos_1d(grid_x, dim_half, device, dtype)
        pos = torch.cat([pos_y, pos_x], dim=1)
        return pos.unsqueeze(0)


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


class ConvBackbone(nn.Module):
    def __init__(self, in_channels=3, channels=(64, 128, 256), out_channels=256):
        super().__init__()
        layers = []
        prev = in_channels
        for ch in channels:
            layers.append(nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU(inplace=True))
            prev = ch
        self.body = nn.Sequential(*layers)
        self.proj = nn.Conv2d(prev, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.body(x)
        return self.proj(x)


class SimpleDecoder(nn.Module):
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


class RTDETRPose(nn.Module):
    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        use_uncertainty=False,
        backbone_channels=(64, 128, 256),
        num_decoder_layers=3,
        nhead=8,
    ):
        super().__init__()
        self.backbone = ConvBackbone(channels=backbone_channels, out_channels=hidden_dim)
        self.position = SinePositionEmbedding(hidden_dim)
        self.decoder = SimpleDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            nhead=nhead,
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
        _, _, height, width = feats.shape
        memory = feats.flatten(2).permute(0, 2, 1)
        pos = self.position(height, width, feats.device, feats.dtype)
        memory = memory + pos
        queries = self.query_embed.weight.unsqueeze(0).repeat(batch, 1, 1)
        tgt = torch.zeros_like(queries) + queries
        dec = self.decoder(tgt, memory)
        out = self.head(dec)
        out["offsets"] = self.offset_head(dec)
        out["k_delta"] = self.k_head(dec.mean(dim=1))
        return out
