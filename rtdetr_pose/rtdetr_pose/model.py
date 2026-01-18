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
    def __init__(self, num_classes=80, hidden_dim=256, num_queries=300, use_uncertainty=False):
        super().__init__()
        self.backbone = nn.Identity()
        self.decoder = nn.Linear(hidden_dim, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.head = HeadFast(hidden_dim, num_classes, use_uncertainty=use_uncertainty)
        self.offset_head = CenterOffsetHead(hidden_dim)
        self.k_head = GlobalKHead(hidden_dim)

    def forward(self, x):
        if torch is None:
            raise RuntimeError("torch is required for RTDETRPose")
        batch = x.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).repeat(batch, 1, 1)
        dec = self.decoder(queries)
        out = self.head(dec)
        out["offsets"] = self.offset_head(dec)
        out["k_delta"] = self.k_head(dec.mean(dim=1))
        return out
