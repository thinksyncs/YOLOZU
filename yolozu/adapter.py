class ModelAdapter:
    def predict(self, records):
        raise NotImplementedError


class DummyAdapter(ModelAdapter):
    def predict(self, records):
        return [
            {"image": record["image"], "detections": []} for record in records
        ]


class PrecomputedAdapter(ModelAdapter):
    """Adapter that returns detections loaded from a JSON file.

    This is useful when you run real inference elsewhere (torch/TRT/etc.)
    and want to evaluate the pipeline in this repo without heavyweight deps.

    Supported JSON formats:

    1) List of per-image entries:
       [{"image": "/abs/or/rel/path.jpg", "detections": [...]}, ...]

    2) Dict with top-level key:
       {"predictions": [ ...same as above... ]}

    3) Dict mapping image->detections:
       {"/path.jpg": [...], "000000000009.jpg": [...]}  # values are detections
    """

    def __init__(self, predictions_path):
        from pathlib import Path

        self.predictions_path = str(predictions_path)
        self._path = Path(predictions_path)
        self._index = None

    def _load(self):
        from .predictions import load_predictions_index

        self._index = load_predictions_index(self._path)

    def predict(self, records):
        if self._index is None:
            self._load()

        outputs = []
        for record in records:
            image = record["image"]
            dets = self._index.get(image)
            if dets is None:
                base = str(image).split("/")[-1]
                dets = self._index.get(base, [])
            outputs.append({"image": image, "detections": dets})
        return outputs


class RTDETRPoseAdapter(ModelAdapter):
    """Adapter that runs the RT-DETR pose scaffold (optional dependency).

    This adapter is intentionally dependency-light at import time.
    If torch isn't installed, it raises a clear RuntimeError on first use.

    Output schema per image:
      {
        "image": <path>,
        "detections": [
          {
            "class_id": int,
            "score": float,
            "bbox": {"cx": float, "cy": float, "w": float, "h": float},
            "log_z": float,
            "rot6d": [float, ...],
            "offsets": [float, float],
            "k_delta": [float, float, float, float],
          },
          ...
        ]
      }
    """

    def __init__(
        self,
        config_path="rtdetr_pose/configs/base.json",
        checkpoint_path=None,
        device="cpu",
        image_size=(320, 320),
        score_threshold=0.3,
        max_detections=50,
    ):
        self.config_path = str(config_path)
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.image_size = tuple(image_size)
        self.score_threshold = float(score_threshold)
        self.max_detections = int(max_detections)
        self._backend = None

    def _ensure_backend(self):
        if self._backend is not None:
            return

        # Ensure the in-repo rtdetr_pose package is importable without installation.
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root / "rtdetr_pose"))

        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "RTDETRPoseAdapter requires 'torch'. Install PyTorch (and optionally torchvision) to enable real inference."
            ) from exc

        from PIL import Image

        from rtdetr_pose.config import load_config
        from rtdetr_pose.model import RTDETRPose

        cfg = load_config(self.config_path)
        model = RTDETRPose(
            num_classes=cfg.model.num_classes,
            hidden_dim=cfg.model.hidden_dim,
            num_queries=cfg.model.num_queries,
            use_uncertainty=cfg.model.use_uncertainty,
            stem_channels=getattr(cfg.model, "stem_channels", 32),
            backbone_channels=tuple(getattr(cfg.model, "backbone_channels", (64, 128, 256))),
            stage_blocks=tuple(getattr(cfg.model, "stage_blocks", (1, 2, 2))),
            num_encoder_layers=getattr(cfg.model, "num_encoder_layers", 1),
            num_decoder_layers=cfg.model.num_decoder_layers,
            nhead=cfg.model.nhead,
            encoder_dim_feedforward=getattr(cfg.model, "encoder_dim_feedforward", None),
            decoder_dim_feedforward=getattr(cfg.model, "decoder_dim_feedforward", None),
        ).eval()

        if self.checkpoint_path:
            state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                model_state = model.state_dict()
                filtered = {
                    k: v
                    for k, v in state.items()
                    if k in model_state and hasattr(v, "shape") and v.shape == model_state[k].shape
                }
                model.load_state_dict(filtered, strict=False)
            else:
                model.load_state_dict(state, strict=False)

        model.to(self.device)

        def preprocess(path):
            img = Image.open(path).convert("RGB")
            img = img.resize(self.image_size)
            x = torch.tensor(list(img.getdata()), dtype=torch.float32)
            x = x.view(self.image_size[1], self.image_size[0], 3).permute(2, 0, 1)
            x = x / 255.0
            return x.unsqueeze(0)

        self._backend = {"torch": torch, "model": model, "preprocess": preprocess}

    def predict(self, records):
        self._ensure_backend()
        torch = self._backend["torch"]
        model = self._backend["model"]
        preprocess = self._backend["preprocess"]

        outputs = []
        for record in records:
            image_path = record["image"]
            x = preprocess(image_path).to(self.device)
            with torch.no_grad():
                out = model(x)

            logits = out["logits"][0]
            bbox = out["bbox"][0]
            log_z = out["log_z"][0]
            rot6d = out["rot6d"][0]
            offsets = out["offsets"][0]
            k_delta = out["k_delta"][0]

            probs = torch.softmax(logits, dim=-1)
            scores, class_ids = torch.max(probs, dim=-1)
            k = min(self.max_detections, int(scores.shape[0]))
            top_scores, top_idx = torch.topk(scores, k=k)

            detections = []
            for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
                if score < self.score_threshold:
                    continue
                cls_id = int(class_ids[idx].item())
                box = torch.sigmoid(bbox[idx]).tolist()
                det = {
                    "class_id": cls_id,
                    "score": float(score),
                    "bbox": {"cx": float(box[0]), "cy": float(box[1]), "w": float(box[2]), "h": float(box[3])},
                    "log_z": float(log_z[idx].item()),
                    "rot6d": [float(v) for v in rot6d[idx].tolist()],
                    "offsets": [float(v) for v in offsets[idx].tolist()],
                    "k_delta": [float(v) for v in k_delta.tolist()],
                }
                detections.append(det)

            outputs.append({"image": image_path, "detections": detections})
        return outputs
