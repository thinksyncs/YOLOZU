import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit("torch is required; install requirements-test.txt") from exc

from rtdetr_pose.dataset import build_manifest
from rtdetr_pose.losses import Losses
from rtdetr_pose.training import build_query_aligned_targets
from rtdetr_pose.model import RTDETRPose


class ManifestDataset(Dataset):
    def __init__(
        self,
        records,
        num_queries,
        num_classes,
        image_size,
        seed,
        use_matcher,
        synthetic_pose,
    ):
        self.records = records
        self.num_queries = int(num_queries)
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.use_matcher = bool(use_matcher)
        self.synthetic_pose = bool(synthetic_pose)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # We intentionally do NOT decode JPEGs here (keeps deps minimal).
        # This is a training-loop scaffold: it exercises model/loss/optimizer plumbing.
        gen = torch.Generator()
        gen.manual_seed(self.seed + int(idx))
        image = torch.rand(3, self.image_size, self.image_size, generator=gen)

        instances = record.get("labels") or []
        if self.use_matcher:
            gt_labels = []
            gt_bbox = []
            gt_z = []
            gt_R = []
            for inst in instances:
                class_id = int(inst.get("class_id", -1))
                if not (0 <= class_id < self.num_classes):
                    continue
                bb = inst.get("bbox") or {}
                gt_labels.append(class_id)
                gt_bbox.append(
                    [
                        float(bb.get("cx", 0.0)),
                        float(bb.get("cy", 0.0)),
                        float(bb.get("w", 0.0)),
                        float(bb.get("h", 0.0)),
                    ]
                )

                if self.synthetic_pose:
                    # Simple synthetic depth + random rotation to exercise extra loss terms.
                    gt_z.append(float(torch.rand((), generator=gen) * 0.9 + 0.1))
                    a = torch.randn(3, 3, generator=gen)
                    q, _ = torch.linalg.qr(a)
                    if torch.det(q) < 0:
                        q[:, 0] = -q[:, 0]
                    gt_R.append(q)
            return {
                "image": image,
                "targets": {
                    "gt_labels": torch.tensor(gt_labels, dtype=torch.long),
                    "gt_bbox": torch.tensor(gt_bbox, dtype=torch.float32),
                    "gt_z": torch.tensor(gt_z, dtype=torch.float32).unsqueeze(-1)
                    if self.synthetic_pose
                    else torch.zeros((0, 1), dtype=torch.float32),
                    "gt_R": torch.stack(gt_R, dim=0)
                    if (self.synthetic_pose and gt_R)
                    else torch.zeros((0, 3, 3), dtype=torch.float32),
                },
            }

        labels = torch.full((self.num_queries,), -1, dtype=torch.long)
        bbox = torch.zeros((self.num_queries, 4), dtype=torch.float32)
        for qi, inst in enumerate(instances[: self.num_queries]):
            class_id = int(inst.get("class_id", -1))
            if 0 <= class_id < self.num_classes:
                labels[qi] = class_id
            bb = inst.get("bbox") or {}
            bbox[qi, 0] = float(bb.get("cx", 0.0))
            bbox[qi, 1] = float(bb.get("cy", 0.0))
            bbox[qi, 2] = float(bb.get("w", 0.0))
            bbox[qi, 3] = float(bb.get("h", 0.0))

        return {"image": image, "targets": {"labels": labels, "bbox": bbox}}


def collate(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = [item["targets"] for item in batch]
    return images, targets




def main():
    parser = argparse.ArgumentParser(description="Minimal RTDETRPose training scaffold (CPU).")
    parser.add_argument("--dataset-root", type=str, default="", help="Path to data/coco128")
    parser.add_argument("--split", type=str, default="train2017")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=30, help="Cap steps per epoch")
    parser.add_argument("--log-every", type=int, default=10, help="Print every N steps")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-matcher", action="store_true", help="Use Hungarian matching")
    parser.add_argument("--cost-cls", type=float, default=1.0)
    parser.add_argument("--cost-bbox", type=float, default=5.0)
    parser.add_argument("--cost-z", type=float, default=0.0, help="Optional matching cost for depth")
    parser.add_argument(
        "--cost-rot",
        type=float,
        default=0.0,
        help="Optional matching cost for rotation (geodesic angle)",
    )
    parser.add_argument(
        "--synthetic-pose",
        action="store_true",
        help="Generate synthetic z/R GT per instance (scaffold only)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        dataset_root = repo_root / "data" / "coco128"
        if not dataset_root.exists():
            dataset_root = repo_root.parent / "data" / "coco128"

    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest.get("images") or []
    if not records:
        raise SystemExit(
            f"No records found under {dataset_root}. "
            "Fetch coco128 first: bash tools/fetch_coco128.sh"
        )

    stats = {
        "mask": 0,
        "depth": 0,
        "pose": 0,
        "intrinsics": 0,
        "cad_points": 0,
    }
    for rec in records:
        if rec.get("mask_path") is not None:
            stats["mask"] += 1
        if rec.get("depth_path") is not None:
            stats["depth"] += 1
        if rec.get("R_gt") is not None or rec.get("t_gt") is not None or rec.get("pose") is not None:
            stats["pose"] += 1
        if rec.get("K_gt") is not None or rec.get("intrinsics") is not None:
            stats["intrinsics"] += 1
        if rec.get("cad_points") is not None:
            stats["cad_points"] += 1
    print(
        "dataset_stats "
        + " ".join(f"{key}={value}" for key, value in sorted(stats.items()))
    )

    ds = ManifestDataset(
        records,
        num_queries=args.num_queries,
        num_classes=args.num_classes,
        image_size=args.image_size,
        seed=args.seed,
        use_matcher=args.use_matcher,
        synthetic_pose=args.synthetic_pose,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
    )

    model = RTDETRPose(
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=2,
        nhead=4,
    )
    losses_fn = Losses()

    device = torch.device("cpu")
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(int(args.epochs)):
        running = 0.0
        steps = 0
        for images, targets in loader:
            images = images.to(device)

            out = model(images)

            if args.use_matcher:
                aligned = build_query_aligned_targets(
                    out["logits"],
                    out["bbox"],
                    targets,
                    num_queries=args.num_queries,
                    cost_cls=args.cost_cls,
                    cost_bbox=args.cost_bbox,
                    log_z_pred=out.get("log_z"),
                    rot6d_pred=out.get("rot6d"),
                    cost_z=args.cost_z,
                    cost_rot=args.cost_rot,
                )
                out = dict(out)
                # For box regression we train in normalized space.
                out["bbox"] = aligned["bbox_norm"]
                targets = {
                    "labels": aligned["labels"],
                    "bbox": aligned["bbox"],
                    "mask": aligned["mask"],
                    "z_gt": aligned["z_gt"],
                    "R_gt": aligned["R_gt"],
                    "offsets": aligned["offsets"],
                }
            else:
                # legacy padded targets
                targets = {
                    "labels": torch.stack([t["labels"] for t in targets], dim=0).to(device),
                    "bbox": torch.stack([t["bbox"] for t in targets], dim=0).to(device),
                }

            loss_dict = losses_fn(out, targets)
            loss = loss_dict["loss"]

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running += float(loss.detach().cpu())
            steps += 1

            if steps == 1 or (args.log_every and steps % int(args.log_every) == 0):
                avg = running / steps
                print(f"epoch={epoch} step={steps} loss={avg:.4f}")

            if args.max_steps and steps >= int(args.max_steps):
                break

        avg = running / max(1, steps)
        print(f"epoch={epoch} done steps={steps} loss={avg:.4f}")


if __name__ == "__main__":
    main()
