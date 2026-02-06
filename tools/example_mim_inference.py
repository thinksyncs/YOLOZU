#!/usr/bin/env python3
"""Example: Using masked reconstruction branch in inference loop.

This script demonstrates how to use the MIM branch with entropy loss
and geometric consistency during inference for test-time adaptation.
"""
import argparse
import sys
from pathlib import Path

# Add rtdetr_pose to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "rtdetr_pose"))

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit("torch is required; install requirements-test.txt") from exc

from rtdetr_pose.model import RTDETRPose
from rtdetr_pose.losses import mim_reconstruction_loss, entropy_loss


def generate_block_mask(height, width, patch_size=16, mask_prob=0.6):
    """Generate block mask for feature masking.
    
    Args:
        height: Feature map height
        width: Feature map width
        patch_size: Size of each block
        mask_prob: Probability of masking each block
        
    Returns:
        mask: (H, W) boolean mask
    """
    grid_h = max(1, (height + patch_size - 1) // patch_size)
    grid_w = max(1, (width + patch_size - 1) // patch_size)
    
    # Random mask at grid level
    mask_grid = torch.rand(grid_h, grid_w) < mask_prob
    
    # Repeat to full resolution
    mask = mask_grid.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
    return mask[:height, :width]


def create_geometry_input(mask, depth_obj):
    """Create geometry input tensor from mask and depth.
    
    Args:
        mask: (H, W) binary mask
        depth_obj: (H, W) object-only depth in meters
        
    Returns:
        geom_input: (1, 2, H, W) geometry tensor
    """
    # Reference depth from median
    valid_depth = depth_obj[mask > 0]
    if len(valid_depth) == 0:
        z_ref = torch.tensor(1.0)
    else:
        z_ref = valid_depth.median()
    
    # Normalize depth: M * log(D / z_ref)
    eps = 1e-6
    depth_norm = mask.float() * (torch.log(depth_obj + eps) - torch.log(z_ref + eps))
    
    # Stack: [mask, normalized_depth]
    geom_input = torch.stack([mask.float(), depth_norm], dim=0).unsqueeze(0)
    return geom_input


def inference_with_mim(model, image, geom_input=None, feature_mask=None):
    """Run inference with MIM branch enabled.
    
    Args:
        model: RTDETRPose model with enable_mim=True
        image: (1, 3, H, W) input image
        geom_input: (1, 2, H', W') optional geometry input
        feature_mask: (H', W') optional feature mask
        
    Returns:
        outputs: dict with detection outputs and MIM outputs
    """
    with torch.no_grad():
        outputs = model(
            image,
            geom_input=geom_input,
            feature_mask=feature_mask,
            return_mim=True
        )
    return outputs


def test_time_adapt(model, image, geom_input, num_steps=3, lr=1e-4):
    """Perform test-time adaptation using MIM loss.
    
    Args:
        model: RTDETRPose model with enable_mim=True
        image: (1, 3, H, W) input image
        geom_input: (1, 2, H', W') geometry input
        num_steps: Number of adaptation steps
        lr: Learning rate
        
    Returns:
        losses: List of loss values at each step
    """
    # Select parameters (e.g., batch norm only)
    params = []
    for name, param in model.named_parameters():
        if 'bn' in name.lower() or 'norm' in name.lower():
            if param.requires_grad:
                params.append(param)
    
    if not params:
        print("Warning: No parameters selected for adaptation")
        return []
    
    optimizer = torch.optim.Adam(params, lr=lr)
    model.train()
    
    losses = []
    for step in range(num_steps):
        # Generate new mask for each step
        feature_mask = generate_block_mask(160, 160, patch_size=16, mask_prob=0.6)
        
        # Forward with MIM
        outputs = model(
            image,
            geom_input=geom_input,
            feature_mask=feature_mask,
            return_mim=True
        )
        
        # Compute MIM loss
        mim = outputs["mim"]
        loss_mim = mim_reconstruction_loss(
            mim["recon_feat"],
            mim["teacher_feat"],
            mask=mim["mask"]
        )
        
        # Compute entropy loss
        loss_ent = mim["entropy"]
        
        # Total loss
        loss = loss_mim + 0.1 * loss_ent
        
        # Backward and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append({
            "step": step,
            "total": float(loss.item()),
            "mim": float(loss_mim.item()),
            "entropy": float(loss_ent.item()),
        })
        
        print(f"  Step {step+1}/{num_steps}: loss={loss.item():.4f} "
              f"(mim={loss_mim.item():.4f}, ent={loss_ent.item():.4f})")
    
    model.eval()
    return losses


def main():
    parser = argparse.ArgumentParser(description="MIM inference example")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Model hidden dimension")
    parser.add_argument("--num-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--image-size", type=int, default=640, help="Input image size")
    parser.add_argument("--ttt", action="store_true", help="Enable test-time training")
    parser.add_argument("--ttt-steps", type=int, default=3, help="TTT steps")
    parser.add_argument("--ttt-lr", type=float, default=1e-4, help="TTT learning rate")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda:0)")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print("Creating model with MIM enabled...")
    model = RTDETRPose(
        num_classes=80,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        enable_mim=True,
    )
    model = model.to(device)
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"MIM enabled: {model.enable_mim}")
    
    # Create dummy inputs
    print(f"\nCreating dummy inputs (image size: {args.image_size})...")
    image = torch.randn(1, 3, args.image_size, args.image_size).to(device)
    
    # Create geometry input (mask + depth)
    feat_size = args.image_size // 4  # Assuming 4x downsampling
    mask = torch.rand(feat_size, feat_size) > 0.3
    depth = torch.rand(feat_size, feat_size) * 5.0 + 0.5
    geom_input = create_geometry_input(mask, depth).to(device)
    
    # Generate feature mask
    feature_mask = generate_block_mask(feat_size, feat_size, patch_size=16, mask_prob=0.6)
    
    print(f"Geometry input shape: {geom_input.shape}")
    print(f"Feature mask shape: {feature_mask.shape}")
    print(f"Feature mask ratio: {feature_mask.float().mean():.2%}")
    
    # Inference without TTT
    print("\nRunning inference with MIM...")
    outputs = inference_with_mim(model, image, geom_input, feature_mask)
    
    print(f"Outputs keys: {list(outputs.keys())}")
    print(f"MIM outputs keys: {list(outputs['mim'].keys())}")
    print(f"Detection logits shape: {outputs['logits'].shape}")
    print(f"Reconstructed features shape: {outputs['mim']['recon_feat'].shape}")
    print(f"Teacher features shape: {outputs['mim']['teacher_feat'].shape}")
    print(f"Entropy: {outputs['mim']['entropy'].item():.4f}")
    
    # Test-time adaptation (optional)
    if args.ttt:
        print(f"\nRunning test-time adaptation ({args.ttt_steps} steps)...")
        losses = test_time_adapt(
            model,
            image,
            geom_input,
            num_steps=args.ttt_steps,
            lr=args.ttt_lr
        )
        
        print("\nAdaptation complete. Running inference with adapted model...")
        outputs_adapted = inference_with_mim(model, image, geom_input, feature_mask)
        print(f"Entropy after adaptation: {outputs_adapted['mim']['entropy'].item():.4f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
