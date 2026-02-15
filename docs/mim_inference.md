# Masked Reconstruction Branch for Inference

This document describes how to use the masked reconstruction branch with entropy loss and geometric consistency in the inference loop.

## Overview

The masked reconstruction branch adds the following capabilities to the RT-DETR Pose model:

1. **Geometry-Aligned MIM (Masked Image Modeling)**: Reconstructs masked features using geometry-derived teacher features
2. **Entropy Loss**: Minimizes prediction entropy for geometric consistency
3. **Geometric Consistency**: Ensures reconstructed features match geometric priors from mask and depth

## Components

### RenderTeacher

Small CNN that processes geometry tensors (mask + normalized depth) to produce teacher features:

```python
from rtdetr_pose.model import RenderTeacher

teacher = RenderTeacher(hidden_dim=256, in_channels=2)
# Input: (B, 2, H, W) where channels are [mask, normalized_depth]
# Output: (B, 256, H, W) geometry-derived features
```

### DecoderMIM

Decoder that reconstructs masked features:

```python
from rtdetr_pose.model import DecoderMIM

decoder = DecoderMIM(hidden_dim=256)
# Input: (B, 256, H, W) masked features
# Output: (B, 256, H, W) reconstructed features
```

## Usage

### Training with MIM

Enable MIM when creating the model:

```python
from rtdetr_pose.model import RTDETRPose

model = RTDETRPose(
    num_classes=80,
    hidden_dim=256,
    num_queries=300,
    enable_mim=True,  # Enable MIM branch
)
```

### Inference with MIM

Use the MIM branch during inference for test-time adaptation or geometric consistency checks:

```python
import torch
import torch.nn.functional as F

# Prepare inputs
image = torch.randn(1, 3, 640, 640)  # RGB image

# Prepare geometry input (mask + normalized depth)
mask = torch.rand(1, 1, 160, 160) > 0.5  # Binary mask
depth = torch.rand(1, 1, 160, 160) * 5.0  # Depth in meters

# Normalize depth
z_ref = depth[mask].median()
depth_norm = mask.float() * (torch.log(depth + 1e-6) - torch.log(z_ref + 1e-6))
geom_input = torch.cat([mask.float(), depth_norm], dim=1)  # (1, 2, H, W)

# Generate feature mask for MIM
feature_mask = torch.rand(160, 160) < 0.6  # Mask 60% of features

# Forward pass with MIM
outputs = model(
    image,
    geom_input=geom_input,
    feature_mask=feature_mask,
    return_mim=True  # Enable MIM outputs
)

# Standard outputs
logits = outputs["logits"]
bbox = outputs["bbox"]
log_z = outputs["log_z"]
rot6d = outputs["rot6d"]

# MIM outputs
mim = outputs["mim"]
recon_feat = mim["recon_feat"]  # Reconstructed features
teacher_feat = mim["teacher_feat"]  # Teacher features
entropy = mim["entropy"]  # Prediction entropy
```

### Computing Losses

```python
from rtdetr_pose.losses import Losses, mim_reconstruction_loss, entropy_loss

# Create loss module with MIM weights
loss_module = Losses(weights={
    "cls": 1.0,
    "box": 1.0,
    "z": 1.0,
    "rot": 1.0,
    "mim": 0.1,  # MIM reconstruction weight
    "entropy": 0.01,  # Entropy weight
})

# Compute losses
losses = loss_module(outputs, targets)

print(f"Total loss: {losses['loss']:.4f}")
print(f"MIM loss: {losses['loss_mim']:.4f}")
print(f"Entropy loss: {losses['loss_entropy']:.4f}")
```

## Test-Time Training (TTT) Integration

The MIM branch can be used for test-time adaptation:

```python
import torch.optim as optim

# Create model with MIM enabled
model = RTDETRPose(enable_mim=True)
model.train()

# Select parameters to update (e.g., batch norm only)
params = [p for n, p in model.named_parameters() if 'bn' in n or 'norm' in n]
optimizer = optim.Adam(params, lr=1e-4)

# Test-time adaptation loop
for test_image in test_images:
    # Prepare geometry input from predictions or priors
    geom_input = prepare_geometry(test_image)
    feature_mask = generate_mask(prob=0.6)
    
    # Forward with MIM
    outputs = model(test_image, geom_input=geom_input, 
                   feature_mask=feature_mask, return_mim=True)
    
    # Compute MIM + entropy loss
    mim_loss = mim_reconstruction_loss(
        outputs["mim"]["recon_feat"],
        outputs["mim"]["teacher_feat"],
        mask=outputs["mim"]["mask"]
    )
    ent_loss = outputs["mim"]["entropy"]
    loss = mim_loss + 0.1 * ent_loss
    
    # Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Now use adapted model for prediction
    model.eval()
    with torch.no_grad():
        pred = model(test_image, return_mim=False)
    model.train()
```

## Geometric Consistency

The MIM branch enforces geometric consistency through:

1. **Teacher Features**: Derived from actual geometry (mask + depth), providing ground truth structure
2. **Entropy Minimization**: Lower entropy → more confident predictions that align with geometry
3. **Masked Reconstruction**: Forces the model to understand geometric structure by reconstructing masked regions

### Creating Geometry Input

From ground truth (training):

```python
import torch

def create_geom_input(mask, depth_obj):
    """Create geometry input from mask and object depth.
    
    Args:
        mask: (H, W) binary mask (0 or 1)
        depth_obj: (H, W) object-only depth in meters
        
    Returns:
        geom_input: (2, H, W) concatenated mask and normalized depth
    """
    # Reference depth (median of object pixels)
    valid_depth = depth_obj[mask > 0]
    if len(valid_depth) == 0:
        z_ref = torch.tensor(1.0)
    else:
        z_ref = valid_depth.median()
    
    # Normalize depth: log(D / z_ref)
    depth_norm = mask * (torch.log(depth_obj + 1e-6) - torch.log(z_ref + 1e-6))
    
    # Concatenate
    geom_input = torch.stack([mask.float(), depth_norm], dim=0)
    return geom_input
```

From predictions (inference):

```python
def create_geom_from_predictions(bbox, mask_pred, depth_pred, K):
    """Create geometry input from model predictions.
    
    Args:
        bbox: (4,) bounding box [cx, cy, w, h]
        mask_pred: (H, W) predicted mask (soft or binary)
        depth_pred: scalar or (H, W) predicted depth
        K: (4,) intrinsics [fx, fy, cx, cy]
        
    Returns:
        geom_input: (2, H, W)
    """
    # Binarize mask
    mask = (mask_pred > 0.5).float()
    
    # Create depth map from prediction
    if isinstance(depth_pred, (int, float)):
        depth_map = torch.full_like(mask, depth_pred)
    else:
        depth_map = depth_pred
    
    # Normalize
    geom_input = create_geom_input(mask, depth_map)
    return geom_input
```

## Performance Considerations

- **Train-only by default**: Set `enable_mim=False` for inference-only deployment to save compute
- **Feature level**: MIM operates on P5 neck features (typically 1/32 or 1/16 of input size)
- **Mask ratio**: Use 0.4-0.7 mask ratio for good reconstruction signal
- **TTT steps**: 1-5 steps typically sufficient for test-time adaptation

## References

- Specification: `docs/specs/rt_detr_6dof_geom_mim_spec_en_v0_4.md` Section 6
- Model implementation: `rtdetr_pose/rtdetr_pose/model.py`
- Loss functions: `rtdetr_pose/rtdetr_pose/losses.py`
- Tests: `tests/test_mim_reconstruction.py`
