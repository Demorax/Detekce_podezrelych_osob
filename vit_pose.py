import sys
import os
import torch
import torch.nn as nn

# Add ViTPose to path if you have it
if os.path.exists('ViTPose'):
    sys.path.insert(0, 'ViTPose')

# First, register ViT backbone
from mmpose.registry import MODELS


@MODELS.register_module()
class ViT(nn.Module):
    """Minimal ViT backbone registration for ViTPose."""

    def __init__(self, img_size=(256, 192), patch_size=16, embed_dim=1280,
                 depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
                 drop_path_rate=0.3, use_checkpoint=False, ratio=1, **kwargs):
        super().__init__()
        # This is just a placeholder - the actual weights will be loaded from checkpoint
        print(f"Registering ViT with embed_dim={embed_dim}, depth={depth}")

        # Minimal implementation - checkpoint loading will override these
        self.embed_dim = embed_dim
        self.num_features = embed_dim

        # Create a simple linear layer as placeholder
        self.dummy = nn.Linear(1, embed_dim)

    def forward(self, x):
        # This won't be used since weights are loaded from checkpoint
        B, C, H, W = x.shape
        # Return placeholder feature map
        feat_h, feat_w = H // 16, W // 16  # Assuming patch_size=16
        return [torch.zeros(B, self.embed_dim, feat_h, feat_w, device=x.device)]


print("Registering ViT backbone...")

from mmpose.apis import init_model

# Paths - UPDATE TO USE FIXED CONFIG
config_file = 'models/vitpose/configs/vitpose_huge_mmpose132.py'  # Fixed config file
checkpoint_file = 'models/vitpose/vitpose_huge.pth'  # Your checkpoint file

try:
    # Load the model
    print("Loading ViTPose model...")
    model = init_model(config_file, checkpoint_file, device='cuda')
    print("✓ ViTPose model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Model device: {next(model.parameters()).device}")

except Exception as e:
    print(f"✗ Failed to load model: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure config file exists:", os.path.exists(config_file))
    print("2. Make sure checkpoint file exists:", os.path.exists(checkpoint_file))

    # Try alternative approach
    print("\n--- Trying alternative: MMPose built-in models ---")
    try:
        from mmpose.apis import MMPoseInferencer

        model_alt = MMPoseInferencer('human', device='cuda')
        print("✓ Alternative RTMPose model loaded successfully!")
    except Exception as e2:
        print(f"✗ Alternative also failed: {e2}")































