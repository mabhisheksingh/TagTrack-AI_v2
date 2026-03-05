#!/usr/bin/env python3
"""
Apply Triton GPU fix to Ultralytics transformer.py
This ensures positional embedding tensors are created on the same device as input.
"""
import sys
import os

def apply_patch(transformer_path):
    """Apply the device parameter patch to transformer.py"""
    
    if not os.path.exists(transformer_path):
        print(f"❌ File not found: {transformer_path}")
        return False
    
    with open(transformer_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Patch 1: Update method signature to accept device parameter
    old_signature = '''    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0
    ) -> torch.Tensor:
        """Build 2D sine-cosine position embedding.

        Args:
            w (int): Width of the feature map.
            h (int): Height of the feature map.
            embed_dim (int): Embedding dimension.
            temperature (float): Temperature for the sine/cosine functions.

        Returns:
            (torch.Tensor): Position embedding with shape [1, embed_dim, h*w].
        """'''
    
    new_signature = '''    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0, device: torch.device = None
    ) -> torch.Tensor:
        """Build 2D sine-cosine position embedding.

        Args:
            w (int): Width of the feature map.
            h (int): Height of the feature map.
            embed_dim (int): Embedding dimension.
            temperature (float): Temperature for the sine/cosine functions.
            device (torch.device): Device to create tensors on (cuda/cpu).

        Returns:
            (torch.Tensor): Position embedding with shape [1, embed_dim, h*w].
        """'''
    
    if old_signature in content:
        content = content.replace(old_signature, new_signature)
        print("✓ Updated method signature with device parameter")
    else:
        print("⚠ Method signature not found or already patched")
    
    # Patch 2: Add device parameter to torch.arange calls
    replacements = [
        ('grid_w = torch.arange(w, dtype=torch.float32)',
         'grid_w = torch.arange(w, dtype=torch.float32, device=device)'),
        ('grid_h = torch.arange(h, dtype=torch.float32)',
         'grid_h = torch.arange(h, dtype=torch.float32, device=device)'),
        ('omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim',
         'omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim'),
    ]
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
    
    print("✓ Updated tensor creation to use device parameter")
    
    # Patch 3: Update forward method to pass device
    old_forward = '''        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))'''
    
    new_forward = '''        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c, device=x.device)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(dtype=x.dtype))'''
    
    if old_forward in content:
        content = content.replace(old_forward, new_forward)
        print("✓ Updated forward method to pass device parameter")
    else:
        print("⚠ Forward method not found or already patched")
    
    # Check if any changes were made
    if content == original_content:
        print("⚠ No changes applied - file may already be patched")
        return True
    
    # Backup original file
    backup_path = transformer_path + '.original'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            f.write(original_content)
        print(f"✓ Created backup: {backup_path}")
    
    # Write patched content
    with open(transformer_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Successfully patched {transformer_path}")
    return True

if __name__ == '__main__':
    # Determine transformer.py path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if running in cloned repo context
    repo_path = os.path.join(script_dir, 'ultralytics', 'ultralytics', 'nn', 'modules', 'transformer.py')
    if os.path.exists(repo_path):
        transformer_path = repo_path
    else:
        # Fall back to installed package
        import site
        site_packages = site.getsitepackages()[0]
        transformer_path = os.path.join(site_packages, 'ultralytics', 'nn', 'modules', 'transformer.py')
    
    print(f"Patching: {transformer_path}")
    success = apply_patch(transformer_path)
    sys.exit(0 if success else 1)
