# Triton RT-DETR TorchScript Export Fix

## Problem Summary

When exporting Ultralytics RT-DETR models to TorchScript format for Triton Inference Server deployment, the models fail at runtime with:

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**Root Cause:** Ultralytics' `build_2d_sincos_position_embedding` method in the transformer module creates positional embedding tensors on CPU by default. During TorchScript tracing, these CPU tensors get baked into the graph, causing device mismatch errors when Triton executes the model on GPU.

---

## Permanent Fix

### Option 1: Patch Ultralytics Installation (Recommended)

Apply this patch to your Ultralytics installation before exporting models:

**File to patch:** `ultralytics/nn/modules/transformer.py`

**Location:** `/path/to/python/site-packages/ultralytics/nn/modules/transformer.py`

Example: `/home/richa/.pyenv/versions/3.10.12/lib/python3.10/site-packages/ultralytics/nn/modules/transformer.py`

#### Automated Patch Script

Save and run this script to apply the fix:

```python
#!/usr/bin/env python3
"""
Patch Ultralytics transformer.py to fix TorchScript GPU export for Triton.
This ensures positional embedding tensors are created on the same device as input.
"""
import sys
import os

# Adjust this path to match your Python environment
TRANSFORMER_PATH = "/home/richa/.pyenv/versions/3.10.12/lib/python3.10/site-packages/ultralytics/nn/modules/transformer.py"

def apply_patch():
    if not os.path.exists(TRANSFORMER_PATH):
        print(f"❌ File not found: {TRANSFORMER_PATH}")
        print("Please update TRANSFORMER_PATH to match your environment")
        sys.exit(1)
    
    with open(TRANSFORMER_PATH, 'r') as f:
        content = f.read()
    
    # Patch 1: Update method signature to accept device parameter
    old_signature = '''    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0
    ) -> torch.Tensor:'''
    
    new_signature = '''    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0, device: torch.device = None
    ) -> torch.Tensor:'''
    
    if old_signature in content:
        content = content.replace(old_signature, new_signature)
        print("✓ Updated method signature")
    else:
        print("⚠ Method signature already patched or different version")
    
    # Patch 2: Add device parameter to torch.arange calls
    old_arange_w = 'grid_w = torch.arange(w, dtype=torch.float32)'
    new_arange_w = 'grid_w = torch.arange(w, dtype=torch.float32, device=device)'
    
    old_arange_h = 'grid_h = torch.arange(h, dtype=torch.float32)'
    new_arange_h = 'grid_h = torch.arange(h, dtype=torch.float32, device=device)'
    
    old_arange_omega = 'omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim'
    new_arange_omega = 'omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim'
    
    content = content.replace(old_arange_w, new_arange_w)
    content = content.replace(old_arange_h, new_arange_h)
    content = content.replace(old_arange_omega, new_arange_omega)
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
        print("⚠ Forward method already patched or different version")
    
    # Backup original file
    backup_path = TRANSFORMER_PATH + '.backup'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            with open(TRANSFORMER_PATH, 'r') as orig:
                f.write(orig.read())
        print(f"✓ Created backup: {backup_path}")
    
    # Write patched content
    with open(TRANSFORMER_PATH, 'w') as f:
        f.write(content)
    
    print(f"✅ Successfully patched {TRANSFORMER_PATH}")
    print("\nYou can now export RT-DETR models to TorchScript with GPU support!")

if __name__ == '__main__':
    apply_patch()
```

**To apply:**
```bash
# Update TRANSFORMER_PATH in the script to match your environment
python3 patch_ultralytics_transformer.py
```

---

## Model Export Process

After applying the patch, export your RT-DETR models:

### 1. Plate Region Detection Model

```bash
cd /home/richa/abhishek/model_repository/plate_region_detection_rt_detr/1

# Export with GPU device to ensure all tensors are on CUDA
CUDA_VISIBLE_DEVICES=0 \
yolo export \
  model=plate_region_detection-RT-DETR_weights_best.pt \
  format=torchscript \
  device=0 \
  half=False

# Replace the old model
rm -f model.pt
mv plate_region_detection-RT-DETR_weights_best.torchscript model.pt
```

### 2. Vehicle Detection Model

```bash
cd /home/richa/abhishek/model_repository/vehicle_detection_rt_detr/1

# Export with GPU device
CUDA_VISIBLE_DEVICES=0 \
yolo export \
  model=vehicle_detection_RT-DETR_best.pt \
  format=torchscript \
  device=0 \
  half=False

# Replace the old model
rm -f model.pt
mv vehicle_detection_RT-DETR_best.torchscript model.pt
```

---

## Triton Deployment

### Model Repository Structure

```
model_repository/
├── plate_region_detection_rt_detr/
│   ├── config.pbtxt
│   └── 1/
│       └── model.pt  (TorchScript exported with patch)
└── vehicle_detection_rt_detr/
    ├── config.pbtxt
    └── 1/
        └── model.pt  (TorchScript exported with patch)
```

### Start Triton Server

```bash
docker run --rm --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models --model-control-mode=poll
```

### Verify Models Loaded

Check Triton logs for:
```
I0226 06:23:05.048231 1 server.cc:662] 
+--------------------------------+---------+--------+
| Model                          | Version | Status |
+--------------------------------+---------+--------+
| plate_region_detection_rt_detr | 1       | READY  |
| vehicle_detection_rt_detr      | 1       | READY  |
+--------------------------------+---------+--------+
```

---

## Testing Inference

### Python Client Example

```python
import numpy as np
import cv2
import tritonclient.http as httpclient

# Connect to Triton
client = httpclient.InferenceServerClient(url="localhost:8000")

# Load and preprocess image
img = cv2.imread("test_image.jpg")
img = cv2.resize(img, (640, 640))
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))  # HWC to CHW
img = np.expand_dims(img, axis=0)   # Add batch dimension

# Create input
infer_input = httpclient.InferInput("images", img.shape, "FP32")
infer_input.set_data_from_numpy(img)

# Run inference
results = client.infer(
    model_name="plate_region_detection_rt_detr",
    inputs=[infer_input]
)

print(results.get_response())
```

---

## Troubleshooting

### Issue: Still getting CPU/GPU device mismatch

**Solution:**
1. Verify the patch was applied correctly:
   ```bash
   grep "device: torch.device = None" /path/to/ultralytics/nn/modules/transformer.py
   ```
2. Delete old TorchScript exports and re-export
3. Restart Triton server to reload models

### Issue: Model not found in Triton

**Solution:**
1. Check file is named exactly `model.pt` in version folder
2. Verify config.pbtxt points to correct backend (pytorch)
3. Check Triton logs for loading errors

### Issue: Ultralytics version mismatch

**Solution:**
- This patch was tested with Ultralytics 8.3.248
- For other versions, manually inspect `transformer.py` and adjust patch accordingly
- Key changes needed:
  - Add `device` parameter to `build_2d_sincos_position_embedding`
  - Pass `device=device` to all `torch.arange` calls
  - Call method with `device=x.device` in forward pass

---

## Version Information

- **Ultralytics:** 8.3.248
- **PyTorch:** 2.8.0+cu128
- **Triton Server:** 23.10-py3 (2.39.0)
- **CUDA:** 12.8
- **Python:** 3.10.12

---

## Permanent Fix Status

✅ **Patch Applied:** `/home/richa/.pyenv/versions/3.10.12/lib/python3.10/site-packages/ultralytics/nn/modules/transformer.py`

✅ **Models Exported:**
- `plate_region_detection_rt_detr/1/model.pt` (124 MB)
- `vehicle_detection_rt_detr/1/model.pt` (124 MB)

⚠️ **Note:** If you upgrade Ultralytics or switch Python environments, you'll need to re-apply this patch.

---

## Alternative: Use Patched Ultralytics Fork

For production deployments, consider:

1. Fork Ultralytics repository
2. Apply this patch to your fork
3. Install from your fork:
   ```bash
   pip install git+https://github.com/YOUR_ORG/ultralytics.git@patched-transformer
   ```

This ensures the fix persists across environments and deployments.

---

## References

- **Issue:** TorchScript trace captures CPU tensors in positional embeddings
- **Fix:** Pass device parameter through embedding generation chain
- **Files Modified:** `ultralytics/nn/modules/transformer.py`
- **Methods Changed:** `build_2d_sincos_position_embedding`, `AIFI.forward`

---

**Last Updated:** 2026-02-26  
**Maintained By:** Richa/Abhishek Team
