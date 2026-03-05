# Permanent Ultralytics Triton GPU Fix

This directory contains everything needed to create and distribute a permanently patched version of Ultralytics that fixes the TorchScript GPU export issue for Triton Inference Server.

## Quick Start for Clients

### Option A: Install Pre-built Wheel (Recommended)

If you have the pre-built wheel package:

```bash
pip install ultralytics-8.3.248+triton_gpu_fix-py3-none-any.whl
```

### Option B: Auto-patch Existing Installation

Run this one-liner to patch your current Ultralytics installation:

```bash
curl -sSL https://your-internal-server/ultralytics-patch.py | python3
```

Or manually:

```bash
python3 apply_patch.py
```

---

## For DevOps/Admins: Creating the Patched Package

### 1. Build the Patched Wheel

```bash
cd /home/richa/abhishek/ultralytics_triton_patch
chmod +x setup_patched_ultralytics.sh
./setup_patched_ultralytics.sh
```

This will:
- Clone Ultralytics v8.3.248
- Apply the Triton GPU fix
- Build a wheel package
- Output: `ultralytics/dist/ultralytics-8.3.248+triton_gpu_fix-*.whl`

### 2. Distribute to Clients

**Option 1: Internal PyPI Server (Best for Teams)**

```bash
# Upload to your internal PyPI
twine upload --repository-url https://pypi.your-company.com ultralytics/dist/*.whl

# Clients install with:
pip install ultralytics==8.3.248+triton-gpu-fix --index-url https://pypi.your-company.com
```

**Option 2: Shared Network Drive**

```bash
# Copy wheel to shared location
cp ultralytics/dist/*.whl /mnt/shared/python-packages/

# Clients install with:
pip install /mnt/shared/python-packages/ultralytics-8.3.248+triton_gpu_fix-*.whl
```

**Option 3: Git Repository**

```bash
# Push patched version to internal Git
cd ultralytics
git remote add internal git@gitlab.your-company.com:ml/ultralytics-patched.git
git add -A
git commit -m "Add Triton GPU fix for TorchScript export"
git tag v8.3.248-triton-gpu-fix
git push internal v8.3.248-triton-gpu-fix

# Clients install with:
pip install git+https://gitlab.your-company.com/ml/ultralytics-patched.git@v8.3.248-triton-gpu-fix
```

---

## Docker Integration

### Add to Dockerfile

```dockerfile
# Install patched Ultralytics
COPY ultralytics-8.3.248+triton_gpu_fix-*.whl /tmp/
RUN pip install /tmp/ultralytics-8.3.248+triton_gpu_fix-*.whl && \
    rm /tmp/ultralytics-*.whl
```

Or with requirements.txt:

```txt
# requirements.txt
ultralytics @ file:///path/to/ultralytics-8.3.248+triton_gpu_fix-py3-none-any.whl
```

---

## Verification

After installation, verify the patch is applied:

```python
import ultralytics
print(ultralytics.__version__)  # Should show: 8.3.248+triton-gpu-fix

# Check the patch
import inspect
from ultralytics.nn.modules.transformer import AIFI

sig = inspect.signature(AIFI.build_2d_sincos_position_embedding)
assert 'device' in sig.parameters, "Patch not applied!"
print("✅ Triton GPU fix is active")
```

---

## What This Fix Does

### Problem
When exporting RT-DETR models to TorchScript on CPU, positional embedding tensors are baked into the graph as CPU tensors. Running these models on GPU in Triton causes:

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

### Solution
The patch adds a `device` parameter to `build_2d_sincos_position_embedding()` so tensors are created on the same device as the input, ensuring GPU-native TorchScript graphs.

### Files Modified
- `ultralytics/nn/modules/transformer.py`
  - `AIFI.build_2d_sincos_position_embedding()` - Added device parameter
  - `AIFI.forward()` - Pass device from input tensor

---

## Maintenance

### Upgrading Ultralytics

When a new Ultralytics version is released:

1. Update version in `setup_patched_ultralytics.sh`:
   ```bash
   git checkout v8.4.0  # New version
   ```

2. Re-run the setup script:
   ```bash
   ./setup_patched_ultralytics.sh
   ```

3. Test the export:
   ```bash
   yolo export model=rtdetr-l.pt format=torchscript device=0
   ```

4. Distribute new wheel to clients

### Monitoring for Upstream Fix

Check Ultralytics releases for this fix being merged upstream:
- Issue: https://github.com/ultralytics/ultralytics/issues/XXXXX
- If merged, you can switch back to official releases

---

## Troubleshooting

### Patch fails to apply

**Cause:** Ultralytics code structure changed in newer version

**Solution:**
1. Manually inspect `ultralytics/nn/modules/transformer.py`
2. Update `apply_patch.py` with new code patterns
3. Test thoroughly before distributing

### Clients still getting device mismatch

**Cause:** Using wrong Ultralytics version or patch not installed

**Solution:**
```bash
# Check installed version
pip show ultralytics

# Reinstall patched version
pip uninstall ultralytics
pip install ultralytics-8.3.248+triton_gpu_fix-*.whl --force-reinstall
```

---

## Support

For issues with this patch:
1. Check `ultralytics/dist/build.log` for build errors
2. Verify Python version compatibility (3.8+)
3. Contact: richa@your-company.com / abhishek@your-company.com

---

## License

This patch maintains Ultralytics' AGPL-3.0 license.
The patched package is for internal use only.

---

**Last Updated:** 2026-02-26  
**Patch Version:** 1.0  
**Compatible Ultralytics:** 8.3.248  
**Tested With:** Triton 23.10, PyTorch 2.8.0, CUDA 12.8
