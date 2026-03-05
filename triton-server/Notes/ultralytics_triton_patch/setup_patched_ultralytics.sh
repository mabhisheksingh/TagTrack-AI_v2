#!/bin/bash
# Automated setup for patched Ultralytics with Triton GPU fix
# This script creates a permanent patched version that can be distributed to clients

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${SCRIPT_DIR}"

echo "=== Ultralytics Triton GPU Fix - Permanent Setup ==="
echo ""

# Step 1: Clone Ultralytics if not already present
if [ ! -d "${PATCH_DIR}/ultralytics" ]; then
    echo "📦 Cloning Ultralytics repository..."
    git clone https://github.com/ultralytics/ultralytics.git "${PATCH_DIR}/ultralytics"
    cd "${PATCH_DIR}/ultralytics"
    git checkout v8.3.248  # Pin to tested version
else
    echo "✓ Ultralytics repository already exists"
    cd "${PATCH_DIR}/ultralytics"
fi

# Step 2: Apply the patch
echo ""
echo "🔧 Applying Triton GPU fix patch..."
python3 "${PATCH_DIR}/apply_patch.py"

# Step 3: Create a custom version tag
echo ""
echo "🏷️  Creating custom version tag..."
sed -i "s/__version__ = .*/__version__ = '8.3.248+triton-gpu-fix'/" ultralytics/__init__.py

# Step 4: Build wheel package
echo ""
echo "📦 Building wheel package..."
pip install build
python3 -m build

echo ""
echo "✅ Patched Ultralytics package created!"
echo ""
echo "📍 Wheel location: ${PATCH_DIR}/ultralytics/dist/"
ls -lh "${PATCH_DIR}/ultralytics/dist/"*.whl
echo ""
echo "To install on any machine:"
echo "  pip install ${PATCH_DIR}/ultralytics/dist/ultralytics-8.3.248+triton_gpu_fix-*.whl"
echo ""
echo "Or upload to your internal PyPI server for team-wide distribution"
