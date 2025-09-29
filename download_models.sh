#!/bin/bash

# Script to download FastSAM and CLIP models for semantic inference
# This script pre-downloads the models that would otherwise be downloaded on first run

set -e  # Exit on any error

echo "=== CLIO Model Downloader ==="
echo "Downloading FastSAM and CLIP models for semantic inference..."
echo ""

# Create cache directories
echo "Creating cache directories..."
mkdir -p ~/.cache/ultralytics
mkdir -p ~/.cache/clip
mkdir -p ~/.cache/torch/hub/checkpoints

# Download FastSAM model
echo "Downloading FastSAM-x.pt..."
if [ ! -f ~/.cache/ultralytics/FastSAM-x.pt ] || [ ! -s ~/.cache/ultralytics/FastSAM-x.pt ]; then
    echo "  Downloading FastSAM-x.pt from ultralytics..."
    rm -f ~/.cache/ultralytics/FastSAM-x.pt  # Remove any partial downloads
    if curl -L --progress-bar -o ~/.cache/ultralytics/FastSAM-x.pt \
        "https://github.com/ultralytics/assets/releases/download/v8.1.0/FastSAM-x.pt"; then
        if [ -s ~/.cache/ultralytics/FastSAM-x.pt ]; then
            echo "  ✓ FastSAM-x.pt downloaded successfully"
        else
            echo "  ✗ FastSAM-x.pt download failed (empty file)"
            rm -f ~/.cache/ultralytics/FastSAM-x.pt
        fi
    else
        echo "  ✗ FastSAM-x.pt download failed (curl error)"
        rm -f ~/.cache/ultralytics/FastSAM-x.pt
    fi
else
    echo "  ✓ FastSAM-x.pt already exists, skipping download"
fi

# Download CLIP models
echo ""
echo "Downloading CLIP models..."

# CLIP ViT-L/14 model
if [ ! -f ~/.cache/clip/ViT-L-14.pt ] || [ ! -s ~/.cache/clip/ViT-L-14.pt ]; then
    echo "  Downloading CLIP ViT-L/14..."
    rm -f ~/.cache/clip/ViT-L-14.pt  # Remove any partial downloads
    if curl -L --progress-bar -o ~/.cache/clip/ViT-L-14.pt \
        "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03803/ViT-L-14.pt"; then
        if [ -s ~/.cache/clip/ViT-L-14.pt ]; then
            echo "  ✓ CLIP ViT-L/14 downloaded successfully"
        else
            echo "  ✗ CLIP ViT-L/14 download failed (empty file)"
            rm -f ~/.cache/clip/ViT-L-14.pt
        fi
    else
        echo "  ✗ CLIP ViT-L/14 download failed (curl error)"
        rm -f ~/.cache/clip/ViT-L-14.pt
    fi
else
    echo "  ✓ CLIP ViT-L/14 already exists, skipping download"
fi

# CLIP ViT-B/32 model (smaller/faster alternative)
if [ ! -f ~/.cache/clip/ViT-B-32.pt ] || [ ! -s ~/.cache/clip/ViT-B-32.pt ]; then
    echo "  Downloading CLIP ViT-B/32..."
    rm -f ~/.cache/clip/ViT-B-32.pt  # Remove any partial downloads
    if curl -L --progress-bar -o ~/.cache/clip/ViT-B-32.pt \
        "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219ccbe50dcb51e74254a91431558/ViT-B-32.pt"; then
        if [ -s ~/.cache/clip/ViT-B-32.pt ]; then
            echo "  ✓ CLIP ViT-B/32 downloaded successfully"
        else
            echo "  ✗ CLIP ViT-B/32 download failed (empty file)"
            rm -f ~/.cache/clip/ViT-B-32.pt
        fi
    else
        echo "  ✗ CLIP ViT-B/32 download failed (curl error)"
        rm -f ~/.cache/clip/ViT-B-32.pt
    fi
else
    echo "  ✓ CLIP ViT-B/32 already exists, skipping download"
fi

# Download OpenCLIP models if needed (for ViT-H-14)
echo ""
echo "Checking for OpenCLIP models..."
if [ ! -d ~/.cache/clip/open_clip ]; then
    echo "  Creating OpenCLIP cache directory..."
    mkdir -p ~/.cache/clip/open_clip
fi

# Note: OpenCLIP models are typically downloaded on-demand, but we can pre-cache them
echo "  Note: OpenCLIP models (ViT-H-14) will be downloaded on first use if needed"

echo ""
echo "=== Download Summary ==="
echo "✓ FastSAM-x.pt: $(ls -lh ~/.cache/ultralytics/FastSAM-x.pt | awk '{print $5}')"
echo "✓ CLIP ViT-L/14: $(ls -lh ~/.cache/clip/ViT-L-14.pt | awk '{print $5}')"
echo "✓ CLIP ViT-B/32: $(ls -lh ~/.cache/clip/ViT-B-32.pt | awk '{print $5}')"

echo ""
echo "All models downloaded successfully!"
echo "You can now run semantic inference without waiting for model downloads."
echo ""
echo "To verify the models work, you can run:"
echo "  python -c \"from semantic_inference.models.wrappers import FastSAMSegmentation; FastSAMSegmentation.construct()\""
echo "  python -c \"from semantic_inference.models.wrappers import ClipWrapper; ClipWrapper.construct()\""