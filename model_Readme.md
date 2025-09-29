# CLIO Model Downloader

This script pre-downloads the machine learning models required for semantic inference in CLIO, preventing download delays during first use.

## What it downloads

- **FastSAM-x.pt**: Fast Segment Anything Model for object segmentation (~139MB)
- **CLIP ViT-L/14**: Large CLIP model for text-image matching (~890MB)
- **CLIP ViT-B/32**: Smaller CLIP model for faster inference (~215MB)

## Usage

```bash
cd /catkin_ws/src/Manibot_scene_graph
./download_models.sh
```

## Requirements

- `curl` (for downloading)
- Internet connection
- Sufficient disk space (~1.2GB total)

## What happens

The script will:
1. Create necessary cache directories in `~/.cache/`
2. Download missing models from official sources
3. Verify downloads completed successfully
4. Skip models that already exist

## Verification

After running the script, you can verify the models work:

```bash
# Test FastSAM
python -c "from semantic_inference.models.wrappers import FastSAMSegmentation; FastSAMSegmentation.construct()"

# Test CLIP
python -c "from semantic_inference.models.wrappers import ClipWrapper; ClipWrapper.construct()"
```

## Troubleshooting

- If downloads fail, check your internet connection
- The script will automatically retry failed downloads
- Models are cached permanently, so you only need to download once
- If you encounter permission issues, ensure you have write access to `~/.cache/`