#!/bin/bash
mkdir -p data
if [ ! -d "data/nerf_synthetic" ]; then
     bash ./scripts/download_blender.sh
fi
echo "Getting Multicam/Multi-scaled Blender Dataset"
python scripts/convert_blender_data.py --blenderdir data/nerf_synthetic --outdir data/nerf_multiscale
