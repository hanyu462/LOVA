#!/usr/bin/env bash
# Download COCO 2017 instance segmentation data.
# usage: scripts/download_coco.sh <target_dir> [all|val]
#   all (default): train2017 (~18GB) + val2017 (~1GB) + annotations
#   val          : val2017 + annotations only (for local testing)
set -euo pipefail
DST=${1:?target dir}
WHAT=${2:-all}
mkdir -p "$DST" && cd "$DST"

files=(annotations/annotations_trainval2017.zip zips/val2017.zip)
[ "$WHAT" = "all" ] && files+=(zips/train2017.zip)

for f in "${files[@]}"; do
  name=$(basename "$f")
  dir=${name%.zip}; [ "$dir" = "annotations_trainval2017" ] && dir=annotations
  if [ -d "$dir" ]; then echo "skip $dir (exists)"; continue; fi
  curl -L --fail -C - -o "$name" "http://images.cocodataset.org/$f"
  unzip -q "$name" && rm "$name"
done
ls "$DST"
