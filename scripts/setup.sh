#!/usr/bin/env bash
# Install everything needed to evaluate MoFlo. Run from the release root inside a
# Python 3.10 environment. Safe to re-run.
#   PIP="uv pip"             use uv instead of pip
#   ROBOCASA_ASSETS=/path    link an existing robocasa/models/assets (skips ~8 GB download)
#   TORCH_INDEX=...          PyTorch wheel index (default cu128)
set -euo pipefail
cd "$(dirname "$0")/.."
PIP="${PIP:-python -m pip}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
mkdir -p external

fetch() {  # name url commit
  [ -d "external/$1/.git" ] || git clone -q "$2" "external/$1"
  git -C "external/$1" checkout -q "$3"
}
fetch robocasa       https://github.com/yjy0625/mobipi-robocasa.git  3683fb01d6fc87d7849e7b5a886d01f4b1d7a55d
fetch robomimic      https://github.com/yjy0625/mobipi-robomimic.git ff9f7f4157a5c8257f17c5910067030e2291378f
fetch mimicgen       https://github.com/yjy0625/mobipi-mimicgen.git  05c723bdf9b4e1ddb77fea63bdde21920408b5fd
fetch OpenShape_code https://github.com/Colin97/OpenShape_code.git   abe5aa42b7c99c037c286ad54313f695a114bf0d

$PIP install torch==2.7.1 torchvision==0.22.1 --index-url "$TORCH_INDEX"
$PIP install -r requirements.txt
# The forks pin mutually incompatible torch/numpy/diffusers versions, so install
# them without dependencies; requirements.txt provides a working set.
$PIP install --no-deps robosuite==1.5.0
$PIP install --no-deps -e external/robocasa -e external/robomimic -e external/mimicgen
$PIP install --no-deps -e .

[ -f external/robocasa/robocasa/macros_private.py ] || python external/robocasa/robocasa/scripts/setup_macros.py

ASSETS=external/robocasa/robocasa/models/assets
# The clone ships only XML stubs for fixtures; meshes and textures are separate.
if [ ! -e "$ASSETS/generative_textures" ]; then
  if [ -n "${ROBOCASA_ASSETS:-}" ]; then
    for d in textures fixtures objects generative_textures; do
      if [ -e "$ASSETS/$d" ]; then mv "$ASSETS/$d" "$ASSETS/$d.stub"; fi
      ln -s "$ROBOCASA_ASSETS/$d" "$ASSETS/$d"
    done
  else
    echo y | python external/robocasa/robocasa/scripts/download_kitchen_assets.py
  fi
fi

python scripts/download_policies.py
echo "Setup done. Next: bash scripts/eval.sh poses"
