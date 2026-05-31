#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE="${RECORD_DIR:-$REPO_ROOT/record}"
OUTDIR="${OUTPUT_DIR:-$SCRIPT_DIR/quantization/outputs/torch_ao}"
QUANT_SCRIPT="$SCRIPT_DIR/torch_ao_quantization_eval.py"

MODELS=(
  "cifar10_preactresnet18_badnet_0_01"
  "cifar10_preactresnet18_badnet_0_1"
  "cifar10_preactresnet18_blended_0_01"
  "cifar10_preactresnet18_blended_0_1"
  "cifar10_preactresnet18_bpp_0_01"
  "cifar10_preactresnet18_bpp_0_1"
  "cifar10_preactresnet18_inputaware_0_01"
  "cifar10_preactresnet18_inputaware_0_1"
  "cifar10_preactresnet18_lf_0_01"
  "cifar10_preactresnet18_lf_0_1"
  "cifar10_preactresnet18_sig_0_01"
  "cifar10_preactresnet18_sig_0_1"
  "cifar10_preactresnet18_ssba_0_01"
  "cifar10_preactresnet18_ssba_0_1"
  "cifar10_preactresnet18_wanet_0_01"
  "cifar10_preactresnet18_wanet_0_1"
)

OPTIONS=("dynamic" "static" "float16")
mkdir -p "$OUTDIR"

for mdir in "${MODELS[@]}"; do
  model_path="$BASE/$mdir/attack_result.pt"
  if [ ! -f "$model_path" ]; then
    echo "Warning: $model_path not found, skipping." >&2
    continue
  fi
  for opt in "${OPTIONS[@]}"; do
    out_file="$OUTDIR/${mdir}_${opt}.txt"
    echo "=== MODEL=${mdir} OPTION=${opt} ===" | tee -a "$out_file"
    "$PYTHON_BIN" "$QUANT_SCRIPT" --modelpath "$mdir" --option "$opt" --output-dir "$OUTDIR" >> "$out_file" 2>&1
  done
done
