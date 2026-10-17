#!/usr/bin/env bash
# Runs on a rented CUDA machine: set up NCM, time eager vs KV-cached generation, run the full
# WMT14/XSum evals with the cache on, and leave everything in $OUT (DONE or FAILED marks the end).
set -uo pipefail
REPO=${REPO:-https://github.com/velaia/ncm.git}
BRANCH=${BRANCH:-mac-mps}
WORK=${WORK:-/workspace}
OUT=$WORK/results
mkdir -p "$OUT"
exec >>"$OUT/log.txt" 2>&1
trap 'echo "FAILED at line $LINENO"; touch "$OUT/FAILED"' ERR
set -e
echo "=== start $(date -u)"

command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; }
cd "$WORK"
[ -d ncm ] || git clone -q -b "$BRANCH" "$REPO" ncm
cd ncm
git rev-parse HEAD >"$OUT/commit.txt"
nvidia-smi >"$OUT/nvidia-smi.txt"
pip install -q -r requirements.txt  # torch is already in the image and unpinned here, so it stays
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())" | tee "$OUT/torch.txt"

hf download hazyresearch/ncm-checkpoints --include "wmt14/*" --include "xsum/*" --include "sudoku/*" --local-dir artifacts >/dev/null
hf download hazyresearch/ncm-datasets --repo-type dataset --include "wmt14/*" --include "xsum/*" --include "sudoku/*" --local-dir data >/dev/null
# tokenizers + the GPT-2 used for XSum genPPL; then stay offline so a hung Hub request can't stall a run
hf download Qwen/Qwen3-0.6B --exclude "*.safetensors" >/dev/null
hf download gpt2 --include "*.json" --include "*.txt" --include "model.safetensors" >/dev/null
export HF_HUB_OFFLINE=1

# speed: paper protocol (1024 inputs, batch 16), with and without guidance, plus batch 64
B="python scripts/bench_kv.py"
$B --config config/eval/wmt14.yaml --json "$OUT/bench-wmt14-bs16.json"
$B --config config/eval/wmt14.yaml --cfg 1.0 --json "$OUT/bench-wmt14-cfg1-bs16.json"
$B --config config/eval/xsum.yaml --json "$OUT/bench-xsum-bs16.json"
$B --config config/eval/xsum.yaml --cfg 0.5 --json "$OUT/bench-xsum-cfg05-bs16.json"
$B --config config/eval/wmt14.yaml --bs 64 --json "$OUT/bench-wmt14-bs64.json"
$B --config config/eval/xsum.yaml --bs 64 --json "$OUT/bench-xsum-bs64.json"
for d in easy medium hard; do  # same setup as the Mac run: full 2000-puzzle test set, batch 128
    $B --config config/eval/sudoku-$d.yaml --checkpoint-dir artifacts/sudoku/$d --data-dir data/sudoku/$d --n 2000 --bs 128 --json "$OUT/bench-sudoku-$d.json"
done

# quality: full test sets with the cache on, to compare against the paper
for t in wmt14 xsum; do
    s=$SECONDS
    python evaluate.py --config config/eval/$t.yaml --checkpoint-dir artifacts/$t --data-dir data/$t \
        --kv-cache --out eval/$t-kv 2>&1 | grep -v -i warn | tee "$OUT/eval-$t-kv.txt"
    echo "wall: $((SECONDS - s))s" | tee -a "$OUT/eval-$t-kv.txt"
done
python evaluate.py --config config/eval/wmt14.yaml --checkpoint-dir artifacts/wmt14 --data-dir data/wmt14 \
    --kv-cache --cfg 1.0 --out eval/wmt14-kv-cfg1 2>&1 | grep -v -i warn | tee "$OUT/eval-wmt14-kv-cfg1.txt"

echo "=== done $(date -u)"
touch "$OUT/DONE"
