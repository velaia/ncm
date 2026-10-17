#!/usr/bin/env bash
# Runs on a rented CUDA machine: set up NCM, time eager vs KV-cached generation, run the full
# WMT14/XSum evals with the cache on, and leave everything in $OUT (DONE or FAILED marks the end).
# Works in a PyTorch container (torch preinstalled) and on a bare Ubuntu VM with only the NVIDIA driver
# (torch goes into a uv venv). SWEEP=1 adds the precision / torch.compile / CUDA-graph sweep in $OUT/opt.
set -uo pipefail
REPO=${REPO:-https://github.com/velaia/ncm.git}
BRANCH=${BRANCH:-mac-mps}
TORCH=${TORCH:-torch==2.14.1}  # same build as the 4090 run (cu126)
[ -n "${WORK:-}" ] || { WORK=/workspace; mkdir -p "$WORK" 2>/dev/null && [ -w "$WORK" ] || WORK=$HOME/workspace; }
OUT=$WORK/results
mkdir -p "$OUT"
exec >>"$OUT/log.txt" 2>&1
trap 'echo "FAILED at line $LINENO"; touch "$OUT/FAILED"' ERR
set -eE
rm -f "$OUT/DONE" "$OUT/FAILED"  # safe to re-run after a fix: finished benches/evals are skipped
echo "=== start $(date -u)"

SUDO=$([ "$(id -u)" = 0 ] || echo sudo)
command -v git >/dev/null || { $SUDO apt-get update -qq && $SUDO apt-get install -y -qq git; }
cd "$WORK"
if [ -d ncm ]; then git -C ncm pull -q --ff-only; else git clone -q -b "$BRANCH" "$REPO" ncm; fi
cd ncm
git rev-parse HEAD >"$OUT/commit.txt"
nvidia-smi >"$OUT/nvidia-smi.txt"
if python3 -c "import torch" 2>/dev/null; then
    # pytorch/pytorch images are Ubuntu 24.04 with torch in the system Python (PEP 668); this box is throwaway
    export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
    command -v python >/dev/null || ln -s "$(command -v python3)" /usr/local/bin/python
    pip install -q -r requirements.txt  # torch is already in the image and unpinned here, so it stays
else  # bare VM: torch and everything else into a venv
    command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; export PATH=$HOME/.local/bin:$PATH; }
    [ -x "$WORK/venv/bin/python" ] || uv venv -q --python 3.12 "$WORK/venv"
    . "$WORK/venv/bin/activate"
    uv pip install -q "$TORCH" --index-url https://download.pytorch.org/whl/cu126
    uv pip install -q -r requirements.txt
fi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())" | tee "$OUT/torch.txt"

hf download hazyresearch/ncm-checkpoints --include "wmt14/*" --include "xsum/*" --include "sudoku/*" --local-dir artifacts >/dev/null
hf download hazyresearch/ncm-datasets --repo-type dataset --include "wmt14/*" --include "xsum/*" --include "sudoku/*" --local-dir data >/dev/null
# tokenizers + the GPT-2 used for XSum genPPL; then stay offline so a hung Hub request can't stall a run
hf download Qwen/Qwen3-0.6B --exclude "*.safetensors" >/dev/null
# the legacy alias "gpt2" can't fetch xet-stored weights (404 on xet-read-token): fetch the canonical repo, alias it
hf download openai-community/gpt2 --include "*.json" --include "*.txt" --include "model.safetensors" >/dev/null
C=$(python -c "from huggingface_hub.constants import HF_HUB_CACHE as c; print(c)")
rm -rf "$C/models--gpt2" && ln -s models--openai-community--gpt2 "$C/models--gpt2"
export HF_HUB_OFFLINE=1

# speed: paper protocol (1024 inputs, batch 16), with and without guidance, plus batch 64
bench() { local j="$OUT/bench-$1.json"; shift; [ -s "$j" ] && { echo "skip $j"; return; }; python scripts/bench_kv.py "$@" --json "$j"; }
bench wmt14-bs16 --config config/eval/wmt14.yaml
bench wmt14-cfg1-bs16 --config config/eval/wmt14.yaml --cfg 1.0
bench xsum-bs16 --config config/eval/xsum.yaml
bench xsum-cfg05-bs16 --config config/eval/xsum.yaml --cfg 0.5
bench wmt14-bs64 --config config/eval/wmt14.yaml --bs 64
bench xsum-bs64 --config config/eval/xsum.yaml --bs 64
for d in easy medium hard; do  # same setup as the Mac run: full 2000-puzzle test set, batch 128
    bench sudoku-$d --config config/eval/sudoku-$d.yaml --checkpoint-dir artifacts/sudoku/$d --data-dir data/sudoku/$d --n 2000 --bs 128
done

# quality: full test sets with the cache on, to compare against the paper
full() {  # full() NAME TASK [extra args]: full test-set eval, skipped once its report has a wall time
    local f="$OUT/eval-$1.txt" t=$2 s=$SECONDS; shift 2
    grep -q "^wall:" "$f" 2>/dev/null && { echo "skip $f"; return; }
    python evaluate.py --config config/eval/$t.yaml --checkpoint-dir artifacts/$t --data-dir data/$t \
        --kv-cache "$@" 2>&1 | grep -v -i warn | tee "$f"
    echo "wall: $((SECONDS - s))s" | tee -a "$f"
}
full wmt14-kv wmt14 --out eval/wmt14-kv
full xsum-kv xsum --out eval/xsum-kv
full wmt14-kv-cfg1 wmt14 --cfg 1.0 --out eval/wmt14-kv-cfg1
mkdir -p "$OUT/eval-outputs"; cp -r eval/wmt14-kv eval/xsum-kv eval/wmt14-kv-cfg1 "$OUT/eval-outputs/" 2>/dev/null || true

if [ "${SWEEP:-0}" = 1 ]; then  # same matrix as the 4090 sweep; eager first, the reference for "identical"
    M=eager,kv,kv-tf32,kv-bf16,kv-compile,kv-cudagraph,kv-bf16-cudagraph,eager-cudagraph
    mkdir -p "$OUT/opt"
    sweep() { local j="$OUT/opt/$1.json"; shift; [ -s "$j" ] && { echo "skip $j"; return; }
        python scripts/bench_kv.py "$@" --modes $M --json "$j" 2>&1 | grep -a -E "^\[|^eager|^kv|Error|Traceback" | tee "${j%.json}.txt"; }
    sweep wmt14-bs16 --config config/eval/wmt14.yaml
    sweep xsum-bs16 --config config/eval/xsum.yaml
    sweep wmt14-cfg1-bs16 --config config/eval/wmt14.yaml --cfg 1.0
    sweep wmt14-bs64 --config config/eval/wmt14.yaml --bs 64
    sweep xsum-bs64 --config config/eval/xsum.yaml --bs 64
fi

echo "=== done $(date -u)"
touch "$OUT/DONE"
