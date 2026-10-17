#!/usr/bin/env bash
# Rent one vast.ai GPU, run scripts/vast/remote.sh on it, copy the results back, destroy the instance.
#
#   vastai set api-key <KEY>                         # once
#   vastai create ssh-key ~/.ssh/id_ed25519.pub      # once
#   scripts/vast/launch.sh search 4090               # list offers (also: h100, h100sxm)
#   scripts/vast/launch.sh run <OFFER_ID>            # rent, run, fetch results, destroy
#
# Env: MAX_MINUTES (default 90) destroys the instance after that long no matter what; KEEP=1 skips the destroy.
set -euo pipefail
IMAGE=${IMAGE:-pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime}
DISK=${DISK:-40}
MAX_MINUTES=${MAX_MINUTES:-90}
BRANCH=${BRANCH:-mac-mps}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
FILTER="num_gpus=1 verified=true rentable=true reliability>0.98 cuda_vers>=12.6 direct_port_count>=1 inet_down>=200 disk_space>=$DISK"

case "${1:-}" in
search)
    case "${2:-4090}" in
        4090) GPU="gpu_name=RTX_4090" ;;
        h100) GPU="gpu_name in [H100_SXM,H100_PCIE,H100_NVL]" ;;
        h100sxm) GPU="gpu_name=H100_SXM" ;;
        *) GPU="gpu_name=$2" ;;
    esac
    vastai search offers "$GPU $FILTER" -o dph | head -15
    exit ;;
run) OFFER=${2:?usage: launch.sh run OFFER_ID}
    # the monitor below can't answer a passphrase prompt: refuse to rent until ssh works non-interactively
    ssh-add -l >/dev/null 2>&1 || ssh-keygen -y -P "" -f ~/.ssh/id_ed25519 >/dev/null 2>&1 || {
        echo "no usable ssh key: unlock it first with  ssh-add --apple-use-keychain ~/.ssh/id_ed25519"; exit 1; } ;;
*) sed -n 2,10p "$0"; exit 1 ;;
esac

ONSTART="nohup bash -c 'mkdir -p /workspace && cd /workspace && (command -v git || (apt-get update -qq && apt-get install -y -qq git)) >/dev/null 2>&1 && git clone -q -b $BRANCH https://github.com/velaia/ncm.git ncm && BRANCH=$BRANCH bash ncm/scripts/vast/remote.sh' >/workspace/onstart.log 2>&1 &"
ID=$(vastai create instance "$OFFER" --image "$IMAGE" --disk "$DISK" --ssh --direct --cancel-unavail \
       --label ncm-kv-bench --onstart-cmd "$ONSTART" --raw | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")
echo "instance $ID created (offer $OFFER); billing has started"

destroy() {
    if [ "${KEEP:-0}" = 1 ]; then
        echo "KEEP=1: instance $ID is still running and billing; destroy with: vastai destroy instance $ID"
    else
        vastai destroy instance "$ID" -y >/dev/null 2>&1 || vastai destroy instance "$ID"
        echo "instance $ID destroyed"
    fi
}
trap destroy EXIT

deadline=$(( $(date +%s) + MAX_MINUTES * 60 ))
status() { vastai show instance "$ID" --raw | python3 -c "import json,sys; print(json.load(sys.stdin).get('actual_status') or 'pending')"; }
until [ "$(status)" = running ]; do
    [ "$(date +%s)" -lt "$deadline" ] || { echo "timed out waiting for the instance to start"; exit 1; }
    sleep 20
done
URL=$(vastai ssh-url "$ID")              # ssh://root@HOST:PORT
HOSTPORT=${URL#ssh://}; HOST=${HOSTPORT%:*}; PORT=${HOSTPORT##*:}
SSH=(ssh -p "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "$HOST")
echo "running; ssh -p $PORT $HOST   (log: /workspace/results/log.txt)"

state=""
while :; do
    [ "$(date +%s)" -lt "$deadline" ] || { echo "MAX_MINUTES=$MAX_MINUTES reached"; break; }
    state=$("${SSH[@]}" 'cd /workspace/results 2>/dev/null && { [ -f DONE ] && echo DONE; [ -f FAILED ] && echo FAILED; tail -n 1 log.txt; }' 2>/dev/null || true)
    echo "[$(date +%H:%M)] $(echo "$state" | tail -n 1 | cut -c1-150)"
    case "$state" in DONE*|FAILED*) break ;; esac
    sleep 60
done

GPU=$("${SSH[@]}" "nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_'" 2>/dev/null || echo gpu)
DEST="$ROOT/eval/vast/$(date +%Y%m%d-%H%M)-$GPU-$ID"
mkdir -p "$DEST"
scp -q -P "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -r "$HOST:/workspace/results/*" "$DEST/" || echo "copy failed"
"${SSH[@]}" 'cat /workspace/onstart.log' >"$DEST/onstart.log" 2>/dev/null || true
echo "results in $DEST"
grep -h -E "^\[|^eager|^kv|^speedup|BLEU|R-1|^ +[0-9]" "$DEST"/log.txt 2>/dev/null | tail -n 60 || true
