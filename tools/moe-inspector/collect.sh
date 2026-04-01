#!/usr/bin/env bash
# Collect MoE routing data for code and NL domains.
#
# Usage:
#   ./collect.sh [--ngl N] [--n-predict N] [--ctx-size N]
#
# Environment:
#   MOE_BIN        path to llama-moe-inspector (default: ../../../build/bin/llama-moe-inspector)
#   MOE_MODEL      HF repo or local path  (default: unsloth/Qwen3.5-35B-A3B-GGUF:UD-IQ2_XXS)
#   LLAMA_CACHE    local HF cache dir (passed through to the binary)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

BIN="${MOE_BIN:-$REPO_ROOT/build/bin/llama-moe-inspector}"
MODEL="${MOE_MODEL:-unsloth/Qwen3.5-35B-A3B-GGUF:UD-IQ2_XXS}"
N_PREDICT="${N_PREDICT:-128}"
CTX_SIZE="${CTX_SIZE:-4096}"
NGL="${NGL:-0}"   # GPU layers; 0 = CPU-only

CODE_PROMPTS="$SCRIPT_DIR/data/code_prompts.txt"
NL_PROMPTS="$SCRIPT_DIR/data/nl_prompts.txt"
CODE_OUT="$SCRIPT_DIR/data/code_routing.jsonl"
NL_OUT="$SCRIPT_DIR/data/nl_routing.jsonl"

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --ngl)       NGL="$2";       shift 2 ;;
        --n-predict) N_PREDICT="$2"; shift 2 ;;
        --ctx-size)  CTX_SIZE="$2";  shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

if [[ ! -x "$BIN" ]]; then
    echo "error: binary not found or not executable: $BIN"
    echo "Build first:  cmake --build $REPO_ROOT/build --target llama-moe-inspector -j \$(nproc)"
    exit 1
fi

run_domain() {
    local domain="$1"
    local prompt_file="$2"
    local out_file="$3"

    echo "==> collecting domain='$domain'  model='$MODEL'  ngl=$NGL  n_predict=$N_PREDICT"

    # Remove previous output so we don't append to stale data.
    rm -f "$out_file"

    local model_flag="-hf"
    if [[ -f "$MODEL" ]]; then
        model_flag="-m"
    fi

    "$BIN" \
        $model_flag "$MODEL" \
        -f          "$prompt_file" \
        -c          "$CTX_SIZE" \
        -n          "$N_PREDICT" \
        --temp      0 \
        --seed      42 \
        --no-mmap \
        -ngl        "$NGL" \
        --moe-log   "$out_file" \
        --moe-domain "$domain" \
        --log-disable

    local n_records
    n_records=$(wc -l < "$out_file" 2>/dev/null || echo 0)
    echo "    wrote $n_records records -> $out_file"

    if [[ "$n_records" -eq 0 ]]; then
        echo "error: no records written for domain=$domain"
        exit 1
    fi
}

run_domain "code" "$CODE_PROMPTS" "$CODE_OUT"
run_domain "nl"   "$NL_PROMPTS"  "$NL_OUT"

echo
echo "Collection complete."
echo "  code records : $(wc -l < "$CODE_OUT")"
echo "  nl   records : $(wc -l < "$NL_OUT")"
echo
echo "Next step: python3 analyze.py --code $CODE_OUT --nl $NL_OUT --out out/"
