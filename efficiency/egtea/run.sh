#!/bin/bash

# EGTEA efficiency experiment startup script

echo "=========================================="
echo "EGTEA efficiency experiment - inference time breakdown"
echo "=========================================="
echo ""

# Check the vLLM service (use Python instead of curl)
echo "Checking the vLLM service status..."

python3 << 'PYEOF'
import urllib.request
import sys

try:
    url = "http://localhost:8000/v1/models"
    response = urllib.request.urlopen(url, timeout=5)
    print("✓ The vLLM service is running")
except Exception as e:
    print("✗ The vLLM service is not running or cannot be reached")
    print(f"  Error: {e}")
    print("")
    print("Please start the vLLM service first:")
    print("  vllm serve /mnt/data/xgl/modules/qwen_vl_8b \\")
    print("    --dtype bfloat16 --max-model-len 32768 \\")
    print("    --gpu-memory-utilization 0.4 \\")
    print("    --tensor-parallel-size 1 --host 0.0.0.0 --port 8000 \\")
    print("    --enable-chunked-prefill --max-num-batched-tokens 16384 \\")
    print("    --served-model-name Qwen3-VL-8B-Instruct")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""

cd "$(dirname "$0")/../.."

python efficiency/egtea/main_egtea_efficiency.py \
    --max_videos 100 \
    --base 0 \
    --use_hos 0 \
    --top_k 3 \
    --max_strategies 50 \
    --merge_threshold 200 \
    --batch_size 20
