#!/bin/bash

# Unified startup script of the efficiency experiments
# Supports three datasets: EK100, EGTEA, GTEA

DATASET=${1:-"ek100"}
MAX_VIDEOS=${2:-100}

echo "=========================================="
echo "Action recognition efficiency experiment - inference time breakdown"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Number of videos: $MAX_VIDEOS"
echo ""

# Check the vLLM service (use Python instead of curl)
echo "Checking the vLLM service status..."

python3 << 'EOF'
import urllib.request
import json
import sys

try:
    url = "http://localhost:8000/v1/models"
    response = urllib.request.urlopen(url, timeout=5)
    data = json.loads(response.read().decode('utf-8'))
    
    print("✓ The vLLM service is running (port 8000)")
    print("  Available models:")
    
    if 'data' in data:
        for model in data['data']:
            model_id = model.get('id', 'unknown')
            print(f"    - {model_id}")
    else:
        print("    (Unable to parse the model list)")
        
except Exception as e:
    print("✗ The vLLM service is not running or cannot be reached")
    print(f"  Error: {e}")
    print("")
    print("Please start the vLLM service first, for example:")
    print("  vllm serve /mnt/data/xgl/modules/qwen_vl_8b \\")
    print("    --dtype bfloat16 \\")
    print("    --max-model-len 32768 \\")
    print("    --gpu-memory-utilization 0.4 \\")
    print("    --tensor-parallel-size 1 \\")
    print("    --host 0.0.0.0 \\")
    print("    --port 8000 \\")
    print("    --enable-chunked-prefill \\")
    print("    --max-num-batched-tokens 16384 \\")
    print("    --served-model-name Qwen3-VL-8B-Instruct")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo "Starting the efficiency experiment..."
echo ""

case $DATASET in
    ek100)
        echo "Running the EK100 efficiency experiment..."
        cd "$(dirname "$0")/.."
        python efficiency/ek100/main_ek100_efficiency.py --max_videos $MAX_VIDEOS --base 3 --use_hos 0 --top_k 3
        ;;
    egtea)
        echo "Running the EGTEA efficiency experiment..."
        cd "$(dirname "$0")/.."
        python efficiency/egtea/main_egtea_efficiency.py --max_videos $MAX_VIDEOS --base 0 --use_hos 0 --top_k 3
        ;;
    gtea)
        echo "Running the GTEA efficiency experiment..."
        cd "$(dirname "$0")/.."
        python efficiency/gtea/main_gtea_efficiency.py --max_videos $MAX_VIDEOS --base 0 --use_hos 0 --top_k 3
        ;;
    *)
        echo "Error: unsupported dataset '$DATASET'"
        echo "Supported datasets: ek100, egtea, gtea"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Experiment finished!"
echo "=========================================="
echo ""
echo "View the results:"
echo "  cat ${DATASET}_efficiency_timing_summary.json"
echo ""
