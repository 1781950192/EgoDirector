#!/bin/bash

# EK100 real-time verification experiment startup script
# Compute the average inference time per video against the original video duration and report the "real-time factor"

echo "=========================================="
echo "EK100 real-time verification experiment"
echo "=========================================="
echo ""

# Set the parameters
MAX_VIDEOS=${1:-100}  # Process 100 videos by default
USE_PLAYBOOK="--use_playbook"
TOP_K=${2:-3}
BASE=${3:-3}

echo "Configuration:"
echo "  - Number of videos: $MAX_VIDEOS"
echo "  - Use playbook: $USE_PLAYBOOK"
echo "  - Number of top-k strategies: $TOP_K"
echo "  - Base model: $BASE"
echo ""

# Create the output directory
mkdir -p efficiency/ek100_realtime

# Run the experiment
python efficiency/ek100_realtime/main_ek100_realtime.py \
    --max_videos $MAX_VIDEOS \
    $USE_PLAYBOOK \
    --top_k $TOP_K \
    --base $BASE

echo ""
echo "=========================================="
echo "Experiment finished!"
echo "Result files:"
echo "  - Detailed results: efficiency/ek100_realtime/ek100_realtime_results.txt"
echo "  - Timing data: efficiency/ek100_realtime/ek100_realtime_timing.csv"
echo "  - Statistics summary: efficiency/ek100_realtime/ek100_realtime_timing_summary.json"
echo "=========================================="
