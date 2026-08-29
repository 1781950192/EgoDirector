#!/bin/bash

# EGTEA real-time verification experiment startup script
# Compute the average inference time per video against the original video duration and report the "real-time factor"

echo "=========================================="
echo "EGTEA real-time verification experiment"
echo "=========================================="
echo ""

# Set the parameters
MAX_VIDEOS=${1:-100}  # Process 100 videos by default
TOP_K=${2:-3}
BASE=${3:-0}

echo "Configuration:"
echo "  - Number of videos: $MAX_VIDEOS"
echo "  - Number of top-k strategies: $TOP_K"
echo "  - Base model: $BASE"
echo ""

# Create the output directory
mkdir -p efficiency/egtea_realtime

# Run the experiment
python efficiency/egtea_realtime/main_egtea_realtime.py \
    --max_videos $MAX_VIDEOS \
    --top_k $TOP_K \
    --base $BASE

echo ""
echo "=========================================="
echo "Experiment finished!"
echo "Result files:"
echo "  - Detailed results: efficiency/egtea_realtime/egtea_realtime_results.txt"
echo "  - Timing data: efficiency/egtea_realtime/egtea_realtime_timing.csv"
echo "  - Statistics summary: efficiency/egtea_realtime/egtea_realtime_timing_summary.json"
echo "=========================================="
