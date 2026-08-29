# EK100 Real-Time Verification Experiment

## Purpose

Run the real-time verification on the EK100 dataset: compute the ratio between the average inference time per video and the original video duration (the real-time factor) and evaluate whether the system meets the real-time processing requirement.

## Key metrics

- **Real-Time Factor (RTF)** = inference time / video duration
  - RTF < 1.0: the real-time requirement is met (inference is faster than playback)
  - RTF >= 1.0: the real-time requirement is not met

- **Video duration computation**:
  - From the timestamps: `stop_timestamp - start_timestamp`
  - From the frame count: `(stop_frame - start_frame) / 60 fps` (fallback)
  - The frame rate of EK100 is 60 fps

## Usage

### Quick start

```bash
# Use the default parameters (process 100 videos)
cd efficiency/ek100_realtime
bash run.sh

# Or specify the parameters
bash run.sh 50 3 3
# Parameter description:
#   $1: number of videos to process (default 100)
#   $2: number of top-k strategies (default 3)
#   $3: base model selection (default 3)
```

### Run the Python script directly

```bash
python efficiency/ek100_realtime/main_ek100_realtime.py \
    --max_videos 100 \
    --use_playbook \
    --top_k 3 \
    --base 3
```

## Output files

After the experiment the following files are generated:

1. **Detailed result file**: `efficiency/ek100_realtime/ek100_realtime_results.txt`
   - Contains the recognition result and the timing information of each video
   - CSV format, it contains a real-time factor column

2. **Timing data file**: `efficiency/ek100_realtime/ek100_realtime_timing.csv`
   - Contains the detailed timing data of each video
   - The fields include:
     - `video_id`: the video ID
     - `total_inference_time`: the total inference time
     - `video_duration`: the video duration
     - `real_time_factor`: the real-time factor
     - `is_realtime`: whether the real-time requirement is met (boolean)
     - The per-stage time breakdown (noun selection, knowledge base, action combination, action scoring)

3. **Statistics summary file**: `efficiency/ek100_realtime/ek100_realtime_timing_summary.json`
   - The statistics summary in JSON format
   - It contains the average, the standard deviation, the real-time analysis, etc.

## Statistics output

The experiment prints a detailed statistical analysis to the console:

### Time breakdown
- Noun selection time
- Knowledge base retrieval/update time
- Action combination time
- Action scoring time
- Total inference time

### Real-time analysis
- Average video duration
- Average inference time
- Average real-time factor
- Number and ratio of videos that meet the real-time requirement
- Distribution of the real-time factor (min, max, median, IQR)
- Examples of the most and the least real-time videos

## Difference from the original experiment

This is an **independent real-time verification experiment**, different from the original inference time breakdown experiment (`efficiency/ek100/main_ek100_efficiency.py`):

| Aspect | Original experiment (ek100) | Real-time verification (ek100_realtime) |
|------|----------------|--------------------------------|
| Purpose | Analyze the ratio of the inference time per stage | Evaluate whether the system meets the real-time requirement |
| Key metric | Per-stage time breakdown | Real-time factor (RTF) |
| New fields | - | video_duration, real_time_factor, is_realtime |
| Output focus | Time ratio in percent | Real-time pass rate, RTF distribution |

## Notes

1. **Does not affect the original experiment**: the code is fully independent and does not modify the original efficiency experiment code
2. **Video duration computation**: the timestamps are preferred, the frame count is the fallback
3. **How to read the real-time factor**:
   - RTF = 0.5 means inference is twice as fast as playback (far real-time)
   - RTF = 1.0 means inference is as fast as playback (the real-time limit)
   - RTF = 2.0 means inference is half as fast as playback (not real-time)

## Dependencies

Make sure the following modules are available:
- The `action_recognition_with_timing` function in `moduls.py`
- The utility functions in `utils.py`
- Data processing libraries such as pandas, numpy and scipy
