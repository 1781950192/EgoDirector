# EK100 Real-Time Verification Experiment - Quick Guide

## 📋 Overview

This experiment verifies whether the action recognition system meets the **real-time processing** requirement on the EK100 dataset.

### Key concept

**Real-Time Factor (RTF)** = inference time / video duration

- ✅ **RTF < 1.0**: the real-time requirement is met (inference is faster than playback)
- ❌ **RTF >= 1.0**: the real-time requirement is not met (inference is slower than playback)

### Example
- Video duration: 2 s
- Inference time: 1.5 s
- Real-time factor: 1.5 / 2.0 = 0.75 ✅ (real-time)

- Video duration: 2 s  
- Inference time: 3 s
- Real-time factor: 3.0 / 2.0 = 1.5 ❌ (not real-time)

## 🚀 Quick start

```bash
# Enter the experiment directory
cd /mnt/data/xgl/mycode/action_agent_vllm/efficiency/ek100_realtime

# Run the experiment (100 videos by default)
bash run.sh

# Or specify the parameters
bash run.sh 50    # Process only 50 videos
```

## 📊 Output

After the experiment the following is generated:

1. **ek100_realtime_results.txt** - detailed recognition results
2. **ek100_realtime_timing.csv** - timing data of each video
3. **ek100_realtime_timing_summary.json** - statistics summary

### Key statistics

The console prints:
- Average video duration
- Average inference time
- **Average real-time factor**
- **Ratio of videos that meet the real-time requirement**
- Distribution of the real-time factor (min, max, median)
- Examples of the most and the least real-time videos

## 🔍 Difference from the original experiment

| Item | Original experiment | Real-time verification experiment |
|------|--------|----------------|
| Path | `efficiency/ek100/` | `efficiency/ek100_realtime/` |
| Purpose | Analyze the time ratio per stage | Evaluate the real-time processing ability |
| Key metric | Time breakdown in percent | Real-time factor (RTF) |
| Code changes | - | **Fully independent, does not affect the original experiment** |

## ⚙️ Technical details

- **Video duration**: from the timestamp fields of the CSV (`start_timestamp`, `stop_timestamp`)
- **Frame rate**: the EK100 dataset is 60 fps
- **Fallback**: it can also be computed from the frame count `(stop_frame - start_frame) / 60`

## 📝 Notes

✅ **The original experiment code is unchanged** - this is a fully independent new experiment  
✅ **Independent output directory** - the results are saved in `efficiency/ek100_realtime/`  
✅ **Detailed statistical analysis** - includes the RTF distribution and the pass rate  

## 🎯 Next steps

After running the experiment, check the statistics summary to see the real-time performance of the system. If the real-time factor is high, you can consider:
- Optimizing the model inference speed
- Reducing the number of processed frames
- Using a lighter model
