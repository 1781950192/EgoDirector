# Efficiency Experiment - Inference Time Breakdown

## 📋 Overview

This directory contains the efficiency experiment code for the three datasets (EK100, EGTEA, GTEA). It analyzes in detail how the inference time of the action recognition system is distributed.

## 🎯 Features

For every test video the time is broken down into **four key stages**:

| Stage | Description | Function |
|------|------|---------|
| 🔍 **Noun selection** | Call the LLM to select candidate nouns | `select_nouns()` |
| 📚 **Knowledge base retrieval/update** | Query and update pre_noun & verb_noun | `pre_noun_add()`, `verb_noun_add()` |
| 🔗 **Action combination** | Generate action combinations from nouns and verbs | `combine_actions()` |
| ⭐ **Action scoring** | Score the confidence of the generated actions | `select_actions()` |

## 📁 Directory structure

```
efficiency/
├── README.md                    # This document
├── run_efficiency.sh            # Unified startup script
├── ek100/                       # EK100 efficiency experiment
│   ├── main_ek100_efficiency.py # Main script
│   └── run.sh                   # Quick startup script
├── egtea/                       # EGTEA efficiency experiment
│   ├── main_egtea_efficiency.py # Main script
│   └── run.sh                   # Quick startup script
└── gtea/                        # GTEA efficiency experiment
    ├── main_gtea_efficiency.py  # Main script
    └── run.sh                   # Quick startup script
```

## 🚀 Quick start

### Method 1: unified startup script (recommended)

```bash
# Basic usage
bash efficiency/run_efficiency.sh <dataset> [max_videos]

# Example: run EK100 on 100 videos
bash efficiency/run_efficiency.sh ek100 100

# Example: run EGTEA on 50 videos
bash efficiency/run_efficiency.sh egtea 50

# Example: run GTEA on all videos
bash efficiency/run_efficiency.sh gtea
```

### Method 2: start each dataset independently

```bash
# EK100
bash efficiency/ek100/run.sh

# EGTEA
bash efficiency/egtea/run.sh

# GTEA
bash efficiency/gtea/run.sh
```

### Method 3: run the Python script directly

```bash
# EK100
python efficiency/ek100/main_ek100_efficiency.py --max_videos 100 --use_playbook

# EGTEA
python efficiency/egtea/main_egtea_efficiency.py --max_videos 100 --use_playbook

# GTEA
python efficiency/gtea/main_gtea_efficiency.py --max_videos 100 --use_playbook
```

## 📊 Output files

Each dataset run generates the following files (in the project root):

### 1. Result text file
- `ek100_efficiency_results.txt`
- `egtea_efficiency_results.txt`
- `gtea_efficiency_results.txt`

It contains the recognition results and the corresponding timing information.

### 2. CSV timing data
- `ek100_efficiency_timing.csv`
- `egtea_efficiency_timing.csv`
- `gtea_efficiency_timing.csv`

Detailed timing data in CSV format, one row per video:
```csv
video_id,select_noun_time,knowledge_base_time,combine_actions_time,score_actions_time,total_inference_time,total_video_time
P01_01,8.2345,0.1523,12.4567,6.7890,27.6325,28.1234
...
```

### 3. JSON statistics summary
- `ek100_efficiency_timing_summary.json`
- `egtea_efficiency_timing_summary.json`
- `gtea_efficiency_timing_summary.json`

It contains the average, the standard deviation and the ratio statistics:
```json
{
  "dataset": "EK100",
  "num_videos": 100,
  "average_times": {
    "select_noun": 8.4523,
    "knowledge_base": 0.2156,
    "combine_actions": 12.8934,
    "score_actions": 6.0123,
    "total_inference": 27.5736,
    "total_video_processing": 28.1245
  },
  "std_times": { ... },
  "percentage_of_inference": { ... }
}
```

## 📈 Statistics

For each dataset the following is computed:

- ✅ **Average time**: the average cost of each stage
- ✅ **Standard deviation**: how stable the timing is
- ✅ **Ratio**: the percentage of each stage over the total inference time
- ✅ **Total time comparison**: inference time vs total video processing time

## 💡 Usage examples

### Full EK100 experiment

```bash
bash efficiency/ek100/run.sh
```

Expected output:
```
================================================================================
EK100 efficiency experiment statistics
================================================================================

Number of processed videos: 100

Stage                   Avg time(s)        Std dev(s)        Ratio(%)     
--------------------------------------------------------------------------------
Select noun                  8.4523          1.2345          30.65     
KB retrieval/update           0.2156          0.0892          0.78      
Combine actions                 12.8934         2.1567          46.78     
Score actions                 6.0123          0.9876          21.80     
--------------------------------------------------------------------------------
Total inference                27.5736         3.2145          100.00    
Total video time             28.1245         3.4567          -         
```

### View the statistics summary

```bash
cat ek100_efficiency_timing_summary.json | python -m json.tool
```

### Custom parameters

```bash
python efficiency/ek100/main_ek100_efficiency.py \
    --max_videos 50 \
    --base 3 \
    --use_playbook \
    --use_hos 0 \
    --top_k 3
```

## ⚙️ Parameters

The scripts of all three datasets support the following parameters:

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `--max_videos` | int | 100 | Number of videos to process |
| `--base` | int | 0/3 | Model selection (0:main model, 1:1llm, 2:2llm, 3:3llm) |
| `--use_playbook` | flag | False | Whether to use the playbook |
| `--use_hos` | int | 0 | 0:use HOS, 1:do not use HOS |
| `--top_k` | int | 3 | Take the top-K strategies from the playbook |
| `--max_strategies` | int | 50 | Maximum number of strategies per category |
| `--merge_threshold` | int | 200 | Merge threshold |
| `--batch_size` | int | 20 | Batch size |

### Dataset-specific parameters

**EK100:**
- `--base` defaults to 3

**EGTEA:**
- `--label_file`: path of the label file
- `--video_base_dir`: base directory of the videos
- `--hos_path`: path of the HOS images

**GTEA:**
- `--labels_dir`: label directory
- `--videos_base_path`: base path of the videos
- `--hos_path`: path of the HOS images

## ⏱️ Estimated running time

| Dataset | Number of videos | Estimated time |
|--------|---------|---------|
| EK100 | 100 | ~40-60 min |
| EGTEA | 100 | ~30-50 min |
| GTEA | 100 | ~30-50 min |

*Note: the actual time depends on the GPU performance and the API response speed.*

## ✅ Prerequisites

1. **The vLLM service is running**
   ```bash
   curl http://localhost:8000/v1/models
   ```

2. **The dataset paths are configured correctly**
   - EK100: `/mnt/data/xgl/mydata/ek100/`
   - EGTEA: `/mnt/data/xgl/mydata/EGTEA++/EGTEA/`
   - GTEA: `/mnt/data/xgl/mydata/EGTEA++/EGTEA/`

3. **The configuration files exist**
   - `context/verb_noun.json`
   - `context/pre_noun.json`
   - The corresponding playbook file

## 🔍 Data analysis suggestions

### Bottleneck identification

The **action combination stage** is usually the main bottleneck (about 45-50% of the time). Optimization suggestions:
- Shorten the prompt
- Remove unnecessary context information
- Consider using a faster model

### Stability analysis

Look at the stages with a large standard deviation, they may indicate:
- Unstable API response time
- Network latency fluctuations
- Retries caused by insufficient GPU memory

### Cross-dataset comparison

Compare the time distribution of the three datasets:
```bash
# View the statistics summary of each dataset
cat ek100_efficiency_timing_summary.json | jq '.average_times'
cat egtea_efficiency_timing_summary.json | jq '.average_times'
cat gtea_efficiency_timing_summary.json | jq '.average_times'
```

## ⚠️ Notes

1. **The first run is slower**: unknown nouns require an API call to be generated
2. **GPU memory**: at least 16GB of GPU memory is recommended
3. **No resume support**: an interrupted run has to be restarted
4. **Output location**: all output files are saved in the project root

## 🛠️ Troubleshooting

### The vLLM service is not running
```bash
# Check the service status
curl http://localhost:8000/v1/models

# If it is not running, start the vLLM service first
```

### Wrong dataset path
Check and modify the default path parameters in the script, or specify them on the command line:
```bash
python efficiency/ek100/main_ek100_efficiency.py \
    --max_videos 100 \
    [other parameters...]
```

### Out of memory
Reduce the `--max_videos` parameter:
```bash
python efficiency/ek100/main_ek100_efficiency.py --max_videos 50
```

## 📝 Implementation

### Core module

- The `action_recognition_with_timing()` function in `moduls.py`
- Timestamps are recorded before and after each key stage
- A dict containing the detailed timing information is returned

### Timing precision

- `time.time()` is used for microsecond-level timing
- Results are written to the file on the fly to avoid running out of memory

## 📖 Related documentation

- `EFFICIENCY_EXPERIMENT_README.md` in the project root - detailed description
- `EFFICIENCY_SUMMARY.md` in the project root - implementation summary
- `EFFICIENCY_QUICKSTART.md` in the project root - quick reference

## 🎯 Next steps

1. **Run the experiment**
   ```bash
   bash efficiency/run_efficiency.sh ek100 100
   ```

2. **Analyze the results**
   ```bash
   cat ek100_efficiency_timing_summary.json
   ```

3. **Cross-dataset comparison**
   - Compare the time distribution of the three datasets
   - Identify the common performance bottlenecks
   - Define an optimization strategy

---

**Good luck with the experiments!**
