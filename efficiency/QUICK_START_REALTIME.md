# Quick reference - GTEA/EGTEA real-time verification

## 🚀 Quick start

### 1️⃣ Run the GTEA real-time experiment

```bash
sh efficiency/gtea_realtime/run.sh 100 3 0
```

### 2️⃣ Run the EGTEA real-time experiment

```bash
sh efficiency/egtea_realtime/run.sh 100 3 0
```

### 3️⃣ View the results

```bash
# View the statistics summary
cat efficiency/gtea_realtime/gtea_realtime_timing_summary.json
cat efficiency/egtea_realtime/egtea_realtime_timing_summary.json

# View the detailed data
head efficiency/gtea_realtime/gtea_realtime_timing.csv
head efficiency/egtea_realtime/egtea_realtime_timing.csv
```

### 4️⃣ Generate the visualization

```bash
# GTEA pie chart
python efficiency/visualize_pie_chart.py --dataset gtea --experiment-type realtime

# EGTEA pie chart
python efficiency/visualize_pie_chart.py --dataset egtea --experiment-type realtime

# Compare all datasets
python efficiency/visualize_pie_chart.py --comparison --experiment-type realtime
```

## 📊 How to read the key metrics

### Real-time factor (RTF)
- **RTF < 1.0**: ✅ the system can process in real time (inference is faster than playback)
- **RTF = 1.0**: ⚠️ just meets the real-time requirement
- **RTF > 1.0**: ❌ cannot process in real time (inference is slower than playback)

### Example
```
Average real-time factor: 1.82
→ The inference time is 1.82 times the video duration
→ It takes 1.82 s to process 1 s of video
→ The real-time requirement is not met
```

## 🔧 Common parameters

| Parameter | Description | Example |
|------|------|------|
| `max_videos` | Number of videos to process | `--max_videos 50` |
| `top_k` | Number of strategies used | `--top_k 5` |
| `base` | Model selection (0-3) | `--base 3` |
| `use_playbook` | Use the playbook | `--use_playbook` |
| `use_hos` | Use the HOS features | `--use_hos 0` |

## 📁 File locations

```
efficiency/
├── gtea_realtime/          # GTEA real-time experiment
│   ├── main_gtea_realtime.py
│   ├── run.sh
│   └── gtea_realtime_timing_summary.json  ← main result
│
├── egtea_realtime/         # EGTEA real-time experiment
│   ├── main_egtea_realtime.py
│   ├── run.sh
│   └── egtea_realtime_timing_summary.json  ← main result
│
└── visualize_pie_chart.py  # Visualization tool
```

## 💡 Tips

- The first run creates the playbook, later runs load it automatically
- The results are saved in the `*_summary.json` file of the corresponding directory
- Use `--max_videos` to control the scale of the experiment (default 100)
- The visualization images are saved as PNG by default (300 DPI)
