# Efficiency Experiment - Quick Reference

## 🎯 Start with a single command

```bash
bash efficiency/run_efficiency.sh <dataset> [videos]
```

## 📊 Three datasets

| Dataset | Command | Estimated time |
|--------|------|---------|
| **EK100** | `bash efficiency/run_efficiency.sh ek100 100` | ~50 min |
| **EGTEA** | `bash efficiency/run_efficiency.sh egtea 100` | ~40 min |
| **GTEA** | `bash efficiency/run_efficiency.sh gtea 100` | ~40 min |

## 🔍 Four timing stages

```
┌─────────────────────────────────────┐
│ 1. 🔍 Select noun        (~30%)          │
│ 2. 📚 Knowledge base        (~1%)           │
│ 3. 🔗 Combine actions      (~47%) ⚠️ bottleneck  │
│ 4. ⭐ Score actions      (~22%)          │
└─────────────────────────────────────┘
```

## 📁 Directory structure

```
efficiency/
├── run_efficiency.sh          # Unified entry point
├── README.md                  # Detailed documentation
├── ek100/
│   ├── main_ek100_efficiency.py
│   └── run.sh                 # bash efficiency/ek100/run.sh
├── egtea/
│   ├── main_egtea_efficiency.py
│   └── run.sh                 # bash efficiency/egtea/run.sh
└── gtea/
    ├── main_gtea_efficiency.py
    └── run.sh                 # bash efficiency/gtea/run.sh
```

## 📊 Output files (project root)

```
ek100_efficiency_results.txt          # Recognition results
ek100_efficiency_timing.csv           # Detailed CSV
ek100_efficiency_timing_summary.json  # Statistics summary
```

## 💡 Common commands

```bash
# Quick test (1 video)
python efficiency/ek100/main_ek100_efficiency.py --max_videos 1 --use_playbook

# View the statistics
cat ek100_efficiency_timing_summary.json | python -m json.tool

# Custom parameters
python efficiency/ek100/main_ek100_efficiency.py \
    --max_videos 50 \
    --base 3 \
    --use_playbook \
    --top_k 3
```

## ✅ Prerequisite check

```bash
# Check the vLLM service
curl http://localhost:8000/v1/models

# Check the dataset
ls /mnt/data/xgl/mydata/ek100/
```

## 🎯 Key metrics

After running, pay attention to:
- **Average time**: the average of each stage
- **Standard deviation**: a stability indicator
- **Ratio**: which stage costs the most time
- **Total time**: inference vs processing

## ⚠️ Notes

- vLLM must run on `localhost:8000`
- 16GB+ GPU memory is recommended
- The first run is slower (unknown nouns are generated)
- The output files are written to the project root

## 📖 Detailed documentation

- `efficiency/README.md` - full documentation
- `efficiency/MIGRATION_GUIDE.md` - migration guide
- `efficiency/COMPLETION_SUMMARY.md` - completion summary

---

**Good luck with the experiments!**
