# 🎬 EgoDirector

An agent-based action recognition framework powered by Vision-Language Models and vLLM, supporting multiple egocentric video datasets.

## 📑 Table of Contents

- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Model Preparation](#-model-preparation)
- [HOS Modules](#-hos-modules)
- [vLLM Serving](#-vllm-serving)
- [Quick Start](#-quick-start)
- [Evaluation](#-evaluation)
- [Visualization](#-visualization)

---

## 🔧 Installation

```bash
conda create -n egodirector python=3.11
conda activate egodirector

pip install vllm
pip install pandas
```

---

## 📦 Dataset Preparation

| Dataset | Source |
|---------|--------|
| **GTEA / GTEA+** | [EGTEA Gaze+ Downloader](https://github.com/amitsou/EGTEA_Gaze_Plus_Downloader) |
| **EK100** | [Epic-Kitchens 2020-100](https://epic-kitchens.github.io/2020-100) |

---

## 🤖 Model Preparation

Download the required Vision-Language Models from Hugging Face:

| Model | Hugging Face Link |
|-------|-------------------|
| **Qwen3-VL-8B-Instruct** | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| **Qwen3-VL-32B-Instruct** | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |
| **Gemma-4-12B-IT** | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) |
| **MiniCPM-V-4.6** | [openbmb/MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) |
| **InternVL-3.5-8B** | [internlm/CapRL-InternVL3.5-8B](https://huggingface.co/internlm/CapRL-InternVL3.5-8B) |

---

## 🏠 HOS Modules

**VISOR-HOS**: [epic-kitchens/VISOR-HOS](https://github.com/epic-kitchens/VISOR-HOS)

**VISOR-HOS**: https://pan.baidu.com/s/1KEdXH0eo5HX_anZkKUZE7A?pwd=1111 Access code: 1111 

> 💡 **Tip:** You can also download pre-inferred HOS images directly to skip the inference step.

---

## 🚀 vLLM Serving

Launch the model server with the following command:

```bash
vllm serve Your_model_path \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.4 \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --served-model-name Qwen3-VL-8B-Instruct
```

> ⚙️ Adjust `--tensor-parallel-size` and `--gpu-memory-utilization` based on your hardware.

---

## ▶️ Quick Start

Before running, modify the built-in parameters (e.g., dataset paths) in the script files.

```bash
python main_ek100.py          # Run on the EK100 dataset
python main_gtea.py           # Run on the GTEA dataset
python main_gtea+.py          # Run on the GTEA+ dataset
```

---

## 📊 Evaluation

### Efficiency Evaluation

```bash
bash efficiency/egtea/run.sh              # Evaluate efficiency on GTEA+ dataset
```

> The other two datasets follow the same pattern.

### Result Evaluation

```bash
python ek100_process/trans.py             # Evaluate verb/noun/action accuracy on EK100
python ek100_process/trans_wups.py        # Evaluate verb/noun/action WUPS accuracy on EK100
```

> The other two datasets follow the same pattern.

---

## 🌐 Visualization

Launch the action recognition visualization website:

```bash
bash start_web.sh
```
