# Efficiency Experiment Visualization Tool

## Overview

This tool visualizes the results of the efficiency experiments and of the real-time verification experiments, and generates pie charts of the inference time distribution.

## Supported datasets

- **EK100**: the EPIC-KITCHENS-100 dataset
- **EGTEA**: the EGTEA Gaze+ dataset  
- **GTEA**: the Georgia Tech Egocentric Activity dataset

## Supported experiment types

### 1. Efficiency experiment
- Directory: `efficiency/{dataset}/`
- Statistics file: `{dataset}_efficiency_timing_summary.json`
- It shows the time ratio of the four stages

### 2. Realtime experiment (real-time verification)
- Directory: `efficiency/{dataset}_realtime/`
- Statistics file: `{dataset}_realtime_timing_summary.json`
- It shows the real-time inference time distribution and the real-time factor analysis

## Usage

### Basic usage

#### Generate the efficiency pie chart of a single dataset

```bash
python efficiency/visualize_pie_chart.py --dataset ek100 --experiment-type efficiency
```

#### Generate the real-time pie chart of a single dataset

```bash
python efficiency/visualize_pie_chart.py --dataset gtea --experiment-type realtime
```

### Advanced options

```bash
# Specify the output file path
python efficiency/visualize_pie_chart.py --dataset egtea --experiment-type realtime --output my_custom_chart.png

# Do not show the percentage labels
python efficiency/visualize_pie_chart.py --dataset ek100 --no-percentage

# Generate a comparison chart for multiple datasets (efficiency experiment)
python efficiency/visualize_pie_chart.py --comparison --experiment-type efficiency

# Generate a comparison chart for multiple datasets (real-time experiment)
python efficiency/visualize_pie_chart.py --comparison --experiment-type realtime
```

### Parameters

| Parameter | Description | Default | Options |
|------|------|--------|--------|
| `--dataset` | Dataset name | gtea | ek100, egtea, gtea |
| `--experiment-type` | Experiment type | efficiency | efficiency, realtime |
| `--comparison` | Generate a comparison chart | False | - |
| `--output` | Output file path | auto-generated | any path |
| `--no-percentage` | Do not show the percentages | False | - |

## Output examples

### Pie chart of a single dataset

The generated file name has the format:
- Efficiency experiment: `{dataset}_efficiency_pie_chart.png`
- Realtime experiment: `{dataset}_realtime_pie_chart.png`

For example:
- `ek100_efficiency_pie_chart.png`
- `gtea_realtime_pie_chart.png`

### Comparison chart of multiple datasets

The generated file name has the format:
- Efficiency experiment: `efficiency_comparison_efficiency_pie_charts.png`
- Realtime experiment: `efficiency_comparison_realtime_pie_charts.png`

## Contents of the pie chart

### Efficiency experiment
It shows the time ratio of the four inference stages:
1. **Noun Selection**
2. **Knowledge Base** (retrieval/update)
3. **Action Combination**
4. **Action Scoring**

### Realtime experiment
It also shows the time ratio of the four stages, but computed from the data of the real-time verification experiment.

## Prerequisites

Make sure the corresponding experiment has been run and the statistics have been generated:

```bash
# Run the efficiency experiment
sh efficiency/ek100/run.sh
sh efficiency/egtea/run.sh
sh efficiency/gtea/run.sh

# Or run the real-time verification experiment
sh efficiency/ek100_realtime/run.sh
sh efficiency/egtea_realtime/run.sh
sh efficiency/gtea_realtime/run.sh
```

## Python dependencies

```bash
pip install matplotlib
```

## Notes

1. The experiment has to be run first so that the statistics file exists before visualizing
2. The real-time experiment data contains additional real-time factor analysis information
3. The pie chart layout is adjusted automatically so that it stays readable
4. A high resolution output is recommended (300 DPI by default) for the best result
