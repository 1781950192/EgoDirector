"""
Efficiency experiment visualization - pie chart
Draw a pie chart of the time ratio of the four inference stages.
Supports the EK100, EGTEA and GTEA datasets.
Supports both the efficiency and the realtime experiment types.
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import sys
import os

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_timing_summary(dataset, experiment_type='efficiency'):
    """Load the statistics summary file of a dataset.
    
    Args:
        dataset: Name of the dataset (ek100, egtea, gtea).
        experiment_type: Type of the experiment (efficiency or realtime).
    """
    if experiment_type == 'realtime':
        summary_file = f"efficiency/{dataset}_realtime/{dataset}_realtime_timing_summary.json"
    else:
        summary_file = f"efficiency/{dataset}/{dataset}_efficiency_timing_summary.json"
    
    if not os.path.exists(summary_file):
        print(f"Error: file not found {summary_file}")
        print(f"Please run the {'real-time verification' if experiment_type == 'realtime' else 'efficiency'} experiment first to generate the statistics")
        return None
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def create_pie_chart(dataset, experiment_type='efficiency', save_path=None, show_percentage=True):
    """
    Create a pie chart showing the time ratio of the four stages.
    
    Args:
        dataset: Name of the dataset (ek100, egtea, gtea).
        experiment_type: Type of the experiment (efficiency or realtime).
        save_path: Path to save the figure; if None, the figure is displayed.
        show_percentage: Whether to show the percentages in the chart.
    """
    # Load the data
    data = load_timing_summary(dataset, experiment_type)
    if data is None:
        return
    
    # Extract the percentage data (only for the efficiency type)
    if experiment_type == 'efficiency':
        percentages = data.get('percentage_of_inference', {})
        
        # Define the four stages (English labels)
        stages = {
            'select_noun': 'Noun Selection',
            'knowledge_base': 'Knowledge Base',
            'combine_actions': 'Action Combination',
            'score_actions': 'Action Scoring'
        }
        
        # Prepare the data
        labels = []
        sizes = []
        
        for key, label in stages.items():
            if key in percentages:
                labels.append(label)
                sizes.append(percentages[key])
        
        if not sizes:
            print("Error: no time ratio data found")
            return
        
        chart_title_suffix = "Inference Time Distribution"
    else:
        # realtime type: compute the ratio from the average time
        avg_times = data.get('average_times', {})
        
        stages = {
            'select_noun': 'Noun Selection',
            'knowledge_base': 'Knowledge Base',
            'combine_actions': 'Action Combination',
            'score_actions': 'Action Scoring'
        }
        
        labels = []
        sizes = []
        total_inference = avg_times.get('total_inference', 0)
        
        if total_inference > 0:
            for key, label in stages.items():
                if key in avg_times:
                    labels.append(label)
                    percentage = (avg_times[key] / total_inference) * 100
                    sizes.append(percentage)
        
        if not sizes:
            print("Error: no timing data found")
            return
        
        chart_title_suffix = "Real-Time Inference Time Distribution"
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']  # red, cyan, blue, orange
    
    # Create the pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw the pie chart
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%' if show_percentage else None,
        startangle=90,
        pctdistance=0.85,
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'width': 0.7}
    )
    
    # Set the style of the percentage text
    if show_percentage:
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
    
    # Add the title
    dataset_names = {
        'ek100': 'EK100',
        'egtea': 'EGTEA',
        'gtea': 'GTEA'
    }
    dataset_name = dataset_names.get(dataset, dataset.upper())
    
    exp_type_name = "Real-Time" if experiment_type == 'realtime' else "Efficiency"
    title = f'{dataset_name} Dataset - {exp_type_name} {chart_title_suffix}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add the legend (on the right)
    legend_labels = [f'{label}: {size:.1f}%' for label, size in zip(labels, sizes)]
    ax.legend(
        legend_labels,
        loc='center left',
        bbox_to_anchor=(1, 0.7),
        fontsize=11,
        framealpha=0.9,
        edgecolor='gray'
    )
    
    # Make sure the pie chart is circular
    ax.axis('equal')
    
    # Adjust the layout
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Pie chart saved to: {save_path}")
    else:
        # Default save path
        exp_suffix = "_realtime" if experiment_type == 'realtime' else "_efficiency"
        default_path = f"{dataset}{exp_suffix}_pie_chart.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Pie chart saved to: {default_path}")
        plt.show()
    
    plt.close()

def create_comparison_chart(datasets=['ek100', 'egtea', 'gtea'], experiment_type='efficiency', save_path=None):
    """
    Create a comparison pie chart for several datasets.
    
    Args:
        datasets: List of datasets.
        experiment_type: Type of the experiment (efficiency or realtime).
        save_path: Path to save the figure.
    """
    fig, axes = plt.subplots(1, len(datasets), figsize=(8*len(datasets), 7))
    
    if len(datasets) == 1:
        axes = [axes]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    stages = {
        'select_noun': 'Noun Selection',
        'knowledge_base': 'Knowledge Base',
        'combine_actions': 'Action Combination',
        'score_actions': 'Action Scoring'
    }
    
    dataset_names = {
        'ek100': 'EK100',
        'egtea': 'EGTEA',
        'gtea': 'GTEA'
    }
    
    exp_type_name = "Real-Time" if experiment_type == 'realtime' else "Efficiency"
    
    for idx, dataset in enumerate(datasets):
        data = load_timing_summary(dataset, experiment_type)
        if data is None:
            continue
        
        # Get the data according to the experiment type
        if experiment_type == 'efficiency':
            percentages = data.get('percentage_of_inference', {})
            labels = []
            sizes = []
            
            for key, label in stages.items():
                if key in percentages:
                    labels.append(label)
                    sizes.append(percentages[key])
        else:
            avg_times = data.get('average_times', {})
            total_inference = avg_times.get('total_inference', 0)
            
            labels = []
            sizes = []
            
            if total_inference > 0:
                for key, label in stages.items():
                    if key in avg_times:
                        labels.append(label)
                        percentage = (avg_times[key] / total_inference) * 100
                        sizes.append(percentage)
        
        if not sizes:
            continue
        
        ax = axes[idx]
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.85,
            textprops={'fontsize': 10, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'width': 0.7}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
        
        dataset_name = dataset_names.get(dataset, dataset.upper())
        ax.set_title(f'{dataset_name}', fontsize=14, fontweight='bold', pad=15)
        ax.axis('equal')
    
    plt.suptitle(f'Multi-Dataset {exp_type_name} Inference Time Distribution Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Comparison chart saved to: {save_path}")
    else:
        exp_suffix = "_realtime" if experiment_type == 'realtime' else "_efficiency"
        default_path = f"efficiency_comparison{exp_suffix}_pie_charts.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Comparison chart saved to: {default_path}")
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Efficiency experiment visualization - pie chart")
    parser.add_argument('--dataset', type=str, default='egtea', 
                       choices=['ek100', 'egtea', 'gtea'],
                       help="Name of the dataset (default: ek100)")
    parser.add_argument('--experiment-type', type=str, default='efficiency',
                       choices=['efficiency', 'realtime'],
                       help="Experiment type: efficiency or realtime (default: efficiency)")
    parser.add_argument('--comparison', action='store_true',
                       help="Generate a comparison chart for multiple datasets")
    parser.add_argument('--output', type=str, default=None,
                       help="Path of the output file (generated automatically by default)")
    parser.add_argument('--no-percentage', action='store_true',
                       help="Do not show the percentages in the chart")
    
    args = parser.parse_args()
    
    if args.comparison:
        # Generate the comparison chart
        create_comparison_chart(
            datasets=['ek100', 'egtea', 'gtea'],
            experiment_type=args.experiment_type,
            save_path=args.output
        )
    else:
        # Generate the pie chart of a single dataset
        create_pie_chart(
            dataset=args.dataset,
            experiment_type=args.experiment_type,
            save_path=args.output,
            show_percentage=not args.no_percentage
        )

if __name__ == '__main__':
    main()
