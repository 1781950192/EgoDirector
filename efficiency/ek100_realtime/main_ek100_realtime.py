"""
EK100 real-time verification experiment - compute the real-time factor
Analyze the real-time performance of the test videos, including:
- Average inference time per video
- Original video duration (60 fps)
- Real-time factor = inference time / video duration
- A real-time factor < 1.0 means the real-time requirement is met
"""

import os
import sys
import json
import time
import numpy as np
import argparse
import pandas as pd
from typing import Dict, List
from scipy.spatial.distance import cosine

# Add the parent directory to sys.path so the modules can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils import *
from action_pipeline import action_recognition_with_timing

def load_local_playbook(playbook_path):
    """Load the playbook."""
    if os.path.exists(playbook_path):
        try:
            with open(playbook_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {'one': {}, 'two': {}, 'three': {}}

def parse_timestamp(ts):
    """Parse a timestamp in the 'HH:MM:SS.ms' format into seconds."""
    parts = ts.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def save_result_locally(output_path, data, predicted_action_dict, selected_noun_keys, timing_info, video_duration, real_time_factor):
    """Append the result and the timing information to the output file."""
    with open(output_path, 'a', encoding='utf-8') as f:
        actions = predicted_action_dict.get("actions", [])
        sorted_actions = sorted(actions, key=lambda x: x.get("confidence", 0), reverse=True)
        verbs = ";".join([action.get("verb", "").strip() for action in sorted_actions]) or "no verb"
        nouns = ";".join([action.get("noun", "").strip() for action in sorted_actions]) or "no noun"
        actions_str = ";".join([action.get("action", "").strip() for action in sorted_actions]) or "no action"
        confidences = ";".join([str(action.get("confidence", 0)) for action in sorted_actions]) or "0.0"

        selected_noun_keys_str = (";".join([str(key) for key in selected_noun_keys]) if isinstance(selected_noun_keys, list)
                                  else str(selected_noun_keys) if selected_noun_keys else "no selected noun")

        # Add the timing information and the real-time factor
        timing_str = f"{timing_info['select_noun_time']:.4f},{timing_info['knowledge_base_time']:.4f},{timing_info['combine_actions_time']:.4f},{timing_info['score_actions_time']:.4f},{timing_info['total_time']:.4f},{video_duration:.4f},{real_time_factor:.4f}"
        
        f.write(f"{data['narration_id']},{verbs},{nouns},{actions_str},{confidences},{selected_noun_keys_str},{timing_str}\n")

def process_all_sequentially_realtime(
    csv_file_path,
    output_txt_path,
    timing_csv_path,
    playbook_path='ek100_playbook_8B_average.json',
    max_strategies=50,
    merge_threshold=200,
    batch_size=20,
    use_playbook=True,
    base=3,
    use_hos=False,
    top_k=3,
    max_videos=100
):
    """
    Process all videos and record the detailed inference time and the real-time factor.
    """
    start_time = time.time()
    df = pd.read_csv(csv_file_path, on_bad_lines='skip')
    data_list = df.to_dict('records')
    print(f"{len(data_list)} samples in total, the first {max_videos} videos will be processed for the real-time verification")
    
    # Limit the number of processed videos
    data_list = data_list[:max_videos]

    # Initialize the output file (with a header row)
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("narration_id,verbs,nouns,actions,confidences,selected_noun_keys,select_noun_time,knowledge_base_time,combine_actions_time,score_actions_time,total_inference_time,video_duration,real_time_factor\n")

    # Collect the timing data of every video
    all_timing_data = []

    for idx, data in enumerate(data_list):
        video_start_time = time.time()
        narration_id = data['narration_id']
        
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f"\nProgress: {idx+1}/{len(data_list)} | Elapsed: {elapsed:.1f}s | Current video: {narration_id}")

        # Prepare the data
        base_path = os.path.join('/mnt/data/xgl/mydata/ek100', data['participant_id'], 'rgb_frames', data['video_id'])
        base_path_hos = os.path.join('/mnt/data/xgl/mydata/ek100_hos/average', data['participant_id'], 'rgb_frames', data['video_id'])

        if not os.path.exists(base_path):
            print(f"Path does not exist: {base_path}, skipping")
            continue

        indices = np.linspace(data['start_frame'], data['stop_frame'] - 1, num=16).astype(int)
        frame_paths = [os.path.join(base_path, f"frame_{idx:010d}.jpg") for idx in indices]
        image_box = [os.path.join(base_path_hos, f"frame_{idx:010d}.jpg") for idx in indices]
        frames_urls = [p for p in frame_paths if os.path.exists(p)]
        images_box = [p for p in image_box if os.path.exists(p)]

        if use_hos == 0:
            frames_urls = frames_urls + images_box
        
        print(f"  Input {len(frames_urls)} images")

        # Compute the video duration
        start_time_str = data['start_timestamp']
        stop_time_str = data['stop_timestamp']
        
        start_seconds = parse_timestamp(start_time_str)
        stop_seconds = parse_timestamp(stop_time_str)
        video_duration = stop_seconds - start_seconds
        
        # It can also be computed from the frame count (fallback, 60 fps)
        frame_count = data['stop_frame'] - data['start_frame']
        video_duration_from_frames = frame_count / 60.0
        
        print(f"  Video duration: {video_duration:.2f}s (from timestamps) | {video_duration_from_frames:.2f}s (from frame count, {frame_count} frames)")

        # Read the top strategies
        top_strategies = {'one': [], 'two': [], 'three': []}
        if use_playbook:
            playbook_cache = load_local_playbook(playbook_path)
            for k in ['one', 'two', 'three']:
                items = sorted(playbook_cache[k].items(), key=lambda x: int(x[1]['frequency']), reverse=True)[:top_k]
                for sid, s in items:
                    top_strategies[k].append({
                        'category': s['category'],
                        'content': s['content'],
                        'frame_strategy': s.get('frame_strategy', 'N/A')
                    })

        # Call the action recognition function with timing
        try:
            reflect_dict, predicted_action_dict, selected_noun_keys, timing_info = action_recognition_with_timing(
                frames_urls)
            
            # Compute the total inference time
            total_inference_time = time.time() - video_start_time
            timing_info['total_time'] = total_inference_time
            
            # Compute the real-time factor
            real_time_factor = total_inference_time / video_duration if video_duration > 0 else float('inf')
            realtime_status = "✓ real-time" if real_time_factor < 1.0 else "✗ not real-time"
            
            # Save the result
            save_result_locally(output_txt_path, data, predicted_action_dict, selected_noun_keys, timing_info, video_duration, real_time_factor)
            
            # Collect the timing data for the statistical analysis
            timing_record = {
                'video_id': narration_id,
                'participant_id': data['participant_id'],
                'video_id_raw': data['video_id'],
                'select_noun_time': timing_info['select_noun_time'],
                'knowledge_base_time': timing_info['knowledge_base_time'],
                'combine_actions_time': timing_info['combine_actions_time'],
                'score_actions_time': timing_info['score_actions_time'],
                'total_inference_time': total_inference_time,
                'video_duration': video_duration,
                'video_duration_from_frames': video_duration_from_frames,
                'frame_count': frame_count,
                'real_time_factor': real_time_factor,
                'is_realtime': real_time_factor < 1.0
            }
            all_timing_data.append(timing_record)
            
            print(f"  ✓ Done | Inference: {total_inference_time:.2f}s | "
                  f"Video duration: {video_duration:.2f}s | "
                  f"Real-time factor: {real_time_factor:.3f} | "
                  f"{realtime_status}")
            
        except Exception as e:
            print(f"  × Processing failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compute and print the statistics
    print("\n" + "="*80)
    print("EK100 real-time verification statistics")
    print("="*80)
    
    if all_timing_data:
        df_timing = pd.DataFrame(all_timing_data)
        
        # Compute the average values
        avg_select_noun = df_timing['select_noun_time'].mean()
        avg_knowledge_base = df_timing['knowledge_base_time'].mean()
        avg_combine_actions = df_timing['combine_actions_time'].mean()
        avg_score_actions = df_timing['score_actions_time'].mean()
        avg_total_inference = df_timing['total_inference_time'].mean()
        avg_video_duration = df_timing['video_duration'].mean()
        avg_real_time_factor = df_timing['real_time_factor'].mean()
        
        # Compute the standard deviations
        std_select_noun = df_timing['select_noun_time'].std()
        std_knowledge_base = df_timing['knowledge_base_time'].std()
        std_combine_actions = df_timing['combine_actions_time'].std()
        std_score_actions = df_timing['score_actions_time'].std()
        std_total_inference = df_timing['total_inference_time'].std()
        std_video_duration = df_timing['video_duration'].std()
        std_real_time_factor = df_timing['real_time_factor'].std()
        
        # Compute the real-time statistics
        realtime_videos = (df_timing['real_time_factor'] < 1.0).sum()
        non_realtime_videos = len(df_timing) - realtime_videos
        realtime_percentage = (realtime_videos / len(df_timing)) * 100
        
        print(f"\nNumber of processed videos: {len(all_timing_data)}")
        print(f"\n{'Stage':<20} {'Avg time(s)':<15} {'Std dev(s)':<15} {'Ratio(%)':<10}")
        print("-" * 80)
        print(f"{'Select noun':<20} {avg_select_noun:<15.4f} {std_select_noun:<15.4f} {(avg_select_noun/avg_total_inference*100):<10.2f}")
        print(f"{'KB retrieval/update':<20} {avg_knowledge_base:<15.4f} {std_knowledge_base:<15.4f} {(avg_knowledge_base/avg_total_inference*100):<10.2f}")
        print(f"{'Combine actions':<20} {avg_combine_actions:<15.4f} {std_combine_actions:<15.4f} {(avg_combine_actions/avg_total_inference*100):<10.2f}")
        print(f"{'Score actions':<20} {avg_score_actions:<15.4f} {std_score_actions:<15.4f} {(avg_score_actions/avg_total_inference*100):<10.2f}")
        print("-" * 80)
        print(f"{'Total inference':<20} {avg_total_inference:<15.4f} {std_total_inference:<15.4f} {'100.00':<10}")
        
        print(f"\n{'='*80}")
        print("Real-time analysis")
        print("="*80)
        print(f"Average video duration: {avg_video_duration:.4f}s (±{std_video_duration:.4f}s)")
        print(f"Average inference time: {avg_total_inference:.4f}s (±{std_total_inference:.4f}s)")
        print(f"Average real-time factor: {avg_real_time_factor:.4f} (±{std_real_time_factor:.4f})")
        print("-" * 80)
        print(f"Real-time requirement met (<1.0): {realtime_videos} videos ({realtime_percentage:.2f}%)")
        print(f"Real-time requirement not met: {non_realtime_videos} videos ({100-realtime_percentage:.2f}%)")
        
        # Show the distribution of the real-time factor
        rtf_min = df_timing['real_time_factor'].min()
        rtf_max = df_timing['real_time_factor'].max()
        rtf_median = df_timing['real_time_factor'].median()
        rtf_q1 = df_timing['real_time_factor'].quantile(0.25)
        rtf_q3 = df_timing['real_time_factor'].quantile(0.75)
        print("-" * 80)
        print(f"Real-time factor range: [{rtf_min:.4f}, {rtf_max:.4f}]")
        print(f"Real-time factor median: {rtf_median:.4f}")
        print(f"Real-time factor IQR: [{rtf_q1:.4f}, {rtf_q3:.4f}]")
        
        # Find the most and the least real-time videos
        most_realtime = df_timing.loc[df_timing['real_time_factor'].idxmin()]
        least_realtime = df_timing.loc[df_timing['real_time_factor'].idxmax()]
        print("-" * 80)
        print(f"Most real-time video: {most_realtime['video_id']} (factor={most_realtime['real_time_factor']:.4f}, duration={most_realtime['video_duration']:.2f}s)")
        print(f"Least real-time video: {least_realtime['video_id']} (factor={least_realtime['real_time_factor']:.4f}, duration={least_realtime['video_duration']:.2f}s)")
        
        # Save the detailed CSV statistics file
        df_timing.to_csv(timing_csv_path, index=False, encoding='utf-8-sig')
        print(f"\nDetailed timing data saved to: {timing_csv_path}")
        
        # Save the statistics summary in JSON format
        summary = {
            'dataset': 'EK100',
            'num_videos': len(all_timing_data),
            'frame_rate': 60,
            'average_times': {
                'select_noun': round(avg_select_noun, 4),
                'knowledge_base': round(avg_knowledge_base, 4),
                'combine_actions': round(avg_combine_actions, 4),
                'score_actions': round(avg_score_actions, 4),
                'total_inference': round(avg_total_inference, 4),
                'video_duration': round(avg_video_duration, 4),
                'real_time_factor': round(avg_real_time_factor, 4)
            },
            'std_times': {
                'select_noun': round(std_select_noun, 4),
                'knowledge_base': round(std_knowledge_base, 4),
                'combine_actions': round(std_combine_actions, 4),
                'score_actions': round(std_score_actions, 4),
                'total_inference': round(std_total_inference, 4),
                'video_duration': round(std_video_duration, 4),
                'real_time_factor': round(std_real_time_factor, 4)
            },
            'realtime_analysis': {
                'realtime_videos': int(realtime_videos),
                'non_realtime_videos': int(non_realtime_videos),
                'realtime_percentage': round(realtime_percentage, 2),
                'real_time_factor_range': {
                    'min': round(rtf_min, 4),
                    'max': round(rtf_max, 4),
                    'median': round(rtf_median, 4),
                    'q1': round(rtf_q1, 4),
                    'q3': round(rtf_q3, 4)
                },
                'most_realtime_video': {
                    'video_id': most_realtime['video_id'],
                    'real_time_factor': round(most_realtime['real_time_factor'], 4),
                    'video_duration': round(most_realtime['video_duration'], 4)
                },
                'least_realtime_video': {
                    'video_id': least_realtime['video_id'],
                    'real_time_factor': round(least_realtime['real_time_factor'], 4),
                    'video_duration': round(least_realtime['video_duration'], 4)
                }
            }
        }
        
        summary_path = timing_csv_path.replace('.csv', '_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Statistics summary saved to: {summary_path}")
    
    print(f"\nTotal time: {(time.time() - start_time):.2f} s")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="EK100 real-time verification experiment - compute the real-time factor")
    parser.add_argument('--max_videos', type=int, default=100, help="Number of videos to process (default: 100)")
    parser.add_argument('--max_strategies', type=int, default=50)
    parser.add_argument('--merge_threshold', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--base', type=int, default=3, help="0: main model, 1: 1llm, 2: 2llm, 3: 3llm")
    parser.add_argument('--use_playbook', action='store_true', help="Use the playbook")
    parser.add_argument('--use_hos', type=int, default=0, help="0: use HOS, 1: do not use HOS")
    parser.add_argument('--top_k', type=int, default=3, help="Number of top strategies taken from the playbook for each level")
    args = parser.parse_args()

    csv_file_path = '/mnt/data/xgl/mydata/ek100/EPIC_100_validation.csv'
    output_txt_path = 'efficiency/ek100_realtime/ek100_realtime_results.txt'
    timing_csv_path = 'efficiency/ek100_realtime/ek100_realtime_timing.csv'
    playbook_path = 'output/ek100/ek100_playbook_8B.json'

    print(f"Starting the EK100 real-time verification experiment...")
    print(f"{args.max_videos} videos will be processed")
    print(f"Output file: {output_txt_path}")
    print(f"Timing data: {timing_csv_path}\n")

    process_all_sequentially_realtime(
        csv_file_path, output_txt_path, timing_csv_path, playbook_path,
        args.max_strategies, args.merge_threshold, args.batch_size,
        args.use_playbook, args.base, args.use_hos,
        top_k=args.top_k,
        max_videos=args.max_videos
    )
    overlay_context_base()

if __name__ == '__main__':
    import faulthandler
    faulthandler.enable()
    main()
