"""
GTEA real-time verification experiment - compute the real-time factor
Analyze the real-time performance of the test videos, including:
- Average inference time per video
- Original video duration (15 fps)
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
import cv2
import tempfile
from typing import Dict, List

# Add the parent directory to sys.path so the modules can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils import *
from action_pipeline import action_recognition_with_timing

# ==================== GTEA Label Parsing ====================
def parse_gtea_label_file(label_file_path):
    """Parse a single GTEA label file."""
    data_list = []
    video_name = os.path.basename(label_file_path).replace('.txt', '')
    
    with open(label_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Extract the action part: the text before the parentheses
        if ' (' in line:
            action_part = line.split(' (')[0].strip()
        else:
            action_part = line.split('(')[0].strip()
        
        # Check whether the '><' connector is present (verb and noun are separated)
        if '><' not in action_part:
            # Only <something> without '><' means a single-object annotation such as <bread>; skip it
            continue
        
        # Extract the time part
        time_part = line.split('(')[1].split(')')[0].strip()
        
        # Parse the verb and the noun
        parts = action_part.split('><')
        verb = parts[0].replace('<', '').strip()
        noun = parts[1].replace('>', '').strip()
        
        start_frame, stop_frame = map(int, time_part.split('-'))
        
        data_list.append({
            'video_id': video_name,
            'start_frame': start_frame,
            'stop_frame': stop_frame,
            'verb': verb,
            'noun': noun,
            'action': f"{verb}_{noun}" if noun else verb
        })
    
    return data_list

def parse_all_gtea_labels(labels_dir):
    """Parse all label files in the GTEA directory."""
    all_data = []
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    print(f"Found {len(label_files)} GTEA label files")
    
    for filename in label_files:
        filepath = os.path.join(labels_dir, filename)
        data = parse_gtea_label_file(filepath)
        all_data.extend(data)
    
    print(f"Parsed {len(all_data)} samples from {labels_dir}")
    return all_data

# ==================== Video Processing ====================
def get_video_path(video_id, videos_base_path):
    """Locate the video file path."""
    for ext in ['.mp4', '.avi', '.mov']:
        video_path = os.path.join(videos_base_path, video_id + ext)
        if os.path.exists(video_path):
            return video_path
    return None

def extract_frames_from_video(video_path, start_frame, stop_frame, num_frames=16):
    """Extract frames in the given range from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    
    # Make sure the range is valid
    start_frame = max(0, start_frame)
    stop_frame = min(total_frames - 1, stop_frame)
    
    if start_frame >= stop_frame:
        cap.release()
        return []
    
    indices = np.linspace(start_frame, stop_frame, num=num_frames, dtype=int)
    temp_files = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            cv2.imwrite(temp_file.name, frame)
            temp_files.append(temp_file.name)
    
    cap.release()
    return temp_files

def cleanup_temp_files(temp_files):
    """Remove the temporary files."""
    for f in temp_files:
        if os.path.exists(f):
            os.unlink(f)

def load_local_playbook(playbook_path):
    """Load the playbook."""
    if os.path.exists(playbook_path):
        try:
            with open(playbook_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {'one': {}, 'two': {}, 'three': {}}

def save_result_locally(output_path, data, predicted_action_dict, selected_noun_keys, timing_info, video_duration, real_time_factor):
    """Append the result and the timing information to the output file."""
    narration_id = f"{data['video_id']}_{data['start_frame']}_{data['stop_frame']}"
    
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
        
        f.write(f"{narration_id},{verbs},{nouns},{actions_str},{confidences},{selected_noun_keys_str},{timing_str}\n")

def process_all_sequentially_realtime_gtea(
    labels_dir,
    videos_base_path,
    hos_path,
    output_txt_path,
    timing_csv_path,
    playbook_path='gtea_playbook.json',
    max_strategies=50,
    merge_threshold=200,
    batch_size=20,
    use_playbook=True,
    base=0,
    use_hos=0,
    top_k=3,
    max_videos=100
):
    """
    Process all GTEA videos and record the detailed inference time and the real-time factor.
    """
    start_time = time.time()
    
    # 1. Parse all the labels
    data_list = parse_all_gtea_labels(labels_dir)
    if not data_list:
        print("No data to process, exit")
        return
    
    # Limit the number of processed videos
    data_list = data_list[:max_videos]
    print(f"The first {len(data_list)} videos will be processed for the real-time verification")
    
    # 2. Initialize the output file (with a header row)
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("narration_id,verbs,nouns,actions,confidences,selected_noun_keys,select_noun_time,knowledge_base_time,combine_actions_time,score_actions_time,total_inference_time,video_duration,real_time_factor\n")

    # Collect the timing data of every video
    all_timing_data = []

    # 3. Initialize the playbook
    if use_playbook and (not os.path.exists(playbook_path) or os.path.getsize(playbook_path) < 50):
        save_local_playbook(playbook_path, {'one': {}, 'two': {}, 'three': {}})
    
    playbook_cache = load_local_playbook(playbook_path) if use_playbook else None
    pending_updates = []
    
    # The GTEA frame rate is 15 FPS
    FRAME_RATE = 15.0
    
    for idx, data in enumerate(data_list):
        video_start_time = time.time()
        narration_id = f"{data['video_id']}_{data['start_frame']}_{data['stop_frame']}"
        
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f"\nProgress: {idx+1}/{len(data_list)} | Elapsed: {elapsed:.1f}s | Current video: {narration_id}")
        
        # Locate the video and extract the frames
        video_path = get_video_path(data['video_id'], videos_base_path)
        if not video_path:
            print(f"Skip {narration_id}: video not found")
            continue
        
        frame_paths = extract_frames_from_video(
            video_path, data['start_frame'], data['stop_frame'], num_frames=16
        )
        
        if not frame_paths:
            print(f"Skip {narration_id}: unable to extract frames")
            continue
        
        hos_path_image = [os.path.join(hos_path, data['video_id'], f"{data['video_id']}_{data['start_frame']}_{data['stop_frame']}_vis_hos_{i:03}.jpg") for i in range(16)]
        if use_hos == 0:
            frames_urls = frame_paths + hos_path_image
        else:
            frames_urls = frame_paths
        
        print(f"  Frames read: {len(frames_urls)}, {narration_id}")
        
        # Compute the video duration (from the frame count and the frame rate)
        frame_count = data['stop_frame'] - data['start_frame']
        video_duration = frame_count / FRAME_RATE
        
        print(f"  Video duration: {video_duration:.2f}s (from frame count, {frame_count} frames, {FRAME_RATE} fps)")
        
        # Read the top strategies
        top_strategies = {'one': [], 'two': [], 'three': []}
        
        if use_playbook and playbook_cache:
            for k in ['one', 'two', 'three']:
                strategies = list(playbook_cache[k].values())
                sorted_strategies = sorted(
                    strategies,
                    key=lambda x: int(x.get('frequency', 0)),
                    reverse=True
                )
                
                for s in sorted_strategies[:top_k]:
                    top_strategies[k].append({
                        'category': s.get('category', 'N/A'),
                        'content': s.get('content', 'N/A'),
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
                'select_noun_time': timing_info['select_noun_time'],
                'knowledge_base_time': timing_info['knowledge_base_time'],
                'combine_actions_time': timing_info['combine_actions_time'],
                'score_actions_time': timing_info['score_actions_time'],
                'total_inference_time': total_inference_time,
                'video_duration': video_duration,
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
        finally:
            cleanup_temp_files(frame_paths)
        
        # Update the playbook
        if use_playbook and reflect_dict:
            pending_updates.append((f"id-{narration_id}", reflect_dict))
            if len(pending_updates) >= batch_size:
                update_strategies_batch(playbook_path, pending_updates, max_strategies, merge_threshold)
                pending_updates.clear()
                playbook_cache = load_local_playbook(playbook_path)

    # Process the remaining updates
    if pending_updates and use_playbook:
        update_strategies_batch(playbook_path, pending_updates, max_strategies, merge_threshold)
    
    # Compute and print the statistics
    print("\n" + "="*80)
    print("GTEA real-time verification statistics")
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
            'dataset': 'GTEA',
            'num_videos': len(all_timing_data),
            'frame_rate': FRAME_RATE,
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
    parser = argparse.ArgumentParser(description="GTEA real-time verification experiment - compute the real-time factor")
    parser.add_argument('--labels_dir', type=str, default='/mnt/data/xgl/mydata/EGTEA++/EGTEA/GTEA_Labels_71/labels')
    parser.add_argument('--videos_base_path', type=str, default='/mnt/data/xgl/mydata/EGTEA++/EGTEA/Videos')
    parser.add_argument('--hos_path', type=str, default='/mnt/data/xgl/mydata/EGTEA++/gtea_image')
    parser.add_argument('--max_videos', type=int, default=100, help="Number of videos to process (default: 100)")
    parser.add_argument('--max_strategies', type=int, default=50)
    parser.add_argument('--merge_threshold', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--base', type=int, default=0, help="0: main model, 1: 1llm, 2: 2llm, 3: 3llm")
    parser.add_argument('--use_playbook', action='store_true', help="Use the playbook")
    parser.add_argument('--use_hos', type=int, default=0, help="0: use HOS, 1: do not use HOS")
    parser.add_argument('--top_k', type=int, default=3, help="Number of top strategies taken from the playbook for each level")
    args = parser.parse_args()

    output_txt_path = 'efficiency/gtea_realtime/gtea_realtime_results.txt'
    timing_csv_path = 'efficiency/gtea_realtime/gtea_realtime_timing.csv'
    playbook_path = 'efficiency/gtea_realtime/gtea_playbook.json'

    print(f"Starting the GTEA real-time verification experiment...")
    print(f"{args.max_videos} videos will be processed")
    print(f"Output file: {output_txt_path}")
    print(f"Timing data: {timing_csv_path}\n")

    process_all_sequentially_realtime_gtea(
        args.labels_dir, args.videos_base_path, args.hos_path,
        output_txt_path, timing_csv_path, playbook_path,
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
