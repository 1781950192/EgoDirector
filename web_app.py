"""
Action Recognition Web Visualization System
Flask-based web application with remote access support
"""

import os
import sys
import json
import time
import base64
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

# Project imports
from utils import *
from action_pipeline import (
    action_recognition, action_recognition_base1,
    action_recognition_base2, action_recognition_base3,
    select_nouns, combine_actions, select_actions,
)
from context_manager import extract_noun_probabilities

# ====================== Flask Application Initialization ======================
app = Flask(__name__, static_folder='web/static', template_folder='web/templates')
CORS(app)  # Enable cross-origin access

# ====================== Global Configuration ======================
class Config:
    """Global configuration."""
    # Dataset selection: 'ek100' or 'gtea'
    DATASET = 'ek100'  # Change this to switch datasets
    
    # EK100 configuration
    EK100_CSV_FILE_PATH = '/mnt/data/xgl/mydata/ek100/EPIC_100_validation.csv'
    EK100_BASE_PATH_ROOT = '/mnt/data/xgl/mydata/ek100'
    EK100_HOS_PATH_ROOT = '/mnt/data/xgl/mydata/ek100_hos/average'
    
    # GTEA configuration
    GTEA_LABELS_DIR = '/mnt/data/xgl/mydata/EGTEA++/EGTEA/GTEA_Labels_71/labels'
    GTEA_VIDEOS_PATH = '/mnt/data/xgl/mydata/EGTEA++/EGTEA/Videos'
    
    # Output configuration
    OUTPUT_TXT_PATH = 'ek100_val_8B_web_results.txt'
    PLAYBOOK_PATH = 'ek100_playbook_8B.json'
    
    # Recognition parameters
    USE_PLAYBOOK = True
    USE_HOS = 1  # 0=use HOS, 1=do not use HOS (EK100 only)
    TOP_K = 3
    MAX_STRATEGIES = 50
    MERGE_THRESHOLD = 200
    BATCH_SIZE = 20

config = Config()

# ====================== Global State ======================
class AppState:
    """Global application state."""
    def __init__(self):
        self.current_idx = 0
        self.data_list = []
        self.correct_count = 0
        self.incorrect_count = 0
        self.results_log = []
        self.playbook_cache = None
        self.initialized = False
        self.dataset = config.DATASET
    
    def initialize(self):
        """Initialize the application state."""
        if self.initialized:
            return
        
        # Load the dataset selected in the configuration
        if self.dataset == 'gtea':
            print(f"Loading GTEA dataset: {config.GTEA_LABELS_DIR}")
            self.data_list = self._load_gtea_data()
            self.output_path = 'gtea_web_results.txt'
        else:  # ek100
            print(f"Loading EK100 dataset: {config.EK100_CSV_FILE_PATH}")
            df = pd.read_csv(config.EK100_CSV_FILE_PATH, on_bad_lines='skip')
            self.data_list = df.to_dict('records')
            self.output_path = config.OUTPUT_TXT_PATH
        
        print(f"Loaded {len(self.data_list)} samples")
        
        # Initialize the playbook
        if config.USE_PLAYBOOK:
            self.playbook_cache = self._load_playbook()
        
        # Initialize the output file
        if not os.path.exists(self.output_path):
            with open(self.output_path, 'w', encoding='utf-8') as f:
                f.write("narration_id,verbs,nouns,actions,confidences\n")
        
        self.initialized = True
    
    def _load_gtea_data(self):
        """Load the GTEA dataset."""
        import os
        data_list = []
        label_files = [f for f in os.listdir(config.GTEA_LABELS_DIR) if f.endswith('.txt')]
        
        print(f"Found {len(label_files)} GTEA label files")
        
        for label_file in label_files:
            label_path = os.path.join(config.GTEA_LABELS_DIR, label_file)
            video_name = os.path.basename(label_file).replace('.txt', '')
            
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Extract the action part
                    if ' (' in line:
                        action_part = line.split(' (')[0].strip()
                    else:
                        action_part = line.split('(')[0].strip()
                    
                    # Only handle the verb+noun format
                    if '><' not in action_part:
                        continue
                    
                    time_part = line.split('(')[1].split(')')[0].strip()
                    parts = action_part.split('><')
                    
                    verb = parts[0].replace('<', '').strip()
                    noun = parts[1].replace('>', '').strip()
                    start_frame, end_frame = map(int, time_part.split('-'))
                    
                    narration_id = f"{video_name}_{start_frame}_{end_frame}"
                    action = f"{verb}_{noun}" if noun else verb
                    
                    data_list.append({
                        'narration_id': narration_id,
                        'video_id': video_name,
                        'start_frame': start_frame,
                        'stop_frame': end_frame,
                        'verb': verb,
                        'noun': noun,
                        'action': action,
                        'narration': f"{verb} {noun}",  # Add the narration field
                        'participant_id': 'P01'  # GTEA has no participant_id, use a default value
                    })
                
            except Exception as e:
                print(f"Failed to load file {label_file}: {e}")
        
        print(f"Parsed {len(data_list)} GTEA action instances in total")
        return data_list
    
    def _load_playbook(self):
        """Load the playbook."""
        if os.path.exists(config.PLAYBOOK_PATH):
            try:
                with open(config.PLAYBOOK_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {'one': {}, 'two': {}, 'three': {}}
    
    def get_top_strategies(self):
        """Get the top-k strategies."""
        if not self.playbook_cache:
            return {'one': [], 'two': [], 'three': []}
        
        top_strategies = {'one': [], 'two': [], 'three': []}
        for k in ['one', 'two', 'three']:
            items = sorted(self.playbook_cache[k].items(), 
                          key=lambda x: int(x[1]['frequency']), 
                          reverse=True)[:config.TOP_K]
            for sid, s in items:
                top_strategies[k].append({
                    'category': s['category'],
                    'content': s['content'],
                    'frame_strategy': s.get('frame_strategy', 'N/A')
                })
        return top_strategies
    
    def update_playbook(self, narration_id, reflect_dict):
        """Update the playbook."""
        if not reflect_dict:
            return
        
        for level in ['one', 'two', 'three']:
            key = f"correct_approach_{level}"
            cat_key = f"category_{level}"
            frame_key = f"frame_strategy_{level}"
            
            self.playbook_cache[level][f"id-{narration_id}"] = {
                "category": reflect_dict.get(cat_key, "N/A"),
                "content": reflect_dict.get(key, "N/A"),
                "frame_strategy": reflect_dict.get(frame_key, "N/A"),
                "frequency": "1"
            }
    
    def save_playbook(self):
        """Save the playbook."""
        with open(config.PLAYBOOK_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.playbook_cache, f, ensure_ascii=False, indent=2)

# Global state instance
app_state = AppState()

# ====================== Helper Functions ======================
def image_to_base64(file_path: str) -> str:
    """Convert an image to a base64 string."""
    try:
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        return None

def get_frame_paths(data, num_frames=16):
    """Get frame paths (supports EK100 and GTEA)."""
    if config.DATASET == 'gtea':
        # GTEA dataset: extract frames from the video
        video_path = os.path.join(config.GTEA_VIDEOS_PATH, data['video_id'] + '.mp4')
        
        if not os.path.exists(video_path):
            print(f"Video does not exist: {video_path}")
            return [], []
        
        # Extract frames with cv2
        import cv2
        import tempfile
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stop_frame = min(int(data['stop_frame']), total_frames)
        start_frame = int(data['start_frame'])
        
        # Compute the frame indices to extract
        indices = np.linspace(start_frame, stop_frame - 1, num=num_frames, dtype=int)
        
        frame_paths = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                # Save to a temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                cv2.imwrite(temp_file.name, frame)
                frame_paths.append(temp_file.name)
        
        cap.release()
        return frame_paths, []  # GTEA has no HOS segmentation
    
    else:
        # EK100 dataset
        base_path = os.path.join(
            config.EK100_BASE_PATH_ROOT,
            data['participant_id'],
            'rgb_frames',
            data['video_id']
        )
        hos_path = os.path.join(
            config.EK100_HOS_PATH_ROOT,
            data['participant_id'],
            'rgb_frames',
            data['video_id']
        )
        
        indices = np.linspace(
            int(data['start_frame']),
            int(data['stop_frame']) - 1,
            num=num_frames
        ).astype(int)
        
        frame_paths = []
        hos_paths = []
        
        for idx in indices:
            frame_path = os.path.join(base_path, f"frame_{idx:010d}.jpg")
            hos_path_item = os.path.join(hos_path, f"frame_{idx:010d}.jpg")
            frame_paths.append(frame_path)
            hos_paths.append(hos_path_item)
        
        return frame_paths, hos_paths

def run_action_recognition_pipeline(frames_urls, top_strategies):
    """
    Run the complete three-stage recognition pipeline.
    Agent 1: noun selection
    Agent 2: action selection
    Agent 3: action scoring
    """
    from action_pipeline import select_nouns, combine_actions, select_actions
    from context_manager import extract_noun_probabilities

    results = {}
    context_info = {
        'input_frames': {
            'rgb_count': len([f for f in frames_urls if 'rgb_frames' in f]),
            'hos_count': len([f for f in frames_urls if 'hos' in f]),
            'total': len(frames_urls)
        }
    }

    try:
        # ========== Agent 1: Noun Selection ==========
        print("Agent 1: selecting nouns...")
        noun_keys = []
        i = 0
        selected_noun_result = None

        # Extract the strategy text content
        strategies_one_text = ""
        if top_strategies.get("one"):
            for s in top_strategies["one"]:
                if isinstance(s, dict) and s.get('content'):
                    strategies_one_text += f"- {s.get('category', 'N/A')}: {s.get('content', 'N/A')}\n"
                elif isinstance(s, str):
                    strategies_one_text += f"{s}\n"

        while True:
            selected_noun_result = select_nouns(frames_urls, noun_keys, strategies_one_text)
            if selected_noun_result and "noun" in selected_noun_result:
                noun_keys = selected_noun_result["noun"]
            i += 1
            if len(noun_keys) >= 5 or i >= 5:
                break

        results['agent1'] = {
            'success': True,
            'nouns': noun_keys,
            'result': selected_noun_result
        }
        print(f"Agent 1 selected nouns: {noun_keys}")

        # ========== Semantic Context Section ==========
        print("Loading semantic context...")
        try:
            with open('context/pre_noun.json', 'r', encoding='utf-8') as f:
                pre_noun = json.load(f)

            # Build the pre_nouns dictionary
            key_to_text = {item["key"]: item["generated_text"] for item in pre_noun}
            pre_nouns = {}
            for noun in noun_keys:
                pre_nouns[noun] = key_to_text.get(noun)
            
            # Store the semantic context in the results
            results['semantic_context'] = {
                'success': True,
                'pre_nouns': pre_nouns,
                'raw_data': pre_noun
            }
            print(f"Loaded semantic context for {len(pre_nouns)} nouns")
        except Exception as e:
            print(f"Failed to load semantic context: {e}")
            results['semantic_context'] = {
                'success': False,
                'error': str(e),
                'pre_nouns': {}
            }

        # ========== Agent 2: Action Selection ==========
        print("Agent 2: combining actions...")

        # Get the verb probabilities of the nouns
        keys = [item for item in noun_keys]
        noun_verb = extract_noun_probabilities(keys)

        # Extract the strategy text content
        strategies_two_text = ""
        if top_strategies.get("two"):
            for s in top_strategies["two"]:
                if isinstance(s, dict) and s.get('content'):
                    strategies_two_text += f"- {s.get('category', 'N/A')}: {s.get('content', 'N/A')}\n"
                elif isinstance(s, str):
                    strategies_two_text += f"{s}\n"

        # Use llm=2 (2LLM mode)
        actions_result = combine_actions(frames_urls, noun_keys, noun_verb, strategies_two_text, llm=2)

        results['agent2'] = {
            'success': True,
            'actions': actions_result,
            'nouns': noun_keys,
            'noun_verb': noun_verb
        }
        # Save the Agent 2 context
        context_info['agent2'] = {
            'input_nouns': noun_keys,
            'noun_verb_probabilities': noun_verb,
            'strategies_text': strategies_two_text.strip() if strategies_two_text.strip() else 'No strategy',
            'output': actions_result
        }
        print(f"Agent 2 selected actions: {actions_result}")

        # ========== Agent 3: Action Scoring ==========
        print("Agent 3: scoring actions...")

        # Extract the strategy text content
        strategies_three_text = ""
        if top_strategies.get("three"):
            for s in top_strategies["three"]:
                if isinstance(s, dict) and s.get('content'):
                    strategies_three_text += f"- {s.get('category', 'N/A')}: {s.get('content', 'N/A')}\n"
                elif isinstance(s, str):
                    strategies_three_text += f"{s}\n"

        # Use select_actions for scoring
        action_dict_result = select_actions(frames_urls, actions_result, noun_keys, strategies_three_text)

        results['agent3'] = {
            'success': True,
            'action_dict': action_dict_result,
            'final_actions': action_dict_result
        }
        # Save the Agent 3 context
        context_info['agent3'] = {
            'input_actions': actions_result,
            'input_nouns': noun_keys,
            'strategies_text': strategies_three_text.strip() if strategies_three_text.strip() else 'No strategy',
            'output': action_dict_result
        }
        print(f"Agent 3 scored actions: {action_dict_result}")

        # Attach the context information to the result
        results['context'] = context_info

        return results

    except Exception as e:
        print(f"Recognition pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'agent1': {'success': False, 'error': str(e)},
            'agent2': {'success': False, 'error': str(e)},
            'agent3': {'success': False, 'error': str(e)},
            'semantic_context': {'success': False, 'error': str(e)}
        }

def save_result(data, result, agent_name="web"):
    """Save the recognition result."""
    if result is None:
        return
    
    with open(config.OUTPUT_TXT_PATH, 'a', encoding='utf-8') as f:
        # Handle the different result formats
        if isinstance(result, dict) and 'actions' in result:
            actions = result.get("actions", [])
        elif isinstance(result, dict):
            # If it is a dict, try to extract the actions
            actions = []
            for key, value in result.items():
                if isinstance(value, dict):
                    actions.append(value)
        elif isinstance(result, list):
            actions = result
        else:
            actions = []
        
        sorted_actions = sorted(actions, key=lambda x: float(x.get("confidence", 0)), reverse=True) if actions else []
        verbs = ";".join([action.get("verb", "").strip() for action in sorted_actions]) or "no verb"
        nouns = ";".join([action.get("noun", "").strip() for action in sorted_actions]) or "no noun"
        actions_str = ";".join([action.get("action", "").strip() for action in sorted_actions]) or "no action"
        confidences = ";".join([str(action.get("confidence", 0)) for action in sorted_actions]) or "0.0"
        
        f.write(f"{data['narration_id']},{verbs},{nouns},{actions_str},{confidences}\n")

# ====================== Web Routes ======================
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory(app.static_folder, filename)

@app.route('/api/init', methods=['POST'])
def api_init():
    """Initialize the application."""
    try:
        app_state.initialize()
        return jsonify({
            'success': True,
            'message': f'Loaded {len(app_state.data_list)} samples',
            'total_count': len(app_state.data_list)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/data/current', methods=['GET'])
def api_get_current_data():
    """Get the current sample (including raw frames and HOS frames)."""
    if not app_state.initialized:
        return jsonify({'success': False, 'error': 'Not initialized'}), 400
    
    if app_state.current_idx >= len(app_state.data_list):
        return jsonify({
            'success': False,
            'error': 'All samples have been processed',
            'completed': True
        })
    
    data = app_state.data_list[app_state.current_idx]
    frame_paths, hos_paths = get_frame_paths(data)
    
    # Load image base64 (raw frames + HOS frames)
    images = []
    
    # Raw frames
    for i, path in enumerate(frame_paths):
        img_data = image_to_base64(path)
        images.append({
            'index': i + 1,
            'path': path,
            'type': 'rgb',
            'exists': os.path.exists(path),
            'data': img_data
        })
    
    # HOS frames
    for i, path in enumerate(hos_paths):
        img_data = image_to_base64(path)
        images.append({
            'index': i + 1,
            'path': path,
            'type': 'hos',
            'exists': os.path.exists(path),
            'data': img_data
        })
    
    return jsonify({
        'success': True,
        'data': {
            'narration_id': data['narration_id'],
            'participant_id': data['participant_id'],
            'video_id': data['video_id'],
            'narration': data['narration'],
            'verb': data['verb'],
            'verb_class': data['verb_class'],
            'noun': data['noun'],
            'noun_class': data['noun_class'],
            'start_timestamp': data['start_timestamp'],
            'stop_timestamp': data['stop_timestamp'],
            'start_frame': int(data['start_frame']),
            'stop_frame': int(data['stop_frame']),
            'ground_truth': f"{data['verb']} + {data['noun']}"
        },
        'images': images,
        'progress': {
            'current': app_state.current_idx + 1,
            'total': len(app_state.data_list)
        },
        'stats': {
            'correct': app_state.correct_count,
            'incorrect': app_state.incorrect_count
        }
    })

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    """Run recognition - three-stage pipeline."""
    if not app_state.initialized:
        return jsonify({'success': False, 'error': 'Not initialized'}), 400
    
    data = app_state.data_list[app_state.current_idx]
    frame_paths, hos_paths = get_frame_paths(data)
    
    # Prepare the input frames (raw frames + HOS segmentation frames)
    valid_frames = [p for p in frame_paths if os.path.exists(p)]
    valid_hos = [p for p in hos_paths if os.path.exists(p)]
    
    # Merge the two kinds of images
    all_frames = valid_frames + valid_hos
    
    # Get the strategies
    top_strategies = app_state.get_top_strategies()
    
    print(f"\nProcessing {data['narration_id']}...")
    print(f"Input {len(all_frames)} images (raw: {len(valid_frames)}, HOS: {len(valid_hos)})")
    
    # Run the three-stage recognition pipeline
    results = run_action_recognition_pipeline(all_frames, top_strategies)
    
    # Save the result (use the final result of agent 3)
    if results['agent3']['success']:
        save_result(data, results['agent3']['action_dict'])
    
    # Format the result for the frontend
    formatted_results = {}
    
    # Agent 1: noun selection
    if results['agent1']['success']:
        formatted_results['agent1'] = {
            'success': True,
            'title': 'Noun Selection',
            'nouns': results['agent1']['nouns'],
            'raw_result': results['agent1']['result']
        }
    else:
        formatted_results['agent1'] = {
            'success': False,
            'error': results['agent1'].get('error', 'Recognition failed')
        }
    
    # Agent 2: action selection
    if results['agent2']['success']:
        actions = results['agent2']['actions']
        if isinstance(actions, dict) and 'actions' in actions:
            actions_list = actions['actions']
        elif isinstance(actions, list):
            actions_list = actions
        else:
            actions_list = []
        
        formatted_results['agent2'] = {
            'success': True,
            'title': 'Action Selection',
            'actions': actions_list,
            'nouns': results['agent2']['nouns']
        }
    else:
        formatted_results['agent2'] = {
            'success': False,
            'error': results['agent2'].get('error', 'Recognition failed')
        }
    
    # Agent 3: action scoring
    if results['agent3']['success']:
        action_dict = results['agent3']['action_dict']

        # Convert to a list format
        if isinstance(action_dict, dict) and 'actions' in action_dict:
            actions_list = action_dict['actions']
        elif isinstance(action_dict, dict):
            # If it is a scoring dict, convert it to a list
            actions_list = []
            for key, value in action_dict.items():
                if isinstance(value, dict):
                    actions_list.append(value)
        elif isinstance(action_dict, list):
            actions_list = action_dict
        else:
            actions_list = []

        # Sort by confidence
        if actions_list and len(actions_list) > 0 and isinstance(actions_list[0], dict):
            sorted_actions = sorted(actions_list, key=lambda x: float(x.get('confidence', 0)), reverse=True)[:5]
        else:
            sorted_actions = []

        formatted_results['agent3'] = {
            'success': True,
            'title': 'Action Scoring',
            'actions': sorted_actions,
            'raw_result': action_dict
        }
    else:
        formatted_results['agent3'] = {
            'success': False,
            'error': results['agent3'].get('error', 'Recognition failed')
        }

    # Attach context and semantic_context information to the response
    formatted_results['context'] = results.get('context', {})
    formatted_results['semantic_context'] = results.get('semantic_context', {})

    return jsonify({
        'success': True,
        'results': formatted_results
    })

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """Evaluate the recognition result."""
    if not app_state.initialized:
        return jsonify({'success': False, 'error': 'Not initialized'}), 400
    
    data = request.json
    is_correct = data.get('is_correct', False)
    agent_result = data.get('agent_result', {})
    
    current_data = app_state.data_list[app_state.current_idx]
    
    # Get the prediction result
    pred_verb = agent_result.get('actions', [{}])[0].get('verb', 'no result')
    pred_noun = agent_result.get('actions', [{}])[0].get('noun', 'no result')
    
    # Write the log record
    log_entry = {
        'narration_id': current_data['narration_id'],
        'gt_verb': current_data['verb'],
        'gt_noun': current_data['noun'],
        'pred_verb': pred_verb,
        'pred_noun': pred_noun,
        'is_correct': is_correct,
        'timestamp': time.time()
    }
    
    app_state.results_log.append(log_entry)
    
    if is_correct:
        app_state.correct_count += 1
    else:
        app_state.incorrect_count += 1
    
    return jsonify({
        'success': True,
        'stats': {
            'correct': app_state.correct_count,
            'incorrect': app_state.incorrect_count,
            'total': app_state.correct_count + app_state.incorrect_count,
            'accuracy': app_state.correct_count / (app_state.correct_count + app_state.incorrect_count) 
                       if (app_state.correct_count + app_state.incorrect_count) > 0 else 0
        },
        'log_entry': log_entry
    })

@app.route('/api/next', methods=['POST'])
def api_next():
    """Move to the next sample."""
    if not app_state.initialized:
        return jsonify({'success': False, 'error': 'Not initialized'}), 400
    
    app_state.current_idx += 1
    
    if app_state.current_idx >= len(app_state.data_list):
        return jsonify({
            'success': False,
            'completed': True,
            'message': 'All samples have been processed',
            'final_stats': {
                'total': len(app_state.data_list),
                'correct': app_state.correct_count,
                'incorrect': app_state.incorrect_count,
                'accuracy': app_state.correct_count / (app_state.correct_count + app_state.incorrect_count)
                           if (app_state.correct_count + app_state.incorrect_count) > 0 else 0
            }
        })
    
    return jsonify({
        'success': True,
        'progress': {
            'current': app_state.current_idx + 1,
            'total': len(app_state.data_list)
        }
    })

@app.route('/api/stats', methods=['GET'])
def api_get_stats():
    """Get the statistics."""
    if not app_state.initialized:
        return jsonify({'success': False, 'error': 'Not initialized'}), 400
    
    return jsonify({
        'success': True,
        'stats': {
            'total': len(app_state.data_list),
            'processed': app_state.current_idx,
            'correct': app_state.correct_count,
            'incorrect': app_state.incorrect_count,
            'accuracy': app_state.correct_count / (app_state.correct_count + app_state.incorrect_count)
                       if (app_state.correct_count + app_state.incorrect_count) > 0 else 0,
            'results_log': app_state.results_log[-50:]  # Latest 50 records
        }
    })

@app.route('/api/save', methods=['POST'])
def api_save():
    """Save all data."""
    if not app_state.initialized:
        return jsonify({'success': False, 'error': 'Not initialized'}), 400
    
    # Save the statistics
    stats_path = config.OUTPUT_TXT_PATH.replace('.txt', '_stats.json')
    stats = {
        'total': len(app_state.data_list),
        'correct': app_state.correct_count,
        'incorrect': app_state.incorrect_count,
        'accuracy': app_state.correct_count / (app_state.correct_count + app_state.incorrect_count)
                   if (app_state.correct_count + app_state.incorrect_count) > 0 else 0,
        'results_log': app_state.results_log
    }
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # Save the playbook
    if config.USE_PLAYBOOK:
        app_state.save_playbook()
    
    return jsonify({
        'success': True,
        'message': f'Data saved to {stats_path}',
        'stats': stats
    })

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """Get or update the configuration."""
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'config': {
                'use_playbook': config.USE_PLAYBOOK,
                'use_hos': config.USE_HOS,
                'top_k': config.TOP_K,
                'csv_file': config.CSV_FILE_PATH,
                'output_file': config.OUTPUT_TXT_PATH
            }
        })
    else:
        data = request.json
        if 'use_playbook' in data:
            config.USE_PLAYBOOK = data['use_playbook']
        if 'use_hos' in data:
            config.USE_HOS = data['use_hos']
        if 'top_k' in data:
            config.TOP_K = data['top_k']
        return jsonify({'success': True, 'message': 'Configuration updated'})

# ====================== Error Handling ======================
@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Not Found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Server Error'}), 500

# ====================== Main Function ======================
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Action Recognition Web Visualization System")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Listen address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--use_hos', type=int, default=1, help="0: use HOS, 1: do not use HOS")
    parser.add_argument('--top_k', type=int, default=3, help="playbook top-k strategies")
    args = parser.parse_args()
    
    # Update the configuration
    config.USE_HOS = args.use_hos
    config.TOP_K = args.top_k
    
    print("=" * 60)
    print("Action Recognition Web Visualization System")
    print("Action Recognition Web Visualization System")
    print("=" * 60)
    print(f"\nStarting the server...")
    print(f"Listening on http://{args.host}:{args.port}")
    print(f"\nOpen the address above in your browser to access the system")
    print("\nNotes:")
    print("  - If running on a remote server, make sure the port is open")
    print("  - You can use an SSH tunnel: ssh -L 5000:localhost:5000 user@server")
    print("=" * 60)
    
    # Run the Flask application
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)

if __name__ == '__main__':
    main()
