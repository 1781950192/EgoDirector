import os
import pandas as pd
from collections import defaultdict

import nltk
from nltk.corpus import wordnet as wn

nltk.data.path.append('/mnt/data/xgl/mydata/nltk_data')

def parse_gtea_labels_for_evaluation(labels_dir):
    """Parse a GTEA label file for evaluation."""
    ground_truth = defaultdict(list)
    
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        video_name = label_file.replace('.txt', '')
        
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or '><' not in line:
                continue
            
            action_part = line.split('(')[0].strip()
            time_part = line.split('(')[1].split(')')[0]
            
            parts = action_part.split('><')
            verb = parts[0].replace('<', '')
            noun_str = parts[1].replace('>', '')
            nouns = [n.strip() for n in noun_str.split(',')]
            
            start_frame, end_frame = map(int, time_part.split('-'))
            frame_key = f"{video_name}_{start_frame}_{end_frame}"
            
            ground_truth[frame_key].append({
                'verb': verb,
                'nouns': nouns,
                'start_frame': start_frame,
                'end_frame': end_frame
            })
    
    return ground_truth

def single_wups(word1, word2, pos=wn.NOUN, threshold=0.1):
    """Compute the Wu-Palmer similarity of a single word."""
    if not word1 or not word2:
        return 0.0
    
    syns1 = wn.synsets(word1, pos=pos)
    syns2 = wn.synsets(word2, pos=pos)
    
    if not syns1 or not syns2:
        return 0.0
    
    score = syns1[0].wup_similarity(syns2[0])
    if score is None:
        return 0.0
    
    return 0.0 if score < threshold else score

def calculate_noun_wups_avg_per_gt(gt_nouns, pred_nouns, threshold=0.1):
    """
    Compute the noun WUPS score: for each ground-truth noun take the best match among the predicted nouns, then average them.
    (i.e. the WUPS is the average over gt_nouns)
    
    Args:
        gt_nouns: List of ground-truth nouns.
        pred_nouns: List of predicted nouns.
        threshold: WUPS threshold.
        
    Returns:
        float: The average WUPS score.
    """
    if not gt_nouns:
        return 0.0
    
    # Store the best matching score of each ground-truth noun
    best_scores_per_gt = []
    
    for gt_noun in gt_nouns:
        # For each ground-truth noun, find the best match among the predicted nouns
        if pred_nouns:
            best_score = max(
                single_wups(gt_noun, pred_noun, pos=wn.NOUN, threshold=threshold)
                for pred_noun in pred_nouns
            )
        else:
            best_score = 0.0
        best_scores_per_gt.append(best_score)
    
    # Return the average of the best scores of all ground-truth nouns
    return sum(best_scores_per_gt) / len(best_scores_per_gt)

def evaluate_wups_for_multiple_top_k(result_file, ground_truth, top_k_list=[1, 5], wups_threshold=0.1):
    """
    Compute the WUPS scores for several top-k values, explicitly averaging over gt_nouns.
    
    Args:
        result_file: Path of the prediction result file.
        ground_truth: Ground-truth annotation data.
        top_k_list: List of top-k values to compute.
        wups_threshold: WUPS score threshold.
        
    Returns:
        dict: Evaluation result of each top-k.
    """
    df = pd.read_csv(result_file)
    
    # Initialize the storage structure
    results = {k: {'verb_scores': [], 'noun_scores': []} for k in top_k_list}
    total_predictions = 0
    
    for _, row in df.iterrows():
        video_frame_id = row['narration_id']
        
        if video_frame_id not in ground_truth:
            continue
            
        gt_entry = ground_truth[video_frame_id][0]
        gt_verb = gt_entry['verb']
        gt_nouns = gt_entry['nouns']
        
        total_predictions += 1
        
        # Compute the score for each top-k
        for top_k in top_k_list:
            # Get the top-k predictions
            pred_verbs = [v.strip() for v in row['verbs'].split(';')[:top_k] if v.strip()]
            pred_nouns = [n.strip() for n in row['nouns'].split(';')[:top_k] if n.strip()]
            
            # Verb WUPS: take the best match among the predicted verbs
            if pred_verbs:
                best_verb_wups = max(
                    single_wups(gt_verb, pred_v, pos=wn.VERB, threshold=wups_threshold)
                    for pred_v in pred_verbs
                )
            else:
                best_verb_wups = 0.0
            results[top_k]['verb_scores'].append(best_verb_wups)
            
            # Noun WUPS: explicitly average over gt_nouns
            noun_wups = calculate_noun_wups_avg_per_gt(
                gt_nouns, 
                pred_nouns, 
                threshold=wups_threshold
            )
            results[top_k]['noun_scores'].append(noun_wups)
    
    # Compute the average score of each top-k
    final_results = {}
    for top_k in top_k_list:
        verb_wups = sum(results[top_k]['verb_scores']) / total_predictions if total_predictions > 0 else 0.0
        noun_wups = sum(results[top_k]['noun_scores']) / total_predictions if total_predictions > 0 else 0.0
        
        final_results[top_k] = {
            'verb_wups': verb_wups,
            'noun_wups': noun_wups,
            'total_predictions': total_predictions
        }
    
    for top_k in top_k_list:
        print(f"Top-{top_k:2d}:")
        print(f"  Verb WUPS: {final_results[top_k]['verb_wups']:.4f}")
        print(f"  Noun WUPS: {final_results[top_k]['noun_wups']:.4f}")
        print((final_results[top_k]['verb_wups']+final_results[top_k]['noun_wups'])/2)
        if top_k < max(top_k_list):
            print()
    
    return final_results

def main():
    # Configure the paths
    labels_dir = '/mnt/data/xgl/mydata/EGTEA++/EGTEA/GTEA_Labels_71/labels'
    result_file = '/mnt/data/xgl/mycode/action_agent_vllm/gtea_val_results_32.txt'
    
    # Parse the ground-truth labels
    ground_truth = parse_gtea_labels_for_evaluation(labels_dir)
    
    # Compute the WUPS scores
    results = evaluate_wups_for_multiple_top_k(
        result_file=result_file,
        ground_truth=ground_truth,
        top_k_list=[1, 5],  # More top-k values can be added
        wups_threshold=0.1
    )
    
    return results

if __name__ == '__main__':
    main()