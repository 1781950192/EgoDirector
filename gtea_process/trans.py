import os
import pandas as pd
from collections import defaultdict
import math

# Noun and verb index mapping (kept in case it is needed later)
NOUN_MAPPING = {
    'bread': 0, 'cheese': 1, 'chocolate': 2, 'coffee': 3, 'cup': 4,
    'honey': 5, 'hotdog': 6, 'jam': 7, 'ketchup': 8, 'mayonnaise': 9,
    'mustard': 10, 'peanut': 11, 'spoon': 12, 'sugar': 13, 'tea': 14, 'water': 15
}

VERB_MAPPING = {
    'close': 0, 'fold': 1, 'open': 2, 'pour': 3, 'put': 4,
    'scoop': 5, 'shake': 6, 'spread': 7, 'stir': 8, 'take': 9
}


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


def evaluate_with_multiple_predictions(result_file, ground_truth, top_k=5):
    """Top-k evaluation (the action uses the combination at the corresponding position, standard way)."""
    df = pd.read_csv(result_file)

    total_predictions = 0
    verb_correct_top1 = noun_correct_top1 = action_correct_top1 = 0
    verb_correct_topk = noun_correct_topk = action_correct_topk = 0

    for _, row in df.iterrows():
        video_frame_id = row['narration_id']
        noun_all = row['selected_noun_keys']
        
        pred_verbs = [v.strip() for v in row['verbs'].split(';')[:top_k] if v.strip()]
        pred_nouns = [n.strip() for n in row['nouns'].split(';')[:top_k] if n.strip()]

        if video_frame_id not in ground_truth:
            continue

        gt_entries = ground_truth[video_frame_id]
        gt_entry = gt_entries[0]
        gt_verb = gt_entry['verb']
        gt_nouns = gt_entry['nouns']

        total_predictions += 1

        # Top-1
        top1_verb = pred_verbs[0] if pred_verbs else ""
        top1_noun = pred_nouns[0] if pred_nouns else ""
        verb_match_top1 = (top1_verb == gt_verb)
        if len(gt_nouns)==1:
            noun_match_top1 = (top1_noun == gt_nouns[0])
        else:
            a = len(gt_nouns)
            if isinstance(noun_all, float) and math.isnan(noun_all):
                noun_match_top1 = 0
            else:
                # noun_match_top1 = any(top1_noun == n for n in gt_nouns)
                noun_match_top1 = set(noun_all[:a]).issubset(set(gt_nouns))
        
        action_match_top1 = verb_match_top1 and noun_match_top1

        if verb_match_top1: verb_correct_top1 += 1
        if noun_match_top1: noun_correct_top1 += 1
        if action_match_top1: action_correct_top1 += 1

        # Top-k Verb & Noun
        verb_in_topk = gt_verb in pred_verbs
        if len(gt_nouns)==1:
            noun_in_topk = gt_nouns[0] in pred_nouns
        else:
            noun_in_topk = all(n in pred_nouns for n in gt_nouns)
        if verb_in_topk: verb_correct_topk += 1
        if noun_in_topk: noun_correct_topk += 1

        top_k_verbs = pred_verbs[:top_k]  # Take the first top_k verbs
        top_k_nouns = pred_nouns[:top_k]  # Take the first top_k nouns

        top_k_action_combinations = []
        for verb in top_k_verbs:
            for noun in top_k_nouns:
                top_k_action_combinations.append((verb, noun))

        if len(gt_nouns) == 1:
            gt_action = (gt_verb, gt_nouns[0])
            action_match_topk = gt_action in top_k_action_combinations
        else:
            # Multiple nouns: all of them have to match
            # Find all the combinations that match the verb
            matching_actions = [
                (verb, noun) for verb, noun in top_k_action_combinations 
                if verb == gt_verb
            ]
            
            # Check whether every ground-truth noun has a matching prediction
            matched_nouns = {noun for _, noun in matching_actions}
            action_match_topk = all(n in matched_nouns for n in gt_nouns)

        if action_match_topk:
            action_correct_topk += 1

    # Compute the accuracy
    verb_acc_top1 = verb_correct_top1 / total_predictions
    noun_acc_top1 = noun_correct_top1 / total_predictions
    action_acc_top1 = action_correct_top1 / total_predictions

    verb_acc_topk = verb_correct_topk / total_predictions
    noun_acc_topk = noun_correct_topk / total_predictions
    action_acc_topk = action_correct_topk / total_predictions

    print(f"=== Top-{top_k} evaluation results ===")
    print(f"Verb Top-1: {verb_acc_top1:.4f}  |  Top-{top_k}: {verb_acc_topk:.4f}")
    print(f"Noun Top-1: {noun_acc_top1:.4f}  |  Top-{top_k}: {noun_acc_topk:.4f}")
    print(f"Action Top-1: {action_acc_top1:.4f}  |  Top-{top_k}: {action_acc_topk:.4f}\n")

    return {
        'top1': {'verb': verb_acc_top1, 'noun': noun_acc_top1, 'action': action_acc_top1},
        f'top{top_k}': {'verb': verb_acc_topk, 'noun': noun_acc_topk, 'action': action_acc_topk}
    }


def main():
    # Configure the paths (change them according to your setup)
    labels_dir = '/mnt/data/xgl/mydata/EGTEA++/EGTEA/GTEA_Labels_71/labels'
    result_file = '/mnt/data/xgl/mycode/action_agent_vllm/gtea_val_results_32.txt'

    ground_truth = parse_gtea_labels_for_evaluation(labels_dir)

    # Top-5 evaluation
    multi_results = evaluate_with_multiple_predictions(result_file, ground_truth, top_k=5)


if __name__ == '__main__':
    main()