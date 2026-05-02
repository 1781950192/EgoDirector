import os
import pandas as pd
import csv
import io

# ==================== Verb mapping (aliases supported) ====================
VERB_MAPPING = {
    'inspect': 1, 'read': 1,
    'open': 2,
    'take': 3,
    'cut': 4,
    'turn on': 5,
    'put': 6,
    'operate': 7,
    'close': 8,
    'move around': 9,
    'wash': 10,
    'spread': 11,
    'turn off': 12,
    'divide': 13, 'pull apart': 13,
    'clean': 14, 'wipe': 14,
    'mix': 15,
    'pour': 16,
    'compress': 17,
    'crack': 18,
    'squeeze': 19
}

def get_verb_id(verb_text):
    return VERB_MAPPING.get(verb_text.strip().lower(), -1)


# ==================== Load the official noun mapping ====================
def load_noun_mapping(file_path):
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(maxsplit=1)
            if len(parts) == 2:
                word, idx = parts
                mapping[word.strip().lower()] = int(idx)
    return mapping


# ==================== Read the EGTEA ground-truth labels ====================
def load_egtea_labels(label_file):
    labels = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            vid = parts[0]
            verb_id = int(parts[2])
            noun_ids = [int(n) for n in parts[3:]]
            labels[vid] = (verb_id, noun_ids)  # Keep a list so that its length can be checked
    return labels


# ==================== Parse the prediction results ====================
def parse_predictions(result_file, top_k=5):
    # Use the csv module to handle fields that may contain commas
    predictions = {}
    
    with open(result_file, 'r', encoding='utf-8') as f:
        # Read the first line as the header
        header = f.readline().strip().split(',')
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Use csv.reader to correctly parse fields that contain commas
            reader = csv.reader(io.StringIO(line))
            try:
                row = next(reader)
            except StopIteration:
                continue
            
            if len(row) < 6:
                continue
            
            vid = row[0].strip()
            
            # Predicted verb list
            pred_verbs = [v.strip().lower() for v in row[1].split(';') if v.strip()][:top_k*2]
            
            # Predicted noun list (for the single-noun case)
            pred_nouns = [n.strip().lower() for n in row[2].split(';') if n.strip()][:top_k*2]
            
            # selected_noun_keys (for the multi-noun case)
            selected_str = row[5].strip() if len(row) > 5 else ''
            if not selected_str or selected_str.lower() == 'nan':
                selected_nouns = []
            else:
                selected_nouns = [n.strip().lower() for n in selected_str.split(';') if n.strip()]
            
            predictions[vid] = {
                'verbs': pred_verbs,
                'nouns': pred_nouns,           # Top-K predicted nouns
                'selected_nouns': selected_nouns  # Selected nouns
            }
    
    return predictions


# ==================== Main evaluation function (mixed rules) ====================
def evaluate_egtea(result_file, label_file, noun_map_file, top_k=5):
    # Load the noun mapping
    noun_map = load_noun_mapping(noun_map_file)
    print(f"Loaded {len(noun_map)} nouns from {noun_map_file}")

    # Load the ground-truth labels
    ground_truth = load_egtea_labels(label_file)
    print(f"Loaded {len(ground_truth)} ground truth labels")

    # Parse the predictions
    predictions = parse_predictions(result_file, top_k=top_k)
    print(f"Parsed {len(predictions)} predictions")

    total = 0
    top1_verb = top1_noun = top1_action = 0
    topk_verb = topk_noun = topk_action = 0

    single_noun_cases = 0
    multi_noun_cases = 0

    for vid, pred in predictions.items():
        if vid not in ground_truth:
            continue

        pri_noun = ""
        pri_verb = ""

        true_verb_id, true_noun_ids = ground_truth[vid]
        true_noun_set = set(true_noun_ids)
        num_gt_nouns = len(true_noun_ids)

        pred_verbs = pred['verbs']
        pred_nouns_text = pred['nouns']
        selected_nouns_text = pred['selected_nouns']

        total += 1
        if num_gt_nouns == 1:
            single_noun_cases += 1
        else:
            multi_noun_cases += 1

        # ==================== Verb evaluation (unchanged) ====================
        # Top-1
        top1_verb_text = pred_verbs[0] if pred_verbs else ""
        pred_verb_id_top1 = get_verb_id(top1_verb_text)
        verb_correct_top1 = (pred_verb_id_top1 == true_verb_id) and (pred_verb_id_top1 != -1)
        pri_verb = top1_verb_text
        # Top-k
        pred_verb_ids_topk = {get_verb_id(v) for v in pred_verbs[:top_k] if get_verb_id(v) != -1}
        verb_correct_topk = true_verb_id in pred_verb_ids_topk


        # ==================== Noun evaluation (the rule switches with the noun count) ====================
        if num_gt_nouns == 1:
            # Single noun: use the standard Top-K evaluation on the nouns column
            gt_noun_id = true_noun_ids[0]

            # Top-1
            top1_noun_text = pred_nouns_text[0] if pred_nouns_text else ""
            pred_noun_id_top1 = noun_map.get(top1_noun_text, -1)
            noun_correct_top1 = (pred_noun_id_top1 == gt_noun_id)
            pri_noun = top1_noun_text


            # Top-k
            pred_noun_ids_topk = {noun_map.get(n, -1) for n in pred_nouns_text[:top_k]}
            pred_noun_ids_topk = {nid for nid in pred_noun_ids_topk if nid != -1}
            noun_correct_topk = gt_noun_id in pred_noun_ids_topk

        else:
            # Multiple nouns: use selected_noun_keys and require all of them to hit
            selected_noun_ids = {noun_map.get(n, -1) for n in selected_nouns_text}
            selected_noun_ids = {nid for nid in selected_noun_ids if nid != -1}

            noun_correct = true_noun_set.issubset(selected_noun_ids)
            noun_correct_top1 = noun_correct
            noun_correct_topk = noun_correct

        # ==================== Action ====================
        action_correct_top1 = verb_correct_top1 and noun_correct_top1
        # if(action_correct_top1 == 1):

        action_correct_topk = verb_correct_topk and noun_correct_topk

        # Accumulate
        top1_verb += int(verb_correct_top1)
        top1_noun += int(noun_correct_top1)
        top1_action += int(action_correct_top1)

        topk_verb += int(verb_correct_topk)
        topk_noun += int(noun_correct_topk)
        topk_action += int(action_correct_topk)

    print(f"Total samples: {total}  |  Single-noun: {single_noun_cases}  |  Multi-noun: {multi_noun_cases}")
    print(f"{'':<18} {'Top-1':<25} {'Top-{top_k}':<25}")
    print("-" * 80)
    print(f"{'Verb':<18} {top1_verb/total:.4f} ({top1_verb}/{total})"
          f"{'':>12} {topk_verb/total:.4f} ({topk_verb}/{total})")
    print(f"{'Noun':<18} {top1_noun/total:.4f} ({top1_noun}/{total})"
          f"{'':>12} {topk_noun/total:.4f} ({topk_noun}/{total})")
    print(f"{'Action':<18} {top1_action/total:.4f} ({top1_action}/{total})"
          f"{'':>12} {topk_action/total:.4f} ({topk_action}/{total})")

    return {
        'total': total,
        'single_noun': single_noun_cases,
        'multi_noun': multi_noun_cases,
        'top1': {'verb': top1_verb / total, 'noun': top1_noun / total, 'action': top1_action / total},
        f'top{top_k}': {'verb': topk_verb / total, 'noun': topk_noun / total, 'action': topk_action / total}
    }


# ==================== Main function ====================
def main():
    # Please change this according to your actual path
    result_file = '/mnt/data/xgl/mycode/action_agent_vllm/egtea_results_32frames.txt'
    label_file = '/mnt/data/xgl/mydata/EGTEA++/EGTEA/Action_Annotations/test_split1.txt'
    noun_map_file = '/mnt/data/xgl/mydata/EGTEA++/EGTEA/Action_Annotations/noun_idx.txt'

    evaluate_egtea(
        result_file=result_file,
        label_file=label_file,
        noun_map_file=noun_map_file,
        top_k=5
    )


if __name__ == '__main__':
    main()