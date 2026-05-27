import os
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn

# Make sure wordnet has been downloaded (if not, download it manually to the path below)
nltk.data.path.append('/mnt/data/xgl/mydata/nltk_data')  # Change this according to your path


# ==================== Verb mapping (aliases supported, same as before) ====================
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

# Reverse mapping: ID -> canonical verb text (used for the WUPS computation)
ID_TO_VERB_TEXT = {}
for text, vid in VERB_MAPPING.items():
    ID_TO_VERB_TEXT[vid] = text  # Pick one representative (all aliases map to the same ID)


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

# Reverse: ID -> noun text
def load_reverse_noun_mapping(file_path):
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(maxsplit=1)
            if len(parts) == 2:
                word, idx = parts
                mapping[int(idx)] = word.strip().lower()
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
            labels[vid] = (verb_id, noun_ids)
    return labels


# ==================== Read the prediction results (handle the special format) ====================
def load_predictions(result_file):
    """
    Load the prediction result file and handle fields that contain commas.
    """
    predictions = []
    
    with open(result_file, 'r', encoding='utf-8') as f:
        header = f.readline().strip()
        
        for line_num, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            
            # Split by commas, but only split the first 6 fields
            parts = line.split(',')
            
            if len(parts) < 6:
                print(f"Warning: Line {line_num} has only {len(parts)} fields, skipping")
                continue
            
            # The first 5 fields are fixed
            narration_id = parts[0]
            verbs = parts[1]
            nouns = parts[2]
            actions = parts[3]
            confidences = parts[4]
            
            # The 6th and later fields are merged into selected_noun_keys (may contain commas)
            selected_noun_keys = ','.join(parts[5:])
            
            predictions.append({
                'narration_id': narration_id,
                'verbs': verbs,
                'nouns': nouns,
                'actions': actions,
                'confidences': confidences,
                'selected_noun_keys': selected_noun_keys
            })
    
    df = pd.DataFrame(predictions)
    return df


# ==================== WUPS helper functions ====================
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


def calculate_noun_wups_avg_per_gt(gt_nouns_text, pred_nouns_text, threshold=0.1):
    """
    For each ground-truth noun take the best match among the predicted nouns, then average them (per GT noun average).
    """
    if not gt_nouns_text:
        return 0.0
    
    best_scores_per_gt = []
    for gt_noun in gt_nouns_text:
        if pred_nouns_text:
            best_score = max(
                single_wups(gt_noun, pred_noun, pos=wn.NOUN, threshold=threshold)
                for pred_noun in pred_nouns_text
            )
        else:
            best_score = 0.0
        best_scores_per_gt.append(best_score)
    
    return sum(best_scores_per_gt) / len(best_scores_per_gt)


# ==================== Main WUPS evaluation function (EGTEA version) ====================
def evaluate_wups_egtea(result_file, label_file, noun_map_file, top_k_list=[1, 5], wups_threshold=0.1):
    # Load the noun mapping
    noun_text_to_id = load_noun_mapping(noun_map_file)
    id_to_noun_text = load_reverse_noun_mapping(noun_map_file)
    print(f"Loaded {len(noun_text_to_id)} nouns")

    # Load the ground-truth labels
    ground_truth = load_egtea_labels(label_file)
    print(f"Loaded {len(ground_truth)} ground truth labels")

    # Read the predictions (using the custom parsing function)
    df = load_predictions(result_file)
    print(f"Loaded {len(df)} predictions")

    # Initialize the results
    results = {k: {'verb_scores': [], 'noun_scores': []} for k in top_k_list}
    evaluated_count = 0

    for _, row in df.iterrows():
        vid = str(row['narration_id']).strip()
        
        if vid not in ground_truth:
            continue
        
        true_verb_id, true_noun_ids = ground_truth[vid]
        # Convert to text for WUPS
        true_verb_text = ID_TO_VERB_TEXT.get(true_verb_id, "")
        true_noun_texts = [id_to_noun_text.get(nid, "") for nid in true_noun_ids]
        true_noun_texts = [t for t in true_noun_texts if t]  # Filter out invalid entries

        if not true_verb_text or not true_noun_texts:
            continue

        evaluated_count += 1

        for top_k in top_k_list:
            # Top-k predicted text
            pred_verbs = [v.strip().lower() for v in str(row['verbs']).split(';')[:top_k] if v.strip()]
            pred_nouns = [n.strip().lower() for n in str(row['nouns']).split(';')[:top_k] if n.strip()]

            # ==================== Verb WUPS ====================
            if pred_verbs:
                best_verb_wups = max(
                    single_wups(true_verb_text, pred_v, pos=wn.VERB, threshold=wups_threshold)
                    for pred_v in pred_verbs
                )
            else:
                best_verb_wups = 0.0
            results[top_k]['verb_scores'].append(best_verb_wups)

            # ==================== Noun WUPS ====================
            noun_wups = calculate_noun_wups_avg_per_gt(
                gt_nouns_text=true_noun_texts,
                pred_nouns_text=pred_nouns,
                threshold=wups_threshold
            )
            results[top_k]['noun_scores'].append(noun_wups)

    # ==================== Output the results ====================
    final_results = {}
    for top_k in top_k_list:
        verb_wups = sum(results[top_k]['verb_scores']) / evaluated_count if evaluated_count > 0 else 0.0
        noun_wups = sum(results[top_k]['noun_scores']) / evaluated_count if evaluated_count > 0 else 0.0
        
        final_results[top_k] = {
            'verb_wups': verb_wups,
            'noun_wups': noun_wups,
            'total': evaluated_count
        }

        print(f"Top-{top_k:2d}:")
        print(f"  Verb WUPS: {verb_wups:.4f}")
        print(f"  Noun WUPS: {noun_wups:.4f}")
        print((verb_wups + noun_wups)/2)
        if top_k < max(top_k_list):
            print()

    return final_results


# ==================== Main function ====================
def main():
    # Please change this according to your actual path
    result_file = '/mnt/data/xgl/mycode/action_agent_vllm/egtea_results_cpm.txt'
    label_file = '/mnt/data/xgl/mydata/EGTEA++/EGTEA/Action_Annotations/test_split1.txt'
    noun_map_file = '/mnt/data/xgl/mydata/EGTEA++/EGTEA/Action_Annotations/noun_idx.txt'

    print("=== EGTEA WUPS Evaluation ===")
    results = evaluate_wups_egtea(
        result_file=result_file,
        label_file=label_file,
        noun_map_file=noun_map_file,
        top_k_list=[1, 5],
        wups_threshold=0.1
    )

    return results


if __name__ == '__main__':
    main()