import os
import pandas as pd

import nltk
from nltk.corpus import wordnet as wn

# If running on a server, the nltk_data path has to be specified
# nltk.data.path.append('/mnt/data/xgl/mydata/nltk_data')

# ====================== WUPS base functions ======================

def single_wups(word1, word2, pos=wn.NOUN, threshold=0.1):
    """Compute the Wu-Palmer similarity of a single word."""
    if not word1 or not word2:
        return 0.0
    
    syns1 = wn.synsets(word1, pos=pos)
    syns2 = wn.synsets(word2, pos=pos)
    
    if not syns1 or not syns2:
        return 0.0
    
    # Take the first synset (common practice)
    score = syns1[0].wup_similarity(syns2[0])
    if score is None:
        return 0.0
    
    return 0.0 if score < threshold else score


def calculate_noun_wups(gt_noun, pred_nouns, threshold=0.1):
    """
    Noun WUPS: the best match between a single ground-truth noun and the predicted noun list.
    Each EK100 sample has only one noun, so the best matching score is returned directly.
    """
    if not gt_noun:
        return 0.0
    
    if not pred_nouns:
        return 0.0
    
    return max(
        single_wups(gt_noun, pred_noun, pos=wn.NOUN, threshold=threshold)
        for pred_noun in pred_nouns
    )


# ====================== EK100 data loading ======================

def load_id_to_word(verb_csv_path, noun_csv_path):
    """Load the id -> word (key) mapping."""
    id_to_verb = {}
    id_to_noun = {}
    
    df_verb = pd.read_csv(verb_csv_path)
    for _, row in df_verb.iterrows():
        id_to_verb[int(row['id'])] = row['key']
    
    df_noun = pd.read_csv(noun_csv_path)
    for _, row in df_noun.iterrows():
        id_to_noun[int(row['id'])] = row['key']
    
    return id_to_verb, id_to_noun


def parse_ek100_ground_truth(gt_csv_path, id_to_verb, id_to_noun):
    """
    Parse the EK100 validation csv.
    Returns a dict: narration_id -> {'verb': str, 'noun': str}
    """
    ground_truth = {}
    
    df = pd.read_csv(gt_csv_path)
    for _, row in df.iterrows():
        nid = row['narration_id']
        verb_word = id_to_verb.get(int(row['verb_class']), "")
        noun_word = id_to_noun.get(int(row['noun_class']), "")
        
        if verb_word and noun_word:
            ground_truth[nid] = {
                'verb': verb_word,
                'noun': noun_word   # A plain string, no longer wrapped in a list
            }
    
    return ground_truth


# ====================== Main WUPS evaluation function ======================

def evaluate_wups_for_multiple_top_k(
    result_file,
    ground_truth,
    top_k_list=[1, 5],
    wups_threshold=0.1
):
    """
    Compute the verb and noun WUPS for several top-k values.
    result_file: Prediction csv, it must contain the columns narration_id, verbs, nouns
                 verbs/nouns are separated by ';' and sorted by confidence from high to low
    """
    df = pd.read_csv(result_file)
    
    results = {k: {'verb_scores': [], 'noun_scores': []} for k in top_k_list}
    total_predictions = 0
    
    for _, row in df.iterrows():
        nid = row['narration_id']
        
        if nid not in ground_truth:
            continue
            
        gt = ground_truth[nid]
        gt_verb = gt['verb']
        gt_noun = gt['noun']   # A single string
        
        total_predictions += 1
        
        for top_k in top_k_list:
            # Top-k predictions
            pred_verbs = [v.strip() for v in str(row['verbs']).split(';')[:top_k] if v.strip()]
            pred_nouns = [n.strip() for n in str(row['nouns']).split(';')[:top_k] if n.strip()]
            
            # Verb WUPS: the prediction with the highest similarity to gt_verb
            best_verb_wups = max(
                single_wups(gt_verb, pred_v, pos=wn.VERB, threshold=wups_threshold)
                for pred_v in pred_verbs
            ) if pred_verbs else 0.0
            results[top_k]['verb_scores'].append(best_verb_wups)
            
            # Noun WUPS: the best match between gt_noun and the predicted nouns
            noun_wups = calculate_noun_wups(gt_noun, pred_nouns, threshold=wups_threshold)
            results[top_k]['noun_scores'].append(noun_wups)
    
    # Output the results
    print("\n=== WUPS Evaluation Results (EK100) ===")
    for top_k in top_k_list:
        verb_wups = sum(results[top_k]['verb_scores']) / total_predictions if total_predictions else 0.0
        noun_wups = sum(results[top_k]['noun_scores']) / total_predictions if total_predictions else 0.0
        
        print(f"Top-{top_k:2d}:")
        print(f"  Verb WUPS : {verb_wups:.4f}")
        print(f"  Noun WUPS : {noun_wups:.4f}")
        print((noun_wups+verb_wups)/2)
        if top_k < max(top_k_list):
            print()
    
    return {k: {'verb_wups': sum(results[k]['verb_scores']) / total_predictions,
                'noun_wups': sum(results[k]['noun_scores']) / total_predictions,
                'total': total_predictions}
            for k in top_k_list}


# ====================== Main function ======================

def main():
    verb_csv    = "/mnt/data/xgl/mydata/ek100/EPIC_100_verb_classes.csv"
    noun_csv    = "/mnt/data/xgl/mydata/ek100/EPIC_100_noun_classes.csv"
    gt_csv      = "/mnt/data/xgl/mydata/ek100/EPIC_100_validation.csv"
    result_file = "/mnt/data/xgl/mycode/action_agent_vllm/ek100_val_8B_32frames.txt"  # Your prediction file
    
    print("Loading id -> word mappings...")
    id_to_verb, id_to_noun = load_id_to_word(verb_csv, noun_csv)
    
    print("Parsing EK100 ground truth...")
    ground_truth = parse_ek100_ground_truth(gt_csv, id_to_verb, id_to_noun)
    print(f"Loaded {len(ground_truth)} valid ground truth samples.")
    
    print("Evaluating WUPS...")
    results = evaluate_wups_for_multiple_top_k(
        result_file=result_file,
        ground_truth=ground_truth,
        top_k_list=[1, 5],          # Can be changed to [1, 3, 5, 10], etc.
        wups_threshold=0.1
    )
    
    return results


if __name__ == '__main__':
    main()