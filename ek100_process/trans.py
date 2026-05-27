import csv
import ast
from collections import defaultdict

# ====================== Change these four lines ======================
VERB_CSV   = "/mnt/data/xgl/mydata/ek100/EPIC_100_verb_classes.csv"
NOUN_CSV   = "/mnt/data/xgl/mydata/ek100/EPIC_100_noun_classes.csv"
INPUT_TXT  = "/mnt/data/xgl/mycode/action_agent_vllm/ek100_val_8B_32frames.txt"
VAL_CSV    = "/mnt/data/xgl/mydata/ek100/EPIC_100_validation.csv"
# =====================================================

# Step 1: Load the official GT
print("1. Loading the official validation GT...")
gt = {}
with open(VAL_CSV, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        gt[r['narration_id']] = (int(r['verb_class']), int(r['noun_class']))

# Step 2: Build the key/instances -> id mapping table
print("2. Building the verb/noun mapping table...")
verb_map = {}
noun_map = {}
with open(VERB_CSV, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        key = r['key'].lower()
        for x in [key]:
            verb_map[x] = int(r['id'])
with open(NOUN_CSV, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        key = r['key'].lower()
        for x in [key]:
            noun_map[x] = int(r['id'])


# Step 3: Read the prediction txt and map it to IDs
print("3. Reading and mapping your predictions...")
pred = {}   # nid -> ([v1,v2,..], [n1,n2,..])

with open(INPUT_TXT, encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row or not row[0].startswith('P'):
            continue
        nid = row[0]
        verbs_str = row[1]   # put;put;take;...
        nouns_str = row[2]   # tomato;knife;...

        verb_list = [v.strip().lower() for v in verbs_str.split(';') if v.strip()]
        noun_list = [n.strip().lower() for n in nouns_str.split(';') if n.strip()]

        v_ids = []
        for v in verb_list:
            v = v.replace(' ', '-')           # "turn on" → "turn-on"
            v_ids.append(verb_map.get(v, verb_map.get(v.replace('-', ' '), -1)))

        n_ids = []
        for n in noun_list:
            orig = n.lower()
            n_ids.append(noun_map.get(n, noun_map.get(orig, -1)))

        # Take the top-5 (pad with -1 if not enough)
        pred[nid] = (v_ids[:5], n_ids[:5])

# Step 4: Compute the accuracy
print("4. Computing the accuracy...")
common = set(gt.keys()) & set(pred.keys())
total = len(common)

v1 = n1 = both1 = v5 = n5 = both5 = 0

for nid in common:
    true_v, true_n = gt[nid]
    pred_v5, pred_n5 = pred[nid]

    pred_v1 = pred_v5[0] if pred_v5 else -1
    pred_n1 = pred_n5[0] if pred_n5 else -1

    if pred_v1 == true_v: v1 += 1
    if pred_n1 == true_n: n1 += 1
    if pred_v1 == true_v and pred_n1 == true_n: 
        both1 += 1    

    if true_v in pred_v5: v5 += 1
    if true_n in pred_n5: n5 += 1
    k = min(len(pred_v5), len(pred_n5), 5)
    both5 += any(pred_v5[i] == true_v and pred_n5[i] == true_n for i in range(k))

print(f"Top-1 Verb  Acc    : {v1/total*100:5.2f}%  ({v1}/{total})")
print(f"Top-1 Noun  Acc    : {n1/total*100:5.2f}%  ({n1}/{total})")
print(f"Top-1 Both  Acc    : {both1/total*100:5.2f}%  ({both1}/{total})")
print(f"Top-5 Verb  Hit    : {v5/total*100:5.2f}%")
print(f"Top-5 Noun  Hit    : {n5/total*100:5.2f}%")
print(f"Top-5 Both  Hit    : {both5/total*100:5.2f}%")
print("="*55)