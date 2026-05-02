import os
import re
import json
from collections import defaultdict

def parse_action_line(line):
    pattern = r'<(\w+)><([^>]+)> \(([0-9]+)-([0-9]+)\) \[([0-3])\]'
    match = re.match(pattern, line.strip())
    if not match:
        return None
    verb = match.group(1)
    objects = [obj.strip() for obj in match.group(2).split(',') if obj.strip()]
    return verb, objects

def parse_object_line(line):
    pattern = r'<(\w+)> \(([0-9]+)-([0-9]+)\)'
    match = re.match(pattern, line.strip())
    if match:
        return match.group(1)
    return None

def extract_from_folder(folder_path):
    verbs = set()
    nouns = set()
    noun_verb_counts = defaultdict(lambda: defaultdict(int))  # noun -> verb -> count

    print(f"正在扫描文件夹: {folder_path}")

    for filename in os.listdir(folder_path):
        if not filename.endswith('.txt'):
            continue
        file_path = os.path.join(folder_path, filename)
        print(f"  处理: {filename}")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                action = parse_action_line(line)
                if action:
                    verb, objs = action
                    verbs.add(verb)
                    nouns.update(objs)
                    for obj in objs:
                        noun_verb_counts[obj][verb] += 1
                    continue

                obj = parse_object_line(line)
                if obj:
                    nouns.add(obj)

    # 排序
    verb_list = sorted(verbs)
    noun_list = sorted(nouns)
    verb2id = {v: i for i, v in enumerate(verb_list)}
    noun2id = {n: i for i, n in enumerate(noun_list)}

    # 计算概率：P(verb|noun)
    noun_verb_probs = {}
    for noun in noun_list:
        counts = noun_verb_counts[noun]
        total = sum(counts.values())
        if total > 0:
            # 按概率降序排列
            probs = {verb: count / total for verb, count in counts.items()}
            probs_sorted = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
            noun_verb_probs[noun] = {verb.capitalize(): round(prob, 6) for verb, prob in probs_sorted.items()}

    return verb_list, noun_list, verb2id, noun2id, noun_verb_probs

def save_results(verb_list, noun_list, verb2id, noun2id, noun_verb_probs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. verbs.txt: 词\t编号
    with open(os.path.join(output_dir, 'verbs.txt'), 'w', encoding='utf-8') as f:
        for verb in verb_list:
            f.write(f"{verb}\t{verb2id[verb]}\n")

    # 2. nouns.txt: 词\t编号
    with open(os.path.join(output_dir, 'nouns.txt'), 'w', encoding='utf-8') as f:
        for noun in noun_list:
            f.write(f"{noun}\t{noun2id[noun]}\n")

    # 3. noun_verb_probs.json: 目标格式
    with open(os.path.join(output_dir, 'noun_verb_probs.json'), 'w', encoding='utf-8') as f:
        json.dump(noun_verb_probs, f, indent=2, ensure_ascii=False)

    # 4. vocab.json（完整词表）
    vocab = {
        "verbs": verb2id,
        "nouns": noun2id,
        "verb_list": verb_list,
        "noun_list": noun_list,
        "noun_verb_probs": noun_verb_probs
    }
    with open(os.path.join(output_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    print(f"\n提取完成！")
    print(f"  动词数量: {len(verb_list)}")
    print(f"  名词数量: {len(noun_list)}")
    print(f"  结果已保存至: {output_dir}")
    print(f"  noun_verb_probs.json 已生成（动词首字母大写，按概率降序）")

# ==========================
# 主程序
# ==========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="提取 GTEA 动词名词 + 输出 JSON 概率表")
    parser.add_argument("folder", help="包含 .txt 标签文件的文件夹路径")
    parser.add_argument("-o", "--output", default="gtea_process", help="输出目录（默认: gtea_vocab）")
    parser.add_argument("--no-capitalize", action="store_true", help="动词不首字母大写")

    args = parser.parse_args()

    verb_list, noun_list, verb2id, noun2id, noun_verb_probs_raw = extract_from_folder(args.folder)

    # 可选：是否大写动词
    if not args.no_capitalize:
        noun_verb_probs = {
            noun: {verb.capitalize(): round(prob, 6) for verb, prob in probs.items()}
            for noun, probs in noun_verb_probs_raw.items()
        }
        # 降序排列
        noun_verb_probs = {
            noun: dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
            for noun, probs in noun_verb_probs.items()
        }
    else:
        noun_verb_probs = {
            noun: dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
            for noun, probs in noun_verb_probs_raw.items()
        }

    save_results(verb_list, noun_list, verb2id, noun2id, noun_verb_probs, args.output)