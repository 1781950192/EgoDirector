import csv
import ast
import logging

# 设置日志（可选，用于调试）
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger()


def map_key_to_id(input_str, csv_file_path, is_verb=True, debug=False):
    """
    将单个键或实例映射到CSV文件中对应的ID。
    支持动词连字符补丁、名词反转冒号、去s、chopping特殊处理。
    """
    input_str = input_str.strip()
    input_lower = input_str.lower()

    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = row['key'].lower()
            instances = [inst.lower() for inst in ast.literal_eval(row['instances'])]
            if input_lower == key or input_lower in instances:
                return int(row['id'])

        # 补丁逻辑
        file.seek(0)
        next(reader)

        if is_verb and ' ' in input_lower:
            patched = input_lower.replace(' ', '-')
            if debug:
                logger.debug(f"动词补丁: '{input_str}' -> '{patched}'")
            for row in reader:
                key = row['key'].lower()
                instances = [inst.lower() for inst in ast.literal_eval(row['instances'])]
                if patched == key or patched in instances:
                    return int(row['id'])

        elif not is_verb:
            # 反转两个词：spring onion -> onion:spring
            if ' ' in input_lower and len(input_lower.split()) == 2:
                words = input_lower.split()
                patched = f"{words[1]}:{words[0]}"
                file.seek(0); next(reader)
                if debug:
                    logger.debug(f"名词补丁(反转): '{input_str}' -> '{patched}'")
                for row in reader:
                    key = row['key'].lower()
                    instances = [inst.lower() for inst in ast.literal_eval(row['instances'])]
                    if patched == key or patched in instances:
                        return int(row['id'])

            # 去掉末尾s
            file.seek(0); next(reader)
            if input_lower.endswith('s') and len(input_lower) > 1:
                patched = input_lower[:-1]
                if debug:
                    logger.debug(f"名词补丁(去s): '{input_str}' -> '{patched}'")
                for row in reader:
                    key = row['key'].lower()
                    instances = [inst.lower() for inst in ast.literal_eval(row['instances'])]
                    if patched == key or patched in instances:
                        return int(row['id'])

            # chopping -> board:chopping
            file.seek(0); next(reader)
            if input_lower == 'chopping':
                patched = 'board:chopping'
                if debug:
                    logger.debug(f"名词补丁(chopping): '{input_str}' -> '{patched}'")
                for row in reader:
                    key = row['key'].lower()
                    instances = [inst.lower() for inst in ast.literal_eval(row['instances'])]
                    if patched == key or patched in instances:
                        return int(row['id'])

    if debug:
        logger.debug(f"{'动词' if is_verb else '名词'}未匹配: '{input_str}'")
    return input_str  # 未映射返回原词


def process_text_file(input_file, output_file, verb_csv, noun_csv, unmapped_rows_file, debug=False):
    """
    处理包含多个 verb;noun 对的文件，每对独立映射ID。
    """
    unmapped_rows = []

    with open(input_file, mode='r', encoding='utf-8') as infile, \
         open(output_file, mode='w', encoding='utf-8', newline='') as outfile, \
         open(unmapped_rows_file, mode='w', encoding='utf-8') as unmapped_file:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row_idx, row in enumerate(reader):
            if len(row) < 11:
                writer.writerow(row)
                continue

            # 提取多动作、多对象、多短语
            verbs_str = row[8].strip()      # take;use;take;put;take
            nouns_str = row[9].strip()      # pan;spoon;knife;board:chopping;cloth
            phrases_str = row[10].strip()   # take pan;use spoon;...

            verbs = [v.strip() for v in verbs_str.split(';') if v.strip()]
            nouns = [n.strip() for n in nouns_str.split(';') if n.strip()]

            # 映射每个动词和名词
            mapped_verbs = []
            mapped_nouns = []
            has_unmapped = False

            for v in verbs:
                vid = map_key_to_id(v, verb_csv, is_verb=True, debug=debug)
                mapped_verbs.append(str(vid))
                if isinstance(vid, str):  # 未映射
                    has_unmapped = True

            for n in nouns:
                nid = map_key_to_id(n, noun_csv, is_verb=False, debug=debug)
                mapped_nouns.append(str(nid))
                if isinstance(nid, str):  # 未映射
                    has_unmapped = True

            # 记录未完全映射的行
            if has_unmapped:
                unmapped_rows.append((row_idx, ','.join(row)))

            # 构造新行：替换第9、10列（索引8、9）
            new_row = row[:8] + [';'.join(mapped_verbs), ';'.join(mapped_nouns)] + row[10:]
            writer.writerow(new_row)

        # 写入未映射行信息
        unmapped_file.write('\n'.join(f"{idx}: {content}" for idx, content in unmapped_rows))


import csv
from typing import Dict, List, Tuple
from collections import Counter


def read_txt_file(txt_path: str) -> Dict[str, dict]:
    data = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split(',')
            if len(parts) < 12 or not parts[0].startswith('P'):
                continue

            nid = parts[0]
            verb_ids_str = parts[8].strip()
            noun_ids_str = parts[9].strip()
            narration = parts[10].strip()
            conf_str = parts[11].strip()

            try:
                verb_ids = [int(x) for x in verb_ids_str.split(';') if x.strip().isdigit()]
                noun_ids = [int(x) for x in noun_ids_str.split(';') if x.strip().isdigit()]
            except ValueError as e:
                continue

            data[nid] = {
                'verb_top5_ids': verb_ids[:5],
                'noun_top5_ids': noun_ids[:5],
                'confidence': conf_str,
                'narration': narration,
            }
    return data


def read_csv_file(csv_path: str) -> Dict[str, dict]:
    data = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = row['narration_id']
            try:
                verb_id = int(row['verb_class'].strip())
                noun_id = int(row['noun_class'].strip())
                data[nid] = {
                    'pred_verb_id': verb_id,
                    'pred_noun_id': noun_id,
                    'narration': row['narration'].strip(),
                }
            except (ValueError, KeyError):
                continue
    return data


def analyze_data_distribution(txt_data: Dict, csv_data: Dict):
    """分析两个数据集的分布"""
    common_ids = set(txt_data.keys()) & set(csv_data.keys())
    print(f"\n数据分布分析:")
    print(f"TXT 总样本: {len(txt_data)}")
    print(f"CSV 总样本: {len(csv_data)}")
    print(f"共同样本: {len(common_ids)}")

    # 分析 TXT 中的 ID 范围
    txt_verbs = []
    txt_nouns = []
    for nid in list(txt_data.keys())[:1000]:  # 采样分析
        txt_verbs.extend(txt_data[nid]['verb_top5_ids'][:1])  # 只取top1
        txt_nouns.extend(txt_data[nid]['noun_top5_ids'][:1])

    # 分析 CSV 中的 ID 范围
    csv_verbs = [csv_data[nid]['pred_verb_id'] for nid in list(csv_data.keys())[:1000]]
    csv_nouns = [csv_data[nid]['pred_noun_id'] for nid in list(csv_data.keys())[:1000]]

    print(f"\nTXT verb ID 范围: {min(txt_verbs)} - {max(txt_verbs)}")
    print(f"CSV verb ID 范围: {min(csv_verbs)} - {max(csv_verbs)}")
    print(f"TXT noun ID 范围: {min(txt_nouns)} - {max(txt_nouns)}")
    print(f"CSV noun ID 范围: {min(csv_nouns)} - {max(csv_nouns)}")

    # 检查最常见的ID
    print(f"\nTXT 最常见 verb IDs: {Counter(txt_verbs).most_common(5)}")
    print(f"CSV 最常见 verb IDs: {Counter(csv_verbs).most_common(5)}")
    print(f"TXT 最常见 noun IDs: {Counter(txt_nouns).most_common(5)}")
    print(f"CSV 最常见 noun IDs: {Counter(csv_nouns).most_common(5)}")


def compare_accuracy(txt_data: Dict, csv_data: Dict) -> Tuple:
    common_ids = set(txt_data.keys()) & set(csv_data.keys())
    total = len(common_ids)

    verb_top1 = noun_top1 = both_top1 = 0
    verb_top5 = noun_top5 = both_top5 = 0

    mismatch_examples = []
    match_examples = []

    for nid in common_ids:
        t = txt_data[nid]
        c = csv_data[nid]

        true_verb_top1 = t['verb_top5_ids'][0] if t['verb_top5_ids'] else -1
        true_noun_top1 = t['noun_top5_ids'][0] if t['noun_top5_ids'] else -1

        pred_verb = c['pred_verb_id']
        pred_noun = c['pred_noun_id']

        v1_match = (pred_verb == true_verb_top1)
        n1_match = (pred_noun == true_noun_top1)
        both1 = v1_match and n1_match

        v5_hit = pred_verb in t['verb_top5_ids']
        n5_hit = pred_noun in t['noun_top5_ids']
        both5 = v5_hit and n5_hit

        if v1_match: verb_top1 += 1
        if n1_match: noun_top1 += 1
        if both1: both_top1 += 1
        if v5_hit: verb_top5 += 1
        if n5_hit: noun_top5 += 1
        if both5: both_top5 += 1

        # 收集一些不匹配的例子用于分析
        if not both1 and len(mismatch_examples) < 5:
            mismatch_examples.append({
                'nid': nid,
                'txt_verb': true_verb_top1,
                'csv_verb': pred_verb,
                'txt_noun': true_noun_top1,
                'csv_noun': pred_noun,
                'verb_match': v1_match,
                'noun_match': n1_match
            })
        elif both1 and len(match_examples) < 3:
            match_examples.append({
                'nid': nid,
                'txt_verb': true_verb_top1,
                'csv_verb': pred_verb,
                'txt_noun': true_noun_top1,
                'csv_noun': pred_noun
            })

    print(f"\n匹配样本示例:")
    for example in match_examples:
        print(
            f"  {example['nid']}: TXT(v{example['txt_verb']},n{example['txt_noun']}) = CSV(v{example['csv_verb']},n{example['csv_noun']})")

    print(f"\n不匹配样本示例:")
    for example in mismatch_examples:
        print(
            f"  {example['nid']}: TXT(v{example['txt_verb']},n{example['txt_noun']}) vs CSV(v{example['csv_verb']},n{example['csv_noun']}) - verb_match:{example['verb_match']} noun_match:{example['noun_match']}")

    acc = lambda x: round(x / total * 100, 2) if total > 0 else 0.0

    return (
        both_top1, verb_top1, noun_top1, total,
        acc(both_top1), acc(verb_top1), acc(noun_top1),
        acc(verb_top5), acc(noun_top5), acc(both_top5)
    )


def main(txt_path: str, csv_path: str):
    print("正在读取 TXT 文件...")
    txt_data = read_txt_file(txt_path)
    print(f"读取 TXT 样本数: {len(txt_data)}")

    print("正在读取 CSV 文件...")
    csv_data = read_csv_file(csv_path)
    print(f"读取 CSV 样本数: {len(csv_data)}")

    # 分析数据分布
    analyze_data_distribution(txt_data, csv_data)

    (both_top1, verb_top1, noun_top1, total,
     both_top1_acc, verb_top1_acc, noun_top1_acc,
     verb_top5_acc, noun_top5_acc, both_top5_acc) = compare_accuracy(txt_data, csv_data)

    print("\n" + "=" * 60)
    print("           准确率统计")
    print("=" * 60)
    print(f"共同样本数          : {total}")
    print(f"Top-1 同时准确率    : {both_top1}/{total} = {both_top1_acc}%")
    print(f"Top-1 动词准确率    : {verb_top1}/{total} = {verb_top1_acc}%")
    print(f"Top-1 名词准确率    : {noun_top1}/{total} = {noun_top1_acc}%")
    print(f"Top-5 动词命中率    : {verb_top5_acc}%")
    print(f"Top-5 名词命中率    : {noun_top5_acc}%")
    print(f"Top-5 同时命中率    : {both_top5_acc}%")
    print("=" * 60)


"""============================================================
           准确率统计
============================================================
共同样本数          : 199
Top-1 同时准确率    : 26/199 = 13.07%
Top-1 动词准确率    : 73/199 = 36.68%
Top-1 名词准确率    : 50/199 = 25.13%
Top-5 动词命中率    : 62.81%
Top-5 名词命中率    : 43.22%
Top-5 同时命中率    : 28.64%
============================================================

============================================================
           准确率统计
============================================================
共同样本数          : 199
Top-1 同时准确率    : 22/199 = 11.06%
Top-1 动词准确率    : 73/199 = 36.68%
Top-1 名词准确率    : 48/199 = 24.12%
Top-5 动词命中率    : 60.3%
Top-5 名词命中率    : 41.21%
Top-5 同时命中率    : 27.14%
============================================================"""



"""           准确率统计
============================================================
共同样本数          : 199
Top-1 同时准确率    : 28/199 = 14.07%
Top-1 动词准确率    : 79/199 = 39.7%
Top-1 名词准确率    : 49/199 = 24.62%
Top-5 动词命中率    : 68.84%
Top-5 名词命中率    : 42.21%
Top-5 同时命中率    : 32.66%
============================================================
============================================================
           准确率统计
============================================================
共同样本数          : 199
Top-1 同时准确率    : 30/199 = 15.08%
Top-1 动词准确率    : 82/199 = 41.21%
Top-1 名词准确率    : 50/199 = 25.13%
Top-5 动词命中率    : 68.84%
Top-5 名词命中率    : 41.71%
Top-5 同时命中率    : 35.68%
============================================================
"""

"""============================================================
           准确率统计
============================================================
共同样本数          : 199
Top-1 同时准确率    : 30/199 = 15.08%
Top-1 动词准确率    : 84/199 = 42.21%
Top-1 名词准确率    : 52/199 = 26.13%
Top-5 动词命中率    : 68.84%
Top-5 名词命中率    : 43.22%
Top-5 同时命中率    : 32.16%
============================================================
"""

""""============================================================
           准确率统计
============================================================
共同样本数          : 199
Top-1 同时准确率    : 32/199 = 16.08%
Top-1 动词准确率    : 87/199 = 43.72%
Top-1 名词准确率    : 55/199 = 27.64%
Top-5 动词命中率    : 69.85%
Top-5 名词命中率    : 44.22%
Top-5 同时命中率    : 34.67%
============================================================
============================================================
           准确率统计
============================================================
共同样本数          : 199
Top-1 同时准确率    : 33/199 = 16.58%
Top-1 动词准确率    : 86/199 = 43.22%
Top-1 名词准确率    : 55/199 = 27.64%
Top-5 动词命中率    : 71.36%
Top-5 名词命中率    : 42.71%
Top-5 同时命中率    : 35.18%
============================================================
"""


"""
============================================================
           准确率统计
============================================================
共同样本数          : 199
Top-1 同时准确率    : 31/199 = 15.58%
Top-1 动词准确率    : 89/199 = 44.72%
Top-1 名词准确率    : 52/199 = 26.13%
Top-5 动词命中率    : 70.35%
Top-5 名词命中率    : 41.71%
Top-5 同时命中率    : 32.66%
============================================================
"""
"""
============================================================
           准确率统计
============================================================
共同样本数          : 199
Top-1 同时准确率    : 33/199 = 16.58%
Top-1 动词准确率    : 91/199 = 45.73%
Top-1 名词准确率    : 54/199 = 27.14%
Top-5 动词命中率    : 73.87%
Top-5 名词命中率    : 43.72%
Top-5 同时命中率    : 37.19%
============================================================
"""


# ================== 使用示例 ==================
if __name__ == "__main__":
    noun_csv = "/home/will/Mydata/EPIC-KITCHENS/EPIC_100_noun_classes.csv"  # 名词CSV路径（第一个CSV）
    verb_csv = "/home/will/Mydata/EPIC-KITCHENS/EPIC_100_verb_classes.csv"  # 动词CSV路径（第二个CSV）
    input_txt = "/home/will/Mycode/action_agent/t_300/test_comparison_results_frames8_threshold.txt"  # 输入TXT路径
    output_txt = "t_300/output_test.txt"  # 输出TXT路径
    process_text_file(input_txt, output_txt, verb_csv, noun_csv, 'unmapped_rows.txt')
    txt_path = "/home/will/Mycode/action_agent/t_300/output_test.txt"
    csv_path = "/home/will/Mydata/EPIC-KITCHENS/EPIC_100_validation.csv"
    main(txt_path, csv_path)