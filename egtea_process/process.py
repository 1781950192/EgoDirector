import json
from collections import defaultdict


def analyze_verb_noun_probabilities(file_paths):
    # 定义动词和名词的映射
    verbs = {
        1: "Inspect/Read", 2: "Open", 3: "Take", 4: "Cut", 5: "Turn on",
        6: "Put", 7: "Operate", 8: "Close", 9: "Move Around", 10: "Wash",
        11: "Spread", 12: "Turn off", 13: "Divide/Pull Apart", 14: "Clean/Wipe",
        15: "Mix", 16: "Pour", 17: "Compress", 18: "Crack", 19: "Squeeze"
    }

    nouns = {
        1: "recipe", 2: "fridge", 3: "eating_utensil", 4: "tomato", 5: "faucet",
        6: "cabinet", 7: "condiment_container", 8: "cucumber", 9: "stove", 10: "carrot",
        11: "onion", 12: "drawer", 13: "plate", 14: "bowl", 15: "trash",
        16: "trash_container", 17: "bell_pepper", 18: "cooking_utensil", 19: "paper_towel", 20: "bacon",
        21: "condiment", 22: "bread", 23: "pan", 24: "lettuce", 25: "patty",
        26: "pot", 27: "fridge_drawer", 28: "hand", 29: "seasoning_container", 30: "cup",
        31: "counter", 32: "bread_container", 33: "cutting_board", 34: "sponge", 35: "dishwasher",
        36: "cheese_container", 37: "oil_container", 38: "mixture", 39: "tomato_container", 40: "cheese",
        41: "oil", 42: "pasta_container", 43: "olive", 44: "salad", 45: "pasta",
        46: "grocery_bag", 47: "seasoning", 48: "egg", 49: "water", 50: "sandwich",
        51: "washing_liquid", 52: "microwave", 53: "strainer"
    }

    # 初始化数据结构
    noun_verb_counts = defaultdict(lambda: defaultdict(int))  # noun -> verb -> count
    noun_counts = defaultdict(int)  # 名词出现总次数
    verb_counts = defaultdict(int)  # 动词出现总次数
    total_samples = 0

    # 处理所有文件
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) >= 4:
                        total_samples += 1

                        # 解析标签：第3列是动词，第4列及以后是名词
                        verb_num = int(parts[2])
                        noun_nums = [int(n) for n in parts[3:]]

                        # 验证并转换ID
                        if verb_num in verbs:
                            verb_name = verbs[verb_num]
                            verb_counts[verb_name] += 1

                            # 处理所有名词标签
                            for noun_num in noun_nums:
                                if noun_num in nouns:
                                    noun_name = nouns[noun_num]
                                    noun_counts[noun_name] += 1
                                    noun_verb_counts[noun_name][verb_name] += 1

        except FileNotFoundError:
            print(f"警告：找不到文件 {file_path}，跳过")
        except Exception as e:
            print(f"处理文件 {file_path} 时出错：{e}")

    # 计算两种概率：
    # 1. P(verb|noun) - 给定名词时动词的条件概率
    # 2. P(noun|verb) - 给定动词时名词的条件概率

    verb_given_noun_prob = {}  # P(verb|noun)
    noun_given_verb_prob = {}  # P(noun|verb)

    # 计算 P(verb|noun)
    for noun, verb_counts_dict in noun_verb_counts.items():
        total_for_noun = noun_counts[noun]
        noun_probabilities = {}

        for verb, count in verb_counts_dict.items():
            probability = count / total_for_noun
            noun_probabilities[verb] = probability

        # 按概率从高到低排序
        verb_given_noun_prob[noun] = dict(sorted(noun_probabilities.items(),
                                                 key=lambda x: x[1], reverse=True))

    # 计算 P(noun|verb)
    for verb, count in verb_counts.items():
        total_for_verb = count
        verb_probabilities = {}

        for noun in nouns.values():
            if noun in noun_verb_counts and verb in noun_verb_counts[noun]:
                probability = noun_verb_counts[noun][verb] / total_for_verb
                verb_probabilities[noun] = probability

        # 按概率从高到低排序，只保留概率大于0的
        sorted_probs = {k: v for k, v in sorted(verb_probabilities.items(),
                                                key=lambda x: x[1], reverse=True) if v > 0}
        noun_given_verb_prob[verb] = sorted_probs

    return {
        'verb_given_noun': verb_given_noun_prob,  # P(verb|noun)
        'noun_given_verb': noun_given_verb_prob,  # P(noun|verb)
        'noun_counts': dict(noun_counts),
        'verb_counts': dict(verb_counts),
        'total_samples': total_samples
    }


def save_probabilities_json(probabilities_dict, output_file):
    """保存概率结果到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(probabilities_dict, f, indent=4, ensure_ascii=False)


def print_statistics(results):
    """打印统计信息"""
    total_samples = results['total_samples']
    verb_given_noun = results['verb_given_noun']
    noun_given_verb = results['noun_given_verb']

    print(f"总样本数: {total_samples}")
    print(f"唯一名词数量: {len(verb_given_noun)}")
    print(f"唯一动词数量: {len(noun_given_verb)}")

    print("\n各名词的动作分布 (P(verb|noun)):")
    print("=" * 80)

    for noun, verb_probs in list(verb_given_noun.items())[:10]:  # 只显示前10个
        total_actions = sum(verb_probs.values())
        action_count = len(verb_probs)
        top_verb, top_prob = next(iter(verb_probs.items()))

        print(
            f"{noun:20} | 动作数: {action_count:2d} | 最高概率: {top_verb:15} ({top_prob:.3f}) | 总出现: {results['noun_counts'][noun]:4d}")

    print(f"\n各动词的名词分布 (P(noun|verb)):")
    print("=" * 80)

    for verb, noun_probs in list(noun_given_verb.items())[:10]:  # 只显示前10个
        noun_count = len(noun_probs)
        top_noun, top_prob = next(iter(noun_probs.items()))

        print(
            f"{verb:20} | 名词数: {noun_count:2d} | 最高概率: {top_noun:15} ({top_prob:.3f}) | 总出现: {results['verb_counts'][verb]:4d}")


# 使用示例
if __name__ == "__main__":
    # 定义要处理的文件列表
    file_paths = ["train_split1.txt", "train_split2.txt", "train_split3.txt"]

    # 检查文件是否存在
    import os

    existing_files = [f for f in file_paths if os.path.exists(f)]

    if not existing_files:
        print("错误：未找到任何输入文件")
        print("请确保以下文件存在：")
        for f in file_paths:
            print(f"  - {f}")
    else:
        print(f"找到 {len(existing_files)} 个文件，开始处理...")

        # 分析概率
        results = analyze_verb_noun_probabilities(existing_files)

        # 输出统计信息
        print_statistics(results)

        # 保存为JSON文件
        output_json = "verb_noun_egtea_complete.json"
        save_probabilities_json(results, output_json)
        print(f"\n完整概率结果已保存到: {output_json}")

        # 单独保存 P(verb|noun) 用于您的应用
        output_verb_given_noun = "verb_noun_egtea.json"
        save_probabilities_json(results['verb_given_noun'], output_verb_given_noun)
        print(f"条件概率 P(verb|noun) 已保存到: {output_verb_given_noun}")