import json

# 1. 读取 JSON 文件
with open('gtea_process/noun_verb_probs.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 提取每个食材对应的动词（忽略数值），转为列表
result = {ingredient: list(actions.keys()) for ingredient, actions in data.items()}

with open('gtea_process/verb_noun.json', 'w', encoding='utf-8') as out_f:
    json.dump(result, out_f, indent=2, ensure_ascii=False)