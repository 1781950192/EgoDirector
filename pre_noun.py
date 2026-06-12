import json
import os
import dashscope
import csv
from utils import read_txt_file
from dashscope import Generation

# 配置
MAX_RETRIES = 10          # 最大重试次数
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', 'sk-e0055b4ba5284edcb2a8e0341c3cb748')
dashscope.api_key = DASHSCOPE_API_KEY  # 推荐全局设置一次

results = []


# with open('/home/will/Mydata/EPIC-KITCHENS/EPIC_100_noun_classes.csv', mode='r', encoding='utf-8') as file:
#     csv_reader = csv.DictReader(file)
#     for row in csv_reader:
#         key = row['key']
#         id = row['id']
#         print(f"Processing key: {key}")
#
#         # 构造 prompt
#         template = read_txt_file('/home/will/Mycode/action_agent/prompt/pre_prompt/expand_noun.txt')
#         try:
#             # 注意：row 是 dict，不能直接 format，应传具体字段
#             verb_info = noun_verb.get(row['key'], "暂无相关动词概率信息")
#             prompt = template.format(
#                 noun=row['key'],
#                 instances=row['instances'],
#                 noun_verb=verb_info
#             )
#         except KeyError as e:
#             print(f"跳过 key={key}：缺少字段 {e}")
#             continue
#
#         # 重试逻辑
#         success = False
#         for attempt in range(1, MAX_RETRIES + 1):
#             try:
#                 messages = [
#                     {'role': 'system', 'content': '你是一位自视角动作识别领域的专家'},
#                     {'role': 'user', 'content': prompt}
#                 ]
#                 response = dashscope.Generation.call(
#                     model="qwen-plus",
#                     messages=messages,
#                     result_format='message'
#                 )
#
#                 if response.status_code == 200:
#                     text_output = response.output.choices[0].message.content
#                     result_item = {
#                         "key": key,
#                         "id": id,
#                         "generated_text": text_output
#                     }
#                     print(result_item)
#                     results.append(result_item)
#                     success = True
#                     break  # 成功则跳出重试循环
#                 else:
#                     print(f"API 返回错误（尝试 {attempt}/{MAX_RETRIES}）: {response.code} - {response.message}")
#             except Exception as e:
#                 print(f"请求异常（尝试 {attempt}/{MAX_RETRIES}）: {type(e).__name__}: {e}")
#
#             # 如果不是最后一次尝试，可以加一点延迟再重试
#             if attempt < MAX_RETRIES:
#                 import time
#                 time.sleep(2 ** attempt)  # 指数退避：2s, 4s, 8s...
#
#         if not success:
#             print(f"❌ 最终失败：key={key}，已跳过")
#
# # 保存结果
# output_path = '/home/will/Mycode/action_agent/json_epic/expand_noun_ek100.json'
# with open(output_path, 'w', encoding='utf-8') as f:
#     json.dump(results, f, ensure_ascii=False, indent=2)
#
# print(f"✅ 共处理 {len(results)} 条记录，结果已保存至 {output_path}")

with open('charadesEgo_process/object_verb_probabilities.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    keys = data.keys()
    for key in keys:
        verb = data[key]
        template = read_txt_file('prompt/pre_prompt/pre_noun.txt')
        # 注意：现在没有row字典了，直接使用key变量
        prompt = template.format(
            noun=key,
            verb=verb
        )
        messages = [
            {"role": "system", "content": '你是一位自视角动作识别领域的专家。'},
            {"role": "user", "content": prompt},
        ]
        response = Generation.call(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key = "sk-xxx",
            api_key='sk-e0055b4ba5284edcb2a8e0341c3cb748',
            model="qwen3-max",
            messages=messages,
            result_format="message",
        )
        if response.status_code == 200:
            text_output = response.output.choices[0].message.content
            result_item = {
                "key": key,
                "generated_text": text_output
            }
            results.append(result_item)
output_path = 'charadesEgo_process/pre_noun_charadesEgo.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)


# import json
#
# # 文件路径（请根据实际情况修改）
# original_file = 'context/pre_noun.json'
# new_entries_file = 'pre_noun_gtea.json'
# output_file = 'context/pre_noun.json'  # 可以设为 original_file 来覆盖原文件
#
# # 1. 读取原始 JSON 文件
# with open(original_file, 'r', encoding='utf-8') as f:
#     original_data = json.load(f)
#
# # 2. 读取新增 JSON 文件
# with open(new_entries_file, 'r', encoding='utf-8') as f:
#     new_data = json.load(f)
#
# # 3. 构建原始 key 集合（用于快速去重）
# existing_keys = set(item['key'] for item in original_data if isinstance(item, dict) and 'key' in item)
#
# # 4. 遍历新数据，只添加 key 不重复的项
# for item in new_data:
#     if isinstance(item, dict) and 'key' in item:
#         if item['key'] not in existing_keys:
#             original_data.append(item)
#             existing_keys.add(item['key'])  # 防止 new_data 内部也有重复
#     else:
#         print(f"警告：跳过无效条目（缺少 'key' 字段）: {item}")
#
# # 5. 写入合并后的结果
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(original_data, f, ensure_ascii=False, indent=2)
#
# print(f"合并完成！共新增 {len(original_data) - len(existing_keys) + len(set(item['key'] for item in new_data if isinstance(item, dict) and 'key' in item))} 项（实际新增数量可能因重复而减少）")
