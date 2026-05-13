import json
import os
import dashscope
import csv
from utils import read_txt_file

# 配置
MAX_RETRIES = 10          # 最大重试次数
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', 'sk-e0055b4ba5284edcb2a8e0341c3cb748')
dashscope.api_key = DASHSCOPE_API_KEY  # 推荐全局设置一次

results = []

# 读取 JSON 文件
with open('/home/will/Mycode/action_agent/json_epic/verb_noun_egtea.json', 'r', encoding='utf-8') as f:
    noun_verb = json.load(f)

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
# output_path = '/home/will/Mycode/action_agent/json_epic/expand_noun.json'
# with open(output_path, 'w', encoding='utf-8') as f:
#     json.dump(results, f, ensure_ascii=False, indent=2)
#
# print(f"✅ 共处理 {len(results)} 条记录，结果已保存至 {output_path}")


with open('/home/will/Mydata/EGTEA++/EGTEA/Action_Annotations/noun_idx.txt', mode='r', encoding='utf-8') as file:
    results = []

    for line in file:
        line = line.strip()
        if not line:  # 跳过空行
            continue

        # 解析每行，假设格式是 "noun id"
        parts = line.split()
        if len(parts) < 2:
            print(f"跳过无效行: {line}")
            continue

        # 最后一个部分是ID，其余部分是名词（可能有空格的名词用下划线连接）
        key = ' '.join(parts[:-1]).replace('_', ' ')  # 将下划线替换为空格
        id = parts[-1]
        print(f"Processing key: {key}, id: {id}")

        # 构造 prompt
        template = read_txt_file('/home/will/Mycode/action_agent/prompt/pre_prompt/pre_noun.txt')
        try:
            # 注意：现在没有row字典了，直接使用key变量
            verb_info = noun_verb.get(key, "暂无相关动词概率信息")
            prompt = template.format(
                noun=key,
                instances="",  # txt文件中没有instances字段，根据你的需求调整
                noun_verb=verb_info
            )
        except KeyError as e:
            print(f"跳过 key={key}：缺少字段 {e}")
            continue

        # 重试逻辑
        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                messages = [
                    {'role': 'system', 'content': '你是一位自视角动作识别领域的专家'},
                    {'role': 'user', 'content': prompt}
                ]
                response = dashscope.Generation.call(
                    model="qwen-max",
                    messages=messages,
                    result_format='message'
                )

                if response.status_code == 200:
                    text_output = response.output.choices[0].message.content
                    result_item = {
                        "key": key,
                        "id": id,
                        "generated_text": text_output
                    }
                    print(result_item)
                    results.append(result_item)
                    success = True
                    break  # 成功则跳出重试循环
                else:
                    print(f"API 返回错误（尝试 {attempt}/{MAX_RETRIES}）: {response.code} - {response.message}")
            except Exception as e:
                print(f"请求异常（尝试 {attempt}/{MAX_RETRIES}）: {type(e).__name__}: {e}")

            # 如果不是最后一次尝试，可以加一点延迟再重试
            if attempt < MAX_RETRIES:
                import time

                time.sleep(2 ** attempt)  # 指数退避：2s, 4s, 8s...

        if not success:
            print(f"❌ 最终失败：key={key}，已跳过")

# 保存结果
output_path = '/home/will/Mycode/action_agent/json_epic/pre_noun_egtea.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ 共处理 {len(results)} 条记录，结果已保存至 {output_path}")