# import json
# import csv
# import ast

# def extract_nouns_from_csv(csv_file):
#     """从CSV文件中提取所有名词"""
#     nouns_set = set()
    
#     with open(csv_file, 'r', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             # 提取key字段
#             nouns_set.add(row['key'])
    
#     return nouns_set

# def extract_nouns_from_json(json_file):
#     """从JSON文件中提取所有名词"""
#     nouns_set = set()
    
#     with open(json_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         for item in data:
#             # 提取key字段
#             nouns_set.add(item['key'])
    
#     return nouns_set

# def compare_nouns(csv_file, json_file):
#     """比较两个文件中的名词"""
#     print("正在提取名词...")
    
#     # 提取名词
#     csv_nouns = extract_nouns_from_csv(csv_file)
#     json_nouns = extract_nouns_from_json(json_file)
    
#     print(f"CSV文件中的名词数量: {len(csv_nouns)}")
#     print(f"JSON文件中的名词数量: {len(json_nouns)}")
    
#     # 找出在CSV中但不在JSON中的名词
#     missing_nouns = csv_nouns - json_nouns
    
#     print(f"\n在CSV文件中但不在JSON文件中的名词数量: {len(missing_nouns)}")
    
#     if missing_nouns:
#         print("\n缺失的名词列表:")
#         for i, noun in enumerate(sorted(missing_nouns), 1):
#             print(f"{i}. {noun}")
    
#     # 找出两个文件共有的名词
#     common_nouns = csv_nouns & json_nouns
#     print(f"\n两个文件共有的名词数量: {len(common_nouns)}")
    
#     # 找出只在JSON中的名词
#     extra_nouns = json_nouns - csv_nouns
#     print(f"只在JSON文件中的名词数量: {len(extra_nouns)}")
    
#     return {
#         'csv_nouns': csv_nouns,
#         'json_nouns': json_nouns,
#         'missing_nouns': missing_nouns,
#         'common_nouns': common_nouns,
#         'extra_nouns': extra_nouns
#     }

# # 使用示例
# if __name__ == "__main__":
#     csv_filename = "/mnt/HHD/xgl/mydata/ek100_val/EPIC_100_noun_classes.csv"  # 替换为你的CSV文件路径
#     json_filename = "/mnt/HHD/xgl/mycode/action_agent/context_base/pre_noun.json"  # 替换为你的JSON文件路径
    
#     results = compare_nouns(csv_filename, json_filename)
    
#     print(results)

# import json

# # 读取文件
# with open('/mnt/HHD/xgl/mycode/action_agent/json_epic/pre_noun_gtea.json', 'r', encoding='utf-8') as f:
#     data1 = json.load(f)

# with open('/mnt/HHD/xgl/mycode/action_agent/context_base/pre_noun.json', 'r', encoding='utf-8') as f:
#     data2 = json.load(f)

# # 获取第二个文件中已有的key
# existing_keys = {item['key'] for item in data2}

# # 过滤并处理第一个文件的数据
# new_items = []
# for item in data1:
#     if item['key'] not in existing_keys:
#         new_item = {
#             'key': item['key'],
#             'generated_text': item['generated_text']
#         }
#         new_items.append(new_item)

# # 合并数据
# merged_data = data2 + new_items

# # 保存结果
# with open('11111.json', 'w', encoding='utf-8') as f:
#     json.dump(merged_data, f, ensure_ascii=False, indent=2)

# print(f"合并完成！新增了 {len(new_items)} 个条目")


# import json

# with open('/mnt/HHD/xgl/mycode/action_agent/context_base/pre_noun.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)


# # 过滤并处理第一个文件的数据
# new_items = []
# for item in data:
#     new_item = {
#         'key': item['key'],
#         "frequency": 1,
#         'generated_text': item['generated_text']
#     }
#     new_items.append(new_item)

# # 保存结果
# with open('11111.json', 'w', encoding='utf-8') as f:
#     json.dump(new_items, f, ensure_ascii=False, indent=2)


# def find_missing_nouns(nouns_to_check, noun_list):
#     """
#     找出不在名词列表中的名词
#     """
#     missing_nouns = [noun for noun in nouns_to_check if noun not in noun_list]
#     return missing_nouns

# # 示例用法
# noun_list = ["bowl", "cup", "spoon", "knife", "plate"]
# nouns_to_check = ["bowl", "fork", "spoon"]

# missing = find_missing_nouns(nouns_to_check, noun_list)
# print(f"缺失的名词: {missing}")  # 输出: ['fork']
# print(f"所有名词都在列表中: {len(missing) == 0}")  # 输出: False

# 简洁版本
try:
    import torch
    print(f"PyTorch {torch.__version__} 可用")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch 未安装")