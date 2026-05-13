import base64
import os
import json
import re

import pandas as pd
from http import HTTPStatus
import dashscope
import numpy as np
from typing import List, Dict

from dashscope import Generation

from utils import image_to_base64, prepare_multimodal_message, compare_text_similarity_v3, retry_api_call, \
    read_txt_file, extract_noun_probabilities

# 假设 DASHSCOPE_API_KEY 已设置在环境中
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', 'sk-e0055b4ba5284edcb2a8e0341c3cb748')

prompt_dict = {
    'combine_actions_prompt.txt': 'prompt/egtea_prompt/combine_actions_prompt.txt',
    "reflector_prompt.txt": 'prompt/Reflector_prompt/Reflector_prompt.txt',
    'select_actions_prompt.txt': 'prompt/egtea_prompt/select_actions_prompt.txt',
    'select_noun_prompt.txt': 'prompt/egtea_prompt/select_noun_prompt.txt'
}


def load_keys_from_numbered_file(file_path):
    """从 '文本 数字' 格式的文件中提取文本部分（支持文本含空格）"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    keys = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 从行尾匹配一个数字，前面是任意非数字内容（至少一个字符）
        match = re.match(r'^(.+?)\s+\d+$', line)
        if match:
            keys.append(match.group(1).strip())
        else:
            # 如果没有编号，就把整行当作 key（容错）
            keys.append(line)
    return keys

# 从文件加载错题集
def load_wrong_set(file_path: str = 'json_epic/wrong_set.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            wrong_set = json.load(f)
    else:
        wrong_set = []
    return wrong_set

# 反思器
@retry_api_call(max_attempts=3, delay=0)
def reflector_action(frames_urls: List[str], noun_reason, action_reason, select_action_reason, noun_list, verb_list,
                     noun_verb, pre_noun) -> List[str]:
    template = read_txt_file(prompt_dict["reflector_prompt.txt"])
    prompt = template.format(noun_reason=noun_reason, action_reason=action_reason,
                             select_action_reason=select_action_reason, noun_list=noun_list, verb_list=verb_list,
                             noun_verb=noun_verb, pre_noun=pre_noun)
    messages = prepare_multimodal_message(frames_urls, prompt)
    response = dashscope.MultiModalConversation.call(
        api_key=DASHSCOPE_API_KEY,
        model="qwen-vl-max-latest",
        messages=messages
    )

    if response.status_code == HTTPStatus.OK:
        content = response.output["choices"][0]["message"]["content"][0]["text"]
        content = content.strip()
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        elif content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()
        selected = json.loads(content)
        return selected
    else:
        print(f"反思动作失败: {response.code}, {response.message}")
        return []


# 选择名词
@retry_api_call(max_attempts=3, delay=0)
def select_nouns(frames_urls: List[str], noun_keys: List[str], expand_noun=None,reflect=None) -> List[str]:
    noun_list_str = ", ".join(noun_keys)
    template = read_txt_file(prompt_dict["select_noun_prompt.txt"])
    prompt = template.format(noun_list_str=noun_list_str, expand_noun=expand_noun,reflect=reflect)

    messages = prepare_multimodal_message(frames_urls, prompt)
    response = dashscope.MultiModalConversation.call(
        api_key=DASHSCOPE_API_KEY,
        model="qwen-vl-max-latest",
        messages=messages
    )

    if response.status_code == HTTPStatus.OK:
        content = response.output["choices"][0]["message"]["content"][0]["text"]
        content = content.strip()
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        elif content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()
        selected = json.loads(content)
        return selected
    else:
        print(f"选择名词失败: {response.code}, {response.message}")
        return []


@retry_api_call(max_attempts=3, delay=0)
def combine_actions(frames_urls, selected_nouns, noun_verb, verb_keys, noun_reason, reflect):
    nouns_info = selected_nouns
    verb_list_str = ", ".join(verb_keys)

    template = read_txt_file(prompt_dict["combine_actions_prompt.txt"])
    prompt = template.format(nouns_info=nouns_info, verb_list_str=verb_list_str, noun_verb_list=noun_verb,
                            reflect=reflect)

    frames_urls = frames_urls[0:8]
    messages = prepare_multimodal_message(frames_urls, prompt)
    response = dashscope.MultiModalConversation.call(
        api_key=DASHSCOPE_API_KEY,
        model="qwen-vl-max-latest",
        messages=messages
    )
    if response.status_code == HTTPStatus.OK:
        content = response.output["choices"][0]["message"]["content"][0]["text"]
        content = content.strip()
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        elif content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()
        selected = json.loads(content)
        return selected
    else:
        print(f"选择动作失败: {response.code}, {response.message}")
        return []


@retry_api_call(max_attempts=3, delay=0)
def select_actions(frames_urls: List[str], selected_action, action_reason, noun_reason, pre_noun, reflect) -> Dict[
    str, str]:
    verbs_info = selected_action
    template = read_txt_file(prompt_dict["select_actions_prompt.txt"])
    prompt = template.format(verbs_info=verbs_info, action_reason=action_reason, noun_reason=noun_reason,
                             pre_noun=pre_noun, reflect=reflect)
    messages = prepare_multimodal_message(frames_urls, prompt)
    response = dashscope.MultiModalConversation.call(
        api_key=DASHSCOPE_API_KEY,
        model="qwen-vl-max-latest",
        messages=messages
    )
    if response.status_code == HTTPStatus.OK:
        content = response.output["choices"][0]["message"]["content"][0]["text"]
        content = content.strip()
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        elif content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()
        try:
            selected = json.loads(content)
            if isinstance(selected, dict) and "actions" in selected:
                return selected
            else:
                print(f"动作格式错误: {content}")
                return {"actions": ""}
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {str(e)}, content: {content}")
            return {"actions": ""}
    else:
        print(f"筛选动作失败: {response.code}, {response.message}")
        return {"actions": ""}


# 主动作识别函数
def action_recognition(frames_urls: List[str], nouns_csv_path: str, verbs_csv_path: str, utils,
                       max_iterations: int = 5):
    name, extension = os.path.splitext(nouns_csv_path)
    if extension == ".csv":
        try:
            nouns_df = pd.read_csv(nouns_csv_path)
            verbs_df = pd.read_csv(verbs_csv_path)
        except Exception as e:
            print(f"读取 CSV 文件出错: {str(e)}")
            return {"action": ""}

        noun_keys = nouns_df['key'].tolist()
        verb_keys = verbs_df['key'].tolist()
        for iteration in range(max_iterations):
            print(f"迭代 {iteration + 1}")

            with open('json_epic/expand_noun.json', 'r', encoding='utf-8') as f:
                expand_noun = json.load(f)
            selected_noun = select_nouns(frames_urls, noun_keys, expand_noun, utils["one"])
            selected_noun_keys = selected_noun["noun"]

            while True:
                if len(selected_noun_keys) == 5:
                    break
                selected_noun = select_nouns(frames_urls, noun_keys, expand_noun, utils["one"])

            selected_noun_reason = selected_noun["reason"]

            if selected_noun_keys is None:
                print("选择名词失败，跳过本次迭代")
                continue

            print(f"挑选出来的名词是：{selected_noun_keys}")

            with open('json_epic/pre_noun.json', 'r', encoding='utf-8') as f:
                pre_noun = json.load(f)  # 注意是 load（不是 loads）
            pre_nouns = {}
            # 一次性构建字典映射
            key_to_text = {item["key"]: item["generated_text"] for item in pre_noun}

            # 后续查询 O(1) 时间复杂度
            for noun in selected_noun_keys:
                pre_nouns[noun] = key_to_text.get(noun)

            keys = [item for item in pre_nouns.keys()]
            noun_verb = extract_noun_probabilities(keys)

            selected_action_all = combine_actions(frames_urls, pre_nouns, noun_verb, verb_keys, selected_noun_reason,
                                                  utils["two"])
            selected_action = selected_action_all["action"]
            selected_action_reason = selected_action_all["reason"]

            if selected_action is None:
                print("选择动作失败，跳过本次迭代")
                continue
            print(f"挑选出来的动作是：{selected_action}")

            action_dict = select_actions(frames_urls, selected_action, selected_action_reason, selected_noun_reason,
                                         pre_nouns, utils["three"])
            print("动作评分完成")

            # reflect_dict = reflector_action(frames_urls, selected_noun, selected_action_all, action_dict, noun_keys,
            #                                 verb_keys, noun_verb, pre_nouns)
            reflect_dict = None
            print("动作反思完成")

            if action_dict is None:
                print("筛选动作失败，跳过本次迭代")
                continue
            actions = action_dict.get("actions", "")

            if not actions:
                print("未生成有效动作，继续迭代")
                continue

            return reflect_dict, action_dict, selected_noun_reason, selected_action_reason, selected_noun_keys

        return action_dict if action_dict else {"actions": ""}
    elif extension == ".txt":
        # 读取文件，假设列之间用空格分隔
        noun_keys = load_keys_from_numbered_file(nouns_csv_path)
        verb_keys = load_keys_from_numbered_file(verbs_csv_path)
        for iteration in range(max_iterations):
            print(f"迭代 {iteration + 1}")

            selected_noun = select_nouns(frames_urls, noun_keys, utils["one"])
            selected_noun_keys = selected_noun["noun"]

            while True:
                if len(selected_noun_keys) == 5:
                    break
                selected_noun = select_nouns(frames_urls, noun_keys, utils["one"])

            selected_noun_reason = selected_noun["reason"]

            if selected_noun_keys is None:
                print("选择名词失败，跳过本次迭代")
                continue

            print(f"挑选出来的名词是：{selected_noun_keys}")

            with open('json_epic/pre_noun_egtea.json', 'r', encoding='utf-8') as f:
                pre_noun = json.load(f)  # 注意是 load（不是 loads）
            pre_nouns = {}
            # 一次性构建字典映射
            key_to_text = {item["key"]: item["generated_text"] for item in pre_noun}

            # 后续查询 O(1) 时间复杂度
            for noun in selected_noun_keys:
                pre_nouns[noun] = key_to_text.get(noun)

            keys = [item for item in pre_nouns.keys()]
            noun_verb = extract_noun_probabilities(keys,json_file_path='json_epic/verb_noun_egtea.json')

            selected_action_all = combine_actions(frames_urls, pre_nouns, noun_verb, verb_keys, selected_noun_reason,
                                                  utils["two"])
            selected_action = selected_action_all["action"]
            selected_action_reason = selected_action_all["reason"]

            if selected_action is None:
                print("选择动作失败，跳过本次迭代")
                continue
            print(f"挑选出来的动作是：{selected_action}")

            action_dict = select_actions(frames_urls, selected_action, selected_action_reason, selected_noun_reason,
                                         pre_nouns, utils["three"])
            print("动作评分完成")

            # reflect_dict = reflector_action(frames_urls, selected_noun, selected_action_all, action_dict, noun_keys,
            #                                 verb_keys, noun_verb, pre_nouns)
            reflect_dict = None
            print("动作反思完成")

            if action_dict is None:
                print("筛选动作失败，跳过本次迭代")
                continue
            actions = action_dict.get("actions", "")

            if not actions:
                print("未生成有效动作，继续迭代")
                continue

            return reflect_dict, action_dict, selected_noun_reason, selected_action_reason,selected_noun_keys

        return action_dict if action_dict else {"actions": ""}

