import pandas as pd
from http import HTTPStatus
import numpy as np
from utils import image_to_base64, retry_api_call, read_txt_file, extract_and_load_json

import base64
import json
import os
import re
import time
from typing import List, Dict

import openai

# ====================== vLLM Client 配置 ======================
# 注意：请根据实际 vLLM 服务端口修改
# 当前系统 vLLM 端口：8000 (Qwen3-VL-8B-Instruct)
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",   # 修改为实际端口
    api_key="EMPTY"
)

# 模型名称必须是 vLLM 中注册的完整路径
MODEL_NAME = "Qwen3-VL-8B-Instruct"
# Qwen3-VL-8B-Instruct
# gemma3-12b
#cpm
# internvl

# ====================== 工具函数 ======================
def image_to_base64(file_path: str) -> str | None:
    """将本地图片转为 data:url base64 格式"""
    try:
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        print(f"× 加载图片失败 {file_path}: {e}")
        return None
        
def prepare_image_messages(image_paths: List[str], text_prompt: str) -> List[Dict]:
    """构造单轮带多图的 messages（兼容 OpenAI Vision 格式）"""
    content = []
    for path in image_paths:
        b64 = image_to_base64(path)
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": b64}
            })
    content.append({
        "type": "text",
        "text": text_prompt
    })
    return [{"role": "user", "content": content}]

import re

def remove_think_tags(text):
    # 模式：匹配 <think> 和 </think> 以及之间的所有内容（非贪婪匹配）
    pattern = r'<think>.*?</think>'
    # 将匹配到的部分替换为空字符串
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned

def qwen3_vl_vllm(
    messages: List[Dict],
    max_tokens: int = 1024,
    temperature: float = 0.0
) -> str:
    """统一调用 vLLM 的确定性推理接口"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,      # 0.0 → 贪婪解码，完全确定性
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            seed=42,                      # 固定种子，进一步保证确定性（vLLM ≥0.5.0 支持）
        #     extra_body={
        #     "stop_token_ids": [1, 151645],
        #     "chat_template_kwargs": {"enable_thinking": False},
        # }
        )
        content = response.choices[0].message.content.strip()
        # cleaned = remove_think_tags(content)
        print(content)  # 保留你原来的打印习惯
        return content
    except Exception as e:
        print(f"vLLM API 调用错误：{type(e).__name__}: {str(e)}")
        raise  # 重新抛出异常，让重试装饰器处理


def extract_noun_probabilities(noun_list, json_file_path='context/verb_noun.json'):
    """
    Extract verb usage probabilities for a given list of nouns from a JSON file.

    Args:
        noun_list (list): List of noun names (e.g., ['tap', 'spoon', 'plate'])
        json_file_path (str): Path to the JSON file containing noun-verb probabilities

    Returns:
        dict: Dictionary mapping each noun to its verb probabilities
              (or empty dict for nouns not found)
    """
    # Load the JSON file
    with open(json_file_path, 'r') as f:
        noun_verb_probabilities = json.load(f)

    # Initialize result dictionary
    result = {}

    # Process each noun in the input list
    for noun in noun_list:
        # Get probabilities for the noun, or empty dict if not found
        result[noun] = noun_verb_probabilities.get(noun, {})

    return result


prompt_dict = {
    'combine_actions_prompt.txt': 'prompt/combine_actions_prompt.txt',
    'combine_actions_prompt_2llm.txt': 'prompt/combine_actions_prompt_2llm.txt',
    "reflector_prompt.txt": 'prompt/Reflector_prompt/Reflector_prompt.txt',
    'select_actions_prompt.txt': 'prompt/select_actions_prompt.txt',
    'select_noun_prompt.txt': 'prompt/select_noun_prompt.txt',
    'base.txt':'prompt/base.txt'
}

def pre_noun_add(key):
    template = read_txt_file('prompt/pre_prompt/pre_noun.txt')
    prompt = template.format(noun=key)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    return qwen3_vl_vllm(messages, max_tokens=2048)

def verb_noun_add(key):
    template = read_txt_file('prompt/pre_prompt/verb_noun.txt')
    prompt = template.format(noun=key)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    return extract_and_load_json(qwen3_vl_vllm(messages, max_tokens=2048))



@retry_api_call(max_attempts=3, delay=0)  # 你原来的装饰器保持可用
def reflector_action(frames_urls: List[str], selected_noun, selected_action, action_dict, pre_nouns) -> List[str]:
    template = read_txt_file(prompt_dict["reflector_prompt.txt"])
    prompt = template.format(noun_reason=selected_noun, action_reason=selected_action,
                             select_action_reason=action_dict, pre_nouns=pre_nouns)

    messages = prepare_image_messages(frames_urls, prompt)
    result = qwen3_vl_vllm(messages, max_tokens=2048)
    return extract_and_load_json(result)


@retry_api_call(max_attempts=3, delay=0)
def select_nouns(frames_urls: List[str], noun_keys: List[str], reflect) -> List[str]:
    noun_list_str = ", ".join(noun_keys)
    template = read_txt_file(prompt_dict["select_noun_prompt.txt"])
    prompt = template.format(noun_list_str=noun_list_str, reflect=reflect)

    messages = prepare_image_messages(frames_urls, prompt)
    result = qwen3_vl_vllm(messages, max_tokens=2048)
    return extract_and_load_json(result)


@retry_api_call(max_attempts=3, delay=0)
def combine_actions(frames_urls, selected_nouns, noun_verb, reflect,llm=3):
    if llm == 3:
        template = read_txt_file(prompt_dict["combine_actions_prompt.txt"])
    elif llm == 2:
        template = read_txt_file(prompt_dict["combine_actions_prompt_2llm.txt"])
    prompt = template.format(nouns_info=selected_nouns, noun_verb_list=noun_verb,
                             reflect=reflect)

    messages = prepare_image_messages(frames_urls, prompt)
    result = qwen3_vl_vllm(messages, max_tokens=2048)
    return extract_and_load_json(result)


@retry_api_call(max_attempts=3, delay=0)
def select_actions(frames_urls: List[str], selected_action, pre_noun, reflect) -> Dict[str, str]:
    template = read_txt_file(prompt_dict["select_actions_prompt.txt"])
    prompt = template.format(verbs_info=selected_action,
                             pre_noun=pre_noun, reflect=reflect)

    messages = prepare_image_messages(frames_urls, prompt)
    result = qwen3_vl_vllm(messages, max_tokens=2048)
    return extract_and_load_json(result)

def update_json_file(filename, new_data):
    """更新JSON文件的辅助函数"""
    """当为空时，设置为[]"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data.extend(new_data)  # 使用extend批量添加
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except:
        data = []
        data.extend(new_data)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def update_verb_noun_json(filename: str, new_entries: List[dict]):
    """
    更新 verb_noun.json（dict 格式），添加新名词的动词列表
    new_entries: 列表，每个元素是 { "noun": ["verb1", ...] }
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print("[警告] verb_noun.json 不是 dict 格式，重新初始化")
            data = {}
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    # 更新或添加
    for entry in new_entries:
        data.update(entry)  # entry 是 {noun: verbs}

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[成功] 已更新 {len(new_entries)} 个名词到 {filename}")
    

# 主动作识别函数
def action_recognition(frames_urls: List[str], utils, use_playbook ,max_iterations: int = 5):
    with open('context/verb_noun.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    noun_keys = data.keys()

    for iteration in range(max_iterations):
        print(f"迭代 {iteration + 1}")
        start_time = time.time()
        i = 0
        while True:
            selected_noun = select_nouns(frames_urls, "None", None)
            selected_noun_keys = selected_noun["noun"]
            i = i + 1
            if len(selected_noun_keys) == 5 or i >= 5:
                break
        print(f"挑选名词的时间是{time.time()-start_time:2f}秒")

        start_time = time.time()
        unknown_noun = [noun for noun in selected_noun_keys if noun not in noun_keys]
        if unknown_noun:
            pre_noun_updates = []
            verb_noun_updates = []
            
            for item in unknown_noun:
                # 收集pre_noun更新
                pre_noun_updates.append({
                    "key": item,
                    "generated_text": pre_noun_add(item),
                    "frequency": 1
                })
                
                verb_entry = verb_noun_add(item)  # 返回 {noun: [verbs]}
                verb_noun_updates.append(verb_entry)

            # 批量更新 dict 格式的 json
            update_verb_noun_json('context/verb_noun.json', verb_noun_updates)
            # 批量更新文件
            update_json_file('context/pre_noun.json', pre_noun_updates)
            

        if selected_noun_keys is None:
            print("选择名词失败，跳过本次迭代")
            continue

        print(f"更新上下文的时间为：{time.time()-start_time:2f}秒")
        start_time = time.time()
        print(f"挑选出来的名词是：{selected_noun_keys}")

        with open('context/pre_noun.json', 'r', encoding='utf-8') as f:
            pre_noun = json.load(f)  # 注意是 load（不是 loads）
        pre_nouns = {}
        # 一次性构建字典映射
        key_to_text = {item["key"]: item["generated_text"] for item in pre_noun}

        # 后续查询 O(1) 时间复杂度
        for noun in selected_noun_keys:
            pre_nouns[noun] = key_to_text.get(noun)

        keys = [item for item in pre_nouns.keys()]
        noun_verb = extract_noun_probabilities(keys)

        actions = combine_actions(frames_urls, pre_nouns, noun_verb, None)
        print(f"组合动作的时间为：{time.time()-start_time:2f}秒")
        start_time = time.time()

        if actions is None:
            print("选择动作失败，跳过本次迭代")
            continue
        print(f"挑选出来的动作是：{actions}")

        action_dict = select_actions(frames_urls, actions, pre_nouns, None)
        print(f"动作评分的时间为：{time.time() - start_time:2f}秒")
        start_time = time.time()

        reflect_dict = None

        return reflect_dict, action_dict, selected_noun_keys


# ====================== 效率实验版本（带详细计时）======================
def action_recognition_with_timing(frames_urls: List[str], utils, use_playbook, max_iterations: int = 5):
    """
    带详细计时的动作识别函数，用于效率实验
    返回: (reflect_dict, action_dict, selected_noun_keys, timing_info)
    timing_info 包含各阶段的耗时
    """
    with open('context/verb_noun.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    noun_keys = data.keys()
    
    # 初始化计时字典
    timing_info = {
        'select_noun_time': 0.0,
        'knowledge_base_time': 0.0,
        'combine_actions_time': 0.0,
        'score_actions_time': 0.0
    }
    
    for iteration in range(max_iterations):
        print(f"迭代 {iteration + 1}")
        
        # === 1. 选名词阶段计时 ===
        start_time = time.time()
        i = 0
        while True:
            selected_noun = select_nouns(frames_urls, "None", None)
            selected_noun_keys = selected_noun["noun"]
            i = i + 1
            if len(selected_noun_keys) == 5 or i >= 5:
                break
        select_noun_elapsed = time.time() - start_time
        timing_info['select_noun_time'] = select_noun_elapsed
        print(f"挑选名词的时间是{select_noun_elapsed:.2f}秒")

        # === 2. 知识库检索/更新阶段计时 ===
        start_time = time.time()
        unknown_noun = [noun for noun in selected_noun_keys if noun not in noun_keys]
        if unknown_noun:
            pre_noun_updates = []
            verb_noun_updates = []
            
            for item in unknown_noun:
                # 收集pre_noun更新
                pre_noun_updates.append({
                    "key": item,
                    "generated_text": pre_noun_add(item),
                    "frequency": 1
                })
                
                verb_entry = verb_noun_add(item)  # 返回 {noun: [verbs]}
                verb_noun_updates.append(verb_entry)

            # 批量更新 dict 格式的 json
            update_verb_noun_json('context/verb_noun.json', verb_noun_updates)
            # 批量更新文件
            update_json_file('context/pre_noun.json', pre_noun_updates)
        
        knowledge_base_elapsed = time.time() - start_time
        timing_info['knowledge_base_time'] = knowledge_base_elapsed
        print(f"更新上下文的时间为：{knowledge_base_elapsed:.2f}秒")

        if selected_noun_keys is None:
            print("选择名词失败，跳过本次迭代")
            continue

        print(f"挑选出来的名词是：{selected_noun_keys}")

        with open('context/pre_noun.json', 'r', encoding='utf-8') as f:
            pre_noun = json.load(f)
        pre_nouns = {}
        # 一次性构建字典映射
        key_to_text = {item["key"]: item["generated_text"] for item in pre_noun}

        # 后续查询 O(1) 时间复杂度
        for noun in selected_noun_keys:
            pre_nouns[noun] = key_to_text.get(noun)

        keys = [item for item in pre_nouns.keys()]
        noun_verb = extract_noun_probabilities(keys)

        # === 3. 组合动作阶段计时 ===
        start_time = time.time()
        actions = combine_actions(frames_urls, pre_nouns, noun_verb, None)
        combine_actions_elapsed = time.time() - start_time
        timing_info['combine_actions_time'] = combine_actions_elapsed
        print(f"组合动作的时间为：{combine_actions_elapsed:.2f}秒")

        if actions is None:
            print("选择动作失败，跳过本次迭代")
            continue
        print(f"挑选出来的动作是：{actions}")

        # === 4. 动作评分阶段计时 ===
        start_time = time.time()
        action_dict = select_actions(frames_urls, actions, pre_nouns, None)
        score_actions_elapsed = time.time() - start_time
        timing_info['score_actions_time'] = score_actions_elapsed
        print(f"动作评分的时间为：{score_actions_elapsed:.2f}秒")
        
        reflect_dict = None

        return reflect_dict, action_dict, selected_noun_keys, timing_info


# 主动作识别函数
def action_recognition_base2(frames_urls: List[str], utils ,max_iterations: int = 5):
    noun_keys = []

    for iteration in range(max_iterations):
        print(f"迭代 {iteration + 1}")
        start_time = time.time()
        i = 0
        while True:
            selected_noun = select_nouns(frames_urls, noun_keys, None)
            selected_noun_keys = selected_noun["noun"]
            i = i + 1
            if len(selected_noun_keys) == 5 or i >= 5:
                break
        print(f"挑选名词的时间是{time.time()-start_time:2f}秒")

        start_time = time.time()
        print(f"挑选出来的名词是：{selected_noun_keys}")

        pre_nouns=selected_noun
        noun_verb=None

        actions = combine_actions(frames_urls, pre_nouns, noun_verb, None,llm=2)
        print(f"组合动作的时间为：{time.time()-start_time:2f}秒")
        start_time = time.time()

        print(f"挑选出来的动作是：{actions}")

        reflect_dict = None

        return reflect_dict, actions, selected_noun_keys


# 主动作识别函数
def action_recognition_base3(frames_urls: List[str], utils ,max_iterations: int = 5):
    noun_keys = []

    for iteration in range(max_iterations):
        print(f"迭代 {iteration + 1}")
        start_time = time.time()
        i = 0
        while True:
            selected_noun = select_nouns(frames_urls, noun_keys, None)
            selected_noun_keys = selected_noun["noun"]
            i = i + 1
            if len(selected_noun_keys) == 5 or i >= 5:
                break
        print(f"挑选名词的时间是{time.time()-start_time:2f}秒")

        start_time = time.time()
        print(f"挑选出来的名词是：{selected_noun_keys}")

        pre_nouns=selected_noun
        noun_verb=None


        actions = combine_actions(frames_urls, pre_nouns, noun_verb, None)
        print(f"组合动作的时间为：{time.time()-start_time:2f}秒")
        start_time = time.time()

        print(f"挑选出来的动作是：{actions}")

        action_dict = select_actions(frames_urls, actions, pre_nouns, None)
        print(f"动作评分的时间为：{time.time() - start_time:2f}秒")
        start_time = time.time()

        reflect_dict = None

        return reflect_dict, action_dict, selected_noun_keys



# def action_recognition_base1(frames_urls,utils):
#     prompt = read_txt_file(prompt_dict["base.txt"])
#     messages = prepare_image_messages(frames_urls, prompt)
#     result = qwen3_vl_vllm(messages, max_tokens=1024)
#     out_all = extract_and_load_json(result)
#     nouns = out_all['nouns']
    
#     return None,extract_and_load_json(result),nouns


# 主动作识别函数
def action_recognition_base1(frames_urls: List[str], utils ,max_iterations: int = 5):
    noun_keys = []

    for iteration in range(max_iterations):
        print(f"迭代 {iteration + 1}")
        start_time = time.time()
        pre_nouns=None
        noun_verb=None
        actions = combine_actions(frames_urls, pre_nouns, noun_verb, None,llm=2)
        print(f"组合动作的时间为：{time.time()-start_time:2f}秒")
        start_time = time.time()

        print(f"挑选出来的动作是：{actions}")

        reflect_dict = None

        return reflect_dict, actions, None


if __name__ == "__main__":
    urls = [
        "/mnt/data/xgl/mydata/ek100/P02/rgb_frames/P02_12/frame_0000000041.jpg",
        "/mnt/data/xgl/mydata/ek100/P02/rgb_frames/P02_12/frame_0000000078.jpg",
        "/mnt/data/xgl/mydata/ek100/P02/rgb_frames/P02_12/frame_0000000098.jpg",
        "/mnt/data/xgl/mydata/ek100/P02/rgb_frames/P02_12/frame_0000000124.jpg",
        "/mnt/data/xgl/mydata/ek100/P02/rgb_frames/P02_12/frame_0000000150.jpg",
    ]

    # 简单单轮测试：描述图片
    messages = prepare_image_messages(urls[:5], "请详细描述这几张图片中发生了什么动作，涉及哪些物体？")
    output = qwen3_vl_vllm(messages, max_tokens=2048)
    print("\n【单轮测试结果】")
    print(output)


