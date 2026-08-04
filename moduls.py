import base64
import os
import json
import re

import pandas as pd
from http import HTTPStatus
import numpy as np
from typing import List, Dict
import torch

from utils import image_to_base64, prepare_multimodal_message, compare_text_similarity_v3, retry_api_call, \
    read_txt_file

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import time

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/mnt/data/xgl/qwen_vl_8b",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda:0",
)


processor = AutoProcessor.from_pretrained(
    "/mnt/data/xgl/qwen_vl_8b",
    trust_remote_code=True
)

@torch.no_grad()
def qwen3_vl_local(
    messages: List[Dict],
    max_new_tokens: int = 1024,
) -> str:
    global model, processor

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # generated_ids = model.generate(
    #     **inputs,
    #     max_new_tokens=max_new_tokens,
    #     # temperature=0.0,
    #     # top_p=0.01,
    #     do_sample=False
    # )

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # 关键：关闭采样，使用贪婪解码
        temperature=1.0,          # do_sample=False 时此参数会被忽略，但写上无害
        top_p=1.0,                # 同上
        top_k=0,                  # 0 表示不启用 top_k 过滤
        num_beams=1,              # 1 表示不使用 beam search（beam search 也完全确定性）
        repetition_penalty=1.0, # 可选，设为1.0表示不惩罚重复
    )
    

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Step 5: 立即清理
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()
    print(output_text[0])
    return output_text[0]



def prepare_image_messages(image_urls: List[str], text_prompt: str) -> List[Dict]:
    """把帧URL列表 + 文字提示 → 标准 messages 格式"""
    content = []
    for url in image_urls: 
        content.append({"type": "image", "image": url})
    content.append({"type": "text", "text": text_prompt})
    
    return [{"role": "user", "content": content}]

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
    "reflector_prompt.txt": 'prompt/Reflector_prompt/Reflector_prompt.txt',
    'select_actions_prompt.txt': 'prompt/select_actions_prompt.txt',
    'select_noun_prompt.txt': 'prompt/select_noun_prompt.txt',
    'base.txt':'prompt/base.txt'
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


def pre_noun_add(key):
    template = read_txt_file('prompt/pre_prompt/pre_noun.txt')
    prompt = template.format(noun=key)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    return qwen3_vl_local(messages, max_new_tokens=1024)

def verb_noun_add(key):
    template = read_txt_file('prompt/pre_prompt/verb_noun.txt')
    prompt = template.format(noun=key)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    return json.loads(qwen3_vl_local(messages, max_new_tokens=1024))



# 反思器
@retry_api_call(max_attempts=3, delay=0)
def reflector_action(frames_urls: List[str], selected_noun, selected_action, action_dict, noun_keys, noun_verb, pre_nouns) -> List[str]:
    template = read_txt_file(prompt_dict["reflector_prompt.txt"])
    prompt = template.format(noun_reason=selected_noun, action_reason=selected_action,
                             select_action_reason=action_dict, noun_list=noun_keys,
                             noun_verb=noun_verb, pre_nouns=pre_nouns)
    
    messages = prepare_image_messages(frames_urls, prompt)
    result = qwen3_vl_local(messages, max_new_tokens=1024)
    return json.loads(result)


# 选择名词
@retry_api_call(max_attempts=3, delay=0)
def select_nouns(frames_urls: List[str], noun_keys: List[str],reflect,select_frame) -> List[str]:
    noun_list_str = ", ".join(noun_keys)
    template = read_txt_file(prompt_dict["select_noun_prompt.txt"])
    prompt = template.format(noun_list_str=noun_list_str,reflect=reflect,select_frame=select_frame)

    messages = prepare_image_messages(frames_urls, prompt)
    result = qwen3_vl_local(messages, max_new_tokens=512)
    
    return json.loads(result)



@retry_api_call(max_attempts=3, delay=0)
def combine_actions(frames_urls, selected_nouns, noun_verb, reflect,select_frame):

    template = read_txt_file(prompt_dict["combine_actions_prompt.txt"])
    prompt = template.format(nouns_info=selected_nouns, noun_verb_list=noun_verb,
                            reflect=reflect,select_frame=select_frame)

    messages = prepare_image_messages(frames_urls, prompt)
    result = qwen3_vl_local(messages, max_new_tokens=512)
    
    return json.loads(result)


@retry_api_call(max_attempts=3, delay=0)
def select_actions(frames_urls: List[str], selected_action, pre_noun, reflect,select_frame) -> Dict[
    str, str]:
    verbs_info = selected_action
    template = read_txt_file(prompt_dict["select_actions_prompt.txt"])
    prompt = template.format(verbs_info=verbs_info,
                             pre_noun=pre_noun, reflect=reflect,select_frame=select_frame)
    messages = prepare_image_messages(frames_urls, prompt)
    result = qwen3_vl_local(messages, max_new_tokens=1024)
    
    return json.loads(result)

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
    

def update_pre_noun_frequencies(selected_noun_keys: List[str], max_size: int = 1000):
    """
    更新 pre_noun.json 中选中名词的 frequency，并在总数超过 max_size 时删除最低频项。

    Args:
        selected_noun_keys (List[str]): 当前选中的名词列表。
        max_size (int): pre_noun.json 允许的最大条目数，默认为 1000。
    """
    file_path = 'context/pre_noun.json'

    # 读取当前数据
    if not os.path.exists(file_path):
        print(f"[警告] {file_path} 不存在，跳过频率更新。")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            pre_noun_list = json.load(f)
    except:
        pre_noun_list = []
    # 构建 key -> item 映射（自动去重，保留最后一个同名项）
    key_to_item = {item["key"]: item for item in pre_noun_list}
    # 更新选中名词的频率（仅对已存在的名词）
    updated_count = 0
    for noun in selected_noun_keys:
        if noun in key_to_item:
            key_to_item[noun]["frequency"] += 1
            updated_count += 1

    # 转回列表
    updated_list = list(key_to_item.values())

    # 如果超出最大容量，移除最低频项（可循环删除直到满足条件）
    while len(updated_list) > max_size:
        min_item = min(updated_list, key=lambda x: x["frequency"])
        updated_list.remove(min_item)

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(updated_list, f, ensure_ascii=False, indent=2)

    return updated_list


# 主动作识别函数
def action_recognition(frames_urls: List[str], utils, use_playbook ,max_iterations: int = 5):
    with open('context/verb_noun.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    noun_keys = data.keys()
    
    frames_urls_average = frames_urls

    for iteration in range(max_iterations):
        print(f"迭代 {iteration + 1}")
        start_time = time.time()
        i = 0
        while True:
            selected_noun = select_nouns(frames_urls_average, noun_keys, utils["one"],utils['four'])
            selected_noun_keys = selected_noun["noun"]
            i = i + 1
            if len(selected_noun_keys) == 5 or i >= 5:
                break
        print(f"挑选名词的时间是{time.time()-start_time:2f}秒")

        start_time = time.time()
        pre_noun = update_pre_noun_frequencies(selected_noun_keys, max_size=1000)
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

        actions = combine_actions(frames_urls_average, pre_nouns, noun_verb, utils["two"],utils['four'])
        print(f"组合动作的时间为：{time.time()-start_time:2f}秒")
        start_time = time.time()

        if actions is None:
            print("选择动作失败，跳过本次迭代")
            continue
        print(f"挑选出来的动作是：{actions}")

        action_dict = select_actions(frames_urls_average, actions, pre_nouns, utils["three"],utils['four'])
        print(f"动作评分的时间为：{time.time() - start_time:2f}秒")
        start_time = time.time()

        if use_playbook:
            reflect_dict = reflector_action(frames_urls_average, selected_noun, action_dict, actions, noun_keys, noun_verb, pre_nouns)
            print(f"动作反思的时间为：{time.time()-start_time:2f}秒")
        else:
            reflect_dict = None

        return reflect_dict, action_dict, selected_noun_keys


# 主动作识别函数
def action_recognition_base2(frames_urls: List[str], utils ,max_iterations: int = 5):
    noun_keys = []
    
    frames_urls_average = frames_urls

    for iteration in range(max_iterations):
        print(f"迭代 {iteration + 1}")
        start_time = time.time()
        i = 0
        while True:
            selected_noun = select_nouns(frames_urls_average, noun_keys, utils["one"],utils['four'])
            selected_noun_keys = selected_noun["noun"]
            i = i + 1
            if len(selected_noun_keys) == 5 or i >= 5:
                break
        print(f"挑选名词的时间是{time.time()-start_time:2f}秒")

        start_time = time.time()
        print(f"挑选出来的名词是：{selected_noun_keys}")

        pre_nouns=selected_noun
        noun_verb=None

        actions = combine_actions(frames_urls_average, pre_nouns, noun_verb, utils["two"],utils['four'])
        print(f"组合动作的时间为：{time.time()-start_time:2f}秒")
        start_time = time.time()

        print(f"挑选出来的动作是：{actions}")

        action_dict = select_actions(frames_urls_average, actions, pre_nouns, utils["three"],utils['four'])
        print(f"动作评分的时间为：{time.time() - start_time:2f}秒")
        start_time = time.time()

        reflect_dict = None

        return reflect_dict, action_dict, selected_noun_keys


def action_recognition_multi_turn(
    frames_urls: List[str],
    utils,
    use_playbook,
    max_iterations: int = 1
) -> tuple:
    # 加载模板和名词列表（保持不变）
    select_noun_template = read_txt_file(prompt_dict["select_noun_prompt.txt"])
    combine_actions_template = read_txt_file(prompt_dict["combine_actions_prompt.txt"])
    select_actions_template = read_txt_file(prompt_dict["select_actions_prompt.txt"])
    reflector_template = read_txt_file(prompt_dict["reflector_prompt.txt"])

    with open('context/verb_noun.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    noun_keys = list(data.keys())
    noun_list_str = ", ".join(noun_keys)

    # ==================== 第1轮：带图片的选择名词 ====================
    step1_prompt = select_noun_template.format(
        noun_list_str=noun_list_str,
        reflect=utils.get("one", "")
    )
    messages = prepare_image_messages(frames_urls, step1_prompt)  # content 是 list[dict]，含图片

    print("Step 1: 选择名词...")
    step1_output = qwen3_vl_local(messages, max_new_tokens=1024)

    step1 = json.loads(step1_output)
    selected_noun_keys = step1["noun"]
    i = 0
    while len(selected_noun_keys) != 5:  # 如果你还有强制5个的逻辑，可保留重试
        i = i + 1
        step1_output = qwen3_vl_local(messages, max_new_tokens=1024)
        step1 = json.loads(step1_output)
        selected_noun_keys = step1["noun"]
        if i >= 10:
            break

    print(f"选中的名词: {selected_noun_keys}")

    # ==================== 构建多轮对话历史（从这里开始不传图片） ====================
    # 重新初始化 messages：第一轮带图 + assistant回复 + 后续纯文本
    messages.append({
        "role": "assistant", 
        "content": [{"type": "text", "text": step1_output}]
        })

    # ==================== 第2轮：组合动作 ====================
    noun_verb = extract_noun_probabilities(selected_noun_keys)

    step2_prompt = combine_actions_template.format(
        nouns_info=selected_noun_keys,   # 或传入更详细的 pre_nouns
        noun_verb_list=noun_verb,
        reflect=utils.get("two", "")
    )

    # 注意：这里 content 是字符串，不是 list
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": step2_prompt}]
    })

    print("Step 2: 组合动作...")
    step2_output = qwen3_vl_local(messages, max_new_tokens=1024)

    actions = json.loads(step2_output)  # 假设返回 {"actions": [...]} 或直接是列表
    if isinstance(actions, dict):
        actions = actions.get("actions", actions)

    print(f"组合出的动作: {actions}")

    # 添加 assistant 回复
    messages.append({"role": "assistant", "content": [{"type": "text", "text": step2_output}]})

    # ==================== 第3轮：动作评分/选择 ====================
    step3_prompt = select_actions_template.format(
        verbs_info=actions,
        pre_noun=selected_noun_keys,
        reflect=utils.get("three", "")
    )

    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": step3_prompt}]
    })

    print("Step 3: 动作评分...")
    step3_output = qwen3_vl_local(messages, max_new_tokens=1024)

    action_dict = json.loads(step3_output)

    # 添加 assistant 回复（为后续反思准备）
    messages.append({"role": "assistant", "content": [{"type": "text", "text": step3_output}]})

    # ==================== 第4轮：反思器（可选） ====================
    reflect_dict = None
    if utils.get("use_playbook", True):  # 或根据你的判断
        reflector_prompt = reflector_template.format(
            noun_reason=", ".join(selected_noun_keys),
            action_reason=actions,
            select_action_reason=action_dict,
            noun_list=noun_list_str,
            noun_verb=noun_verb,
            pre_nouns=selected_noun_keys
        )

        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": reflector_prompt}]
        })

        print("Step 4: 反思...")
        reflect_output = qwen3_vl_local(messages, max_new_tokens=8192)

        try:
            reflect_dict = json.loads(reflect_output)
        except:
            print("反思解析失败")
            reflect_dict = None

    return reflect_dict, action_dict, selected_noun_keys



def action_recognition_base1(frames_urls,utils):
    prompt = read_txt_file(prompt_dict["base.txt"])
    messages = prepare_image_messages(frames_urls, prompt)
    result = qwen3_vl_local(messages, max_new_tokens=1024)
    out_all = json.loads(result)
    nouns = out_all['nouns']
    
    return None,json.loads(result),nouns


# 测试模型和函数是否真的能跑多图
if __name__ == "__main__":
    urls = [
        "/mnt/HHD/xgl/mydata/ek100_val/P02/rgb_frames/P02_12/frame_0000000041.jpg",
        "/mnt/HHD/xgl/mydata/ek100_val/P02/rgb_frames/P02_12/frame_0000000078.jpg",
        "/mnt/HHD/xgl/mydata/ek100_val/P02/rgb_frames/P02_12/frame_0000000098.jpg",
        "/mnt/HHD/xgl/mydata/ek100_val/P02/rgb_frames/P02_12/frame_00000000124.jpg",
        "/mnt/HHD/xgl/mydata/ek100_val/P02/rgb_frames/P02_12/frame_00000000150.jpg",
    ] 

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": urls[0]},
                {"type": "image", "image": urls[1]},
                {"type": "image", "image": urls[2]},
                {"type": "text", "text": "这几张图里描述了什么"}
            ]
        }
    ]

    # messages = [
    #     {"role": "user", "content": [{"type": "text", "text": "你是谁"}]}
    # ]

    output = qwen3_vl_local(messages, max_new_tokens=2048)
    print("输出：")
    print(output)


