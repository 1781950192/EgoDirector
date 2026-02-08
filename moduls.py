import base64
import os
import json
import pandas as pd
from http import HTTPStatus
import dashscope
import numpy as np
from typing import List, Dict

from dashscope import Generation

from utils import image_to_base64, prepare_multimodal_message, compare_text_similarity_v3, retry_api_call, read_txt_file,extract_noun_probabilities

# 假设 DASHSCOPE_API_KEY 已设置在环境中
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', 'sk-e0055b4ba5284edcb2a8e0341c3cb748')

prompt_dict = {
    'combine_actions_prompt.txt': 'prompt/prompt_5/combine_actions_prompt.txt',
    'Conclusion_on_incorrect.txt': 'prompt/Conclusion_on_incorrect.txt',
    'judge_error_reason_prompt.txt': 'prompt/judge_error_reason_prompt.txt',
    'optimize_prompt_template.txt': 'prompt/optimize_prompt_template.txt',
    'select_actions_prompt.txt': 'prompt/prompt_5/select_actions_prompt.txt',
    'select_noun_prompt.txt': 'prompt/prompt_5/select_noun_prompt.txt'
}

# 从文件加载错题集
def load_wrong_set(file_path: str = 'json_epic/wrong_set.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            wrong_set = json.load(f)
    else:
        wrong_set = []
    return wrong_set

# 判断动作识别错误的原因的函数
@retry_api_call(max_attempts=3, delay=0)
def judge_error_reason(frames_urls: List[str], selected_reason: str, predicted_action: str, correct_action: str) -> str:
    template = read_txt_file(prompt_dict["judge_error_reason_prompt.txt"])  # 假设你有一个提示模板文件
    prompt = template.format(
        selected_reason=selected_reason,
        predicted_action=predicted_action,
        correct_action=correct_action
    )

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
            result = json.loads(content)
            error_reason = result.get("error_reason", "未知错误原因")
            return error_reason
        except json.JSONDecodeError:
            print(f"JSON 解析失败: {content}")
            return "未知错误原因"
    else:
        print(f"判断错误原因失败: {response.code}, {response.message}")
        return "未知错误原因"

# 利用错题集优化提示的函数
@retry_api_call(max_attempts=3, delay=0)
def optimize_prompts_using_wrong_set(wrong_set):
    template = read_txt_file(prompt_dict["optimize_prompt_template.txt"])
    noun_prompt = read_txt_file(prompt_dict["select_noun_prompt.txt"])
    combine_action_prompt = read_txt_file(prompt_dict["combine_actions_prompt.txt"])
    select_action_prompt = read_txt_file(prompt_dict["select_actions_prompt.txt"])
    prompt = template.format(error_log=wrong_set,noun_prompt=noun_prompt,combine_action_prompt=combine_action_prompt,select_action_prompt=select_action_prompt)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    response = Generation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key = "sk-xxx",
        api_key=DASHSCOPE_API_KEY,
        model="qwen3-max",
        messages=messages,
        result_format="message",
    )

    if response.status_code == HTTPStatus.OK:
        content = response.output["choices"][0]["message"]["content"]
        content = content.strip()
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        elif content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()
        conclusion = json.loads(content)
        return conclusion
    else:
        print(f"总结错题失败: {response.code}, {response.message}")
        return []

# 总结错误
@retry_api_call(max_attempts=3, delay=0)
def conclusion_incorrect(wrong_set) :
    template = read_txt_file(prompt_dict["Conclusion_on_incorrect.txt"])
    prompt = template.format(wrong_set=wrong_set)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    response = Generation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key = "sk-xxx",
        api_key=DASHSCOPE_API_KEY,
        model="qwen3-flash",
        messages=messages,
        result_format="message",
        timeout=(10, 600)
    )

    if response.status_code == HTTPStatus.OK:
        content = response.output["choices"][0]["message"]["content"]
        content = content.strip()
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        elif content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()
        conclusion = content
        return conclusion
    else:
        print(f"总结错题失败: {response.code}, {response.message}")
        return []


# 选择名词
@retry_api_call(max_attempts=3, delay=0)
def select_nouns(frames_urls: List[str], noun_keys: List[str]) -> List[str]:
    noun_list_str = ", ".join(noun_keys)
    template = read_txt_file(prompt_dict["select_noun_prompt.txt"])
    prompt = template.format(noun_list_str=noun_list_str)

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
def combine_actions(frames_urls: List[str], selected_nouns: List[Dict[str, str]], verb_keys: List[str],) -> List[str]:
    nouns_info = json.dumps(selected_nouns)
    verb_list_str = ", ".join(verb_keys)
    keys = [item['key'] for item in selected_nouns]
    n_v = extract_noun_probabilities(keys)

    template = read_txt_file(prompt_dict["combine_actions_prompt.txt"])
    prompt = template.format(nouns_info=nouns_info, verb_list_str=verb_list_str,noun_verb_list=n_v)

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
def select_actions(frames_urls: List[str], selected_verbs,) -> Dict[str, str]:
    verbs_info = json.dumps(selected_verbs)
    template = read_txt_file(prompt_dict["select_actions_prompt.txt"])
    prompt = template.format(verbs_info=verbs_info)

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
            if isinstance(selected, dict) and "action" in selected:
                return selected
            else:
                print(f"动作格式错误: {content}")
                return {"action": ""}
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {str(e)}, content: {content}")
            return {"action": ""}
    else:
        print(f"筛选动作失败: {response.code}, {response.message}")
        return {"action": ""}


# 主动作识别函数
@retry_api_call(max_attempts=3, delay=0)
def action_recognition(frames_urls: List[str], nouns_csv_path: str, verbs_csv_path: str,
                       max_iterations: int = 5) -> Dict[str, str]:
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

        selected_noun = select_nouns(frames_urls, noun_keys)
        selected_noun_keys = selected_noun["noun"]
        selected_noun_reason = selected_noun["reason"]

        if selected_noun_keys is None:
            print("选择名词失败，跳过本次迭代")
            continue

        print(f"挑选出来的名词是：{selected_noun_keys}")

        selected_nouns = nouns_df[nouns_df['key'].isin(selected_noun_keys)][['key', 'instances']].to_dict('records')

        selected_action_all = combine_actions(frames_urls, selected_nouns, verb_keys)
        selected_action = selected_action_all["action"]
        selected_action_reason = selected_action_all["reason"]

        if selected_action is None:
            print("选择动作失败，跳过本次迭代")
            continue
        print(f"挑选出来的动作是：{selected_action}")

        action_dict = select_actions(frames_urls, selected_action)
        if action_dict is None:
            print("筛选动作失败，跳过本次迭代")
            continue
        action = action_dict.get("action", "")
        print(f"输出的动作是：{action_dict}")

        if not action:
            print("未生成有效动作，继续迭代")
            continue

        return action_dict,selected_noun_reason,selected_action_reason

    return action_dict if action_dict else {"action": ""}

if __name__ == '__main__':
    with open("json_epic/wrong_set.json", 'r', encoding='utf-8') as json_wrong:
        data = json.load(json_wrong)
        out = conclusion_incorrect(data)
        out = optimize_prompts_using_wrong_set(out)
        print(out["select_noun_prompt"])
        print(out["combine_actions_prompt"])
        print(out["select_actions_prompt"])


