import json
import time
from typing import List, Dict

from utils import retry_api_call, read_txt_file, extract_and_load_json
from vllm_client import prepare_image_messages, qwen3_vl_vllm
from prompts import prompt_dict, pre_noun_add, verb_noun_add
from context_manager import (
    extract_noun_probabilities,
    update_json_file,
    update_verb_noun_json,
)


@retry_api_call(max_attempts=3, delay=0)
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
def combine_actions(frames_urls, selected_nouns, noun_verb, reflect, llm=3):
    if llm == 3:
        template = read_txt_file(prompt_dict["combine_actions_prompt.txt"])
    elif llm == 2:
        template = read_txt_file(prompt_dict["combine_actions_prompt_2llm.txt"])
    prompt = template.format(nouns_info=selected_nouns, noun_verb_list=noun_verb, reflect=reflect)
    messages = prepare_image_messages(frames_urls, prompt)
    result = qwen3_vl_vllm(messages, max_tokens=2048)
    return extract_and_load_json(result)


@retry_api_call(max_attempts=3, delay=0)
def select_actions(frames_urls: List[str], selected_action, pre_noun, reflect) -> Dict[str, str]:
    template = read_txt_file(prompt_dict["select_actions_prompt.txt"])
    prompt = template.format(verbs_info=selected_action, pre_noun=pre_noun, reflect=reflect)
    messages = prepare_image_messages(frames_urls, prompt)
    result = qwen3_vl_vllm(messages, max_tokens=2048)
    return extract_and_load_json(result)


def action_recognition(frames_urls: List[str], max_iterations: int = 5):
    with open('context/verb_noun.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    noun_keys = data.keys()

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}")
        start_time = time.time()
        i = 0
        while True:
            selected_noun = select_nouns(frames_urls, "None", None)
            selected_noun_keys = selected_noun["noun"]
            i = i + 1
            if len(selected_noun_keys) == 5 or i >= 5:
                break
        print(f"Noun selection took {time.time() - start_time:2f}s")

        start_time = time.time()
        unknown_noun = [noun for noun in selected_noun_keys if noun not in noun_keys]
        if unknown_noun:
            pre_noun_updates = []
            verb_noun_updates = []
            for item in unknown_noun:
                pre_noun_updates.append({
                    "key": item,
                    "generated_text": pre_noun_add(item),
                    "frequency": 1
                })
                verb_entry = verb_noun_add(item)
                verb_noun_updates.append(verb_entry)
            update_verb_noun_json('context/verb_noun.json', verb_noun_updates)
            update_json_file('context/pre_noun.json', pre_noun_updates)

        if selected_noun_keys is None:
            print("Noun selection failed, skip this iteration")
            continue

        print(f"Context update took {time.time() - start_time:2f}s")
        start_time = time.time()
        print(f"Selected nouns: {selected_noun_keys}")

        with open('context/pre_noun.json', 'r', encoding='utf-8') as f:
            pre_noun = json.load(f)
        pre_nouns = {}
        key_to_text = {item["key"]: item["generated_text"] for item in pre_noun}
        for noun in selected_noun_keys:
            pre_nouns[noun] = key_to_text.get(noun)

        keys = [item for item in pre_nouns.keys()]
        noun_verb = extract_noun_probabilities(keys)

        actions = combine_actions(frames_urls, pre_nouns, noun_verb, None)
        print(f"Action combination took {time.time() - start_time:2f}s")
        start_time = time.time()

        if actions is None:
            print("Action selection failed, skip this iteration")
            continue
        print(f"Selected actions: {actions}")

        action_dict = select_actions(frames_urls, actions, pre_nouns, None)
        print(f"Action scoring took {time.time() - start_time:2f}s")
        start_time = time.time()

        reflect_dict = None
        return reflect_dict, action_dict, selected_noun_keys


def action_recognition_with_timing(frames_urls: List[str], max_iterations: int = 5):
    """Action recognition function with detailed timing, used for the efficiency experiment."""
    with open('context/verb_noun.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    noun_keys = data.keys()

    timing_info = {
        'select_noun_time': 0.0,
        'knowledge_base_time': 0.0,
        'combine_actions_time': 0.0,
        'score_actions_time': 0.0
    }

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}")

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
        print(f"Noun selection took {select_noun_elapsed:.2f}s")

        start_time = time.time()
        unknown_noun = [noun for noun in selected_noun_keys if noun not in noun_keys]
        if unknown_noun:
            pre_noun_updates = []
            verb_noun_updates = []
            for item in unknown_noun:
                pre_noun_updates.append({
                    "key": item,
                    "generated_text": pre_noun_add(item),
                    "frequency": 1
                })
                verb_entry = verb_noun_add(item)
                verb_noun_updates.append(verb_entry)
            update_verb_noun_json('context/verb_noun.json', verb_noun_updates)
            update_json_file('context/pre_noun.json', pre_noun_updates)

        knowledge_base_elapsed = time.time() - start_time
        timing_info['knowledge_base_time'] = knowledge_base_elapsed
        print(f"Context update took {knowledge_base_elapsed:.2f}s")

        if selected_noun_keys is None:
            print("Noun selection failed, skip this iteration")
            continue

        print(f"Selected nouns: {selected_noun_keys}")

        with open('context/pre_noun.json', 'r', encoding='utf-8') as f:
            pre_noun = json.load(f)
        pre_nouns = {}
        key_to_text = {item["key"]: item["generated_text"] for item in pre_noun}
        for noun in selected_noun_keys:
            pre_nouns[noun] = key_to_text.get(noun)

        keys = [item for item in pre_nouns.keys()]
        noun_verb = extract_noun_probabilities(keys)

        start_time = time.time()
        actions = combine_actions(frames_urls, pre_nouns, noun_verb, None)
        combine_actions_elapsed = time.time() - start_time
        timing_info['combine_actions_time'] = combine_actions_elapsed
        print(f"Action combination took {combine_actions_elapsed:.2f}s")

        if actions is None:
            print("Action selection failed, skip this iteration")
            continue
        print(f"Selected actions: {actions}")

        start_time = time.time()
        action_dict = select_actions(frames_urls, actions, pre_nouns, None)
        score_actions_elapsed = time.time() - start_time
        timing_info['score_actions_time'] = score_actions_elapsed
        print(f"Action scoring took {score_actions_elapsed:.2f}s")

        reflect_dict = None
        return reflect_dict, action_dict, selected_noun_keys, timing_info


def action_recognition_base1(frames_urls: List[str], max_iterations: int = 5):
    noun_keys = []
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}")
        start_time = time.time()
        pre_nouns = None
        noun_verb = None
        actions = combine_actions(frames_urls, pre_nouns, noun_verb, None, llm=2)
        print(f"Action combination took {time.time() - start_time:2f}s")
        start_time = time.time()
        print(f"Selected actions: {actions}")
        reflect_dict = None
        return reflect_dict, actions, None


def action_recognition_base2(frames_urls: List[str], utils, max_iterations: int = 5):
    noun_keys = []
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}")
        start_time = time.time()
        i = 0
        while True:
            selected_noun = select_nouns(frames_urls, noun_keys, None)
            selected_noun_keys = selected_noun["noun"]
            i = i + 1
            if len(selected_noun_keys) == 5 or i >= 5:
                break
        print(f"Noun selection took {time.time() - start_time:2f}s")

        start_time = time.time()
        print(f"Selected nouns: {selected_noun_keys}")

        pre_nouns = selected_noun
        noun_verb = None
        actions = combine_actions(frames_urls, pre_nouns, noun_verb, None, llm=2)
        print(f"Action combination took {time.time() - start_time:2f}s")
        start_time = time.time()
        print(f"Selected actions: {actions}")

        reflect_dict = None
        return reflect_dict, actions, selected_noun_keys


def action_recognition_base3(frames_urls: List[str], utils, max_iterations: int = 5):
    noun_keys = []
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}")
        start_time = time.time()
        i = 0
        while True:
            selected_noun = select_nouns(frames_urls, noun_keys, None)
            selected_noun_keys = selected_noun["noun"]
            i = i + 1
            if len(selected_noun_keys) == 5 or i >= 5:
                break
        print(f"Noun selection took {time.time() - start_time:2f}s")

        start_time = time.time()
        print(f"Selected nouns: {selected_noun_keys}")

        pre_nouns = selected_noun
        noun_verb = None
        actions = combine_actions(frames_urls, pre_nouns, noun_verb, None)
        print(f"Action combination took {time.time() - start_time:2f}s")
        start_time = time.time()
        print(f"Selected actions: {actions}")

        action_dict = select_actions(frames_urls, actions, pre_nouns, None)
        print(f"Action scoring took {time.time() - start_time:2f}s")
        start_time = time.time()

        reflect_dict = None
        return reflect_dict, action_dict, selected_noun_keys
