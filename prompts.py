from utils import read_txt_file, extract_and_load_json
from vllm_client import qwen3_vl_vllm, prepare_image_messages

prompt_dict = {
    'combine_actions_prompt.txt': 'prompt/combine_actions_prompt.txt',
    'combine_actions_prompt_2llm.txt': 'prompt/combine_actions_prompt_2llm.txt',
    "reflector_prompt.txt": 'prompt/Reflector_prompt/Reflector_prompt.txt',
    'select_actions_prompt.txt': 'prompt/select_actions_prompt.txt',
    'select_noun_prompt.txt': 'prompt/select_noun_prompt.txt',
    'base.txt': 'prompt/base.txt'
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
