# backward-compatibility shim — re-exports from new modules
# All implementations now live in:
#   vllm_client.py, prompts.py, context_manager.py, action_pipeline.py

from vllm_client import (
    client,
    MODEL_NAME,
    image_to_base64,
    prepare_image_messages,
    qwen3_vl_vllm,
    remove_think_tags,
)

from prompts import (
    prompt_dict,
    pre_noun_add,
    verb_noun_add,
)

from context_manager import (
    extract_noun_probabilities,
    update_json_file,
    update_verb_noun_json,
    overlay_context_base,
)

from action_pipeline import (
    reflector_action,
    select_nouns,
    combine_actions,
    select_actions,
    action_recognition,
    action_recognition_with_timing,
    action_recognition_base1,
    action_recognition_base2,
    action_recognition_base3,
)
