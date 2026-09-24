import json
import os
from pathlib import Path
from typing import List
import shutil


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
    with open(json_file_path, 'r') as f:
        noun_verb_probabilities = json.load(f)

    result = {}
    for noun in noun_list:
        result[noun] = noun_verb_probabilities.get(noun, {})

    return result


def update_json_file(filename, new_data):
    """Helper function to update a JSON file (list format)."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data.extend(new_data)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        data = []
        data.extend(new_data)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def update_verb_noun_json(filename: str, new_entries: List[dict]):
    """
    Update verb_noun.json (dict format) and add the verb lists of the new nouns.
    new_entries: A list whose elements are { "noun": ["verb1", ...] }.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print("[Warning] verb_noun.json is not in dict format, re-initialize it")
            data = {}
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    for entry in new_entries:
        data.update(entry)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[Success] Updated {len(new_entries)} nouns in {filename}")


def overlay_context_base():
    """Overlay the files in context_base onto context."""
    base_dir = Path("/mnt/data/xgl/mycode/action_agent_vllm/context_base")
    target_dir = Path("/mnt/data/xgl/mycode/action_agent_vllm/context")

    if not base_dir.is_dir():
        print(f"The source folder does not exist: {base_dir}")
        return
    if not target_dir.is_dir():
        print(f"The target folder does not exist: {target_dir}")
        return

    print(f"Overlay the files in {base_dir} onto {target_dir}")

    for src in base_dir.rglob("*"):
        if src.is_file():
            rel = src.relative_to(base_dir)
            dst = target_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"Overwrite/create -> {dst.name}")
