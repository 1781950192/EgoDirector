import base64
import re
from typing import List, Dict

import openai

# ====================== vLLM Client Configuration ======================
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

MODEL_NAME = "Qwen3-VL-8B-Instruct"


def image_to_base64(file_path: str) -> str | None:
    """Convert a local image to the data:url base64 format."""
    try:
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        print(f"× Failed to load the image {file_path}: {e}")
        return None


def prepare_image_messages(image_paths: List[str], text_prompt: str) -> List[Dict]:
    """Build a single-turn messages payload with multiple images (OpenAI Vision compatible)."""
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


def remove_think_tags(text: str) -> str:
    pattern = r'<think>.*?</think>'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def qwen3_vl_vllm(
    messages: List[Dict],
    max_tokens: int = 1024,
    temperature: float = 0.0
) -> str:
    """Unified call to the deterministic inference interface of vLLM."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            seed=42,
        )
        content = response.choices[0].message.content.strip()
        print(content)
        return content
    except Exception as e:
        print(f"vLLM API call error: {type(e).__name__}: {str(e)}")
        raise
