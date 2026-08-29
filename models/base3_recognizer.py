"""
Base3 识别器模块（base=3）

复杂识别器：选名词 → 加载先验 → 组合动作，不评分
"""

import json
import time
from typing import List, Dict, Optional, Tuple

# 导入 moduls.py 中的工具函数
from moduls import select_nouns, combine_actions, extract_noun_probabilities

from .base_recognizer import BaseRecognizer


class Base3Recognizer(BaseRecognizer):
    """
    Base3 识别器（base=3）
    
    特点：
    - 选择名词
    - 加载名词先验信息
    - 使用名词和先验组合动作
    - 不进行动作评分
    
    使用示例：
        recognizer = Base3Recognizer()
        reflect, actions, nouns = recognizer.recognize(frame_paths)
    """
    
    @property
    def name(self) -> str:
        return "Base3Recognizer (base=3)"
    
    def recognize(
        self,
        frames_urls: List[str],
        max_iterations: int = 5
    ) -> Tuple[Optional[Dict], Dict, Optional[List[str]]]:
        """
        执行复杂的动作识别流程
        
        流程：
        1. 选择名词（最多5个）
        2. 加载名词先验信息
        3. 提取名词-动词概率
        4. 组合动作
        
        Args:
            frames_urls: 帧文件路径列表
            max_iterations: 最大迭代次数（当前实现只执行一次）
            
        Returns:
            Tuple[Optional[Dict], Dict, Optional[List[str]]]: 
                (reflect_dict, action_dict, selected_noun_keys)
        """
        for iteration in range(max_iterations):
            print(f"迭代 {iteration + 1}")
            
            # === 阶段1: 选择名词 ===
            start_time = time.time()
            noun_keys = []  # Base3 不使用已知名词列表
            i = 0
            while True:
                selected_noun = select_nouns(frames_urls, noun_keys, None)
                selected_noun_keys = selected_noun["noun"]
                i = i + 1
                if len(selected_noun_keys) == 5 or i >= 5:
                    break
            print(f"挑选名词的时间是{time.time()-start_time:.2f}秒")

            start_time = time.time()
            print(f"挑选出来的名词是：{selected_noun_keys}")

            # === 阶段2: 加载名词先验信息 ===
            with open('context/pre_noun.json', 'r', encoding='utf-8') as f:
                pre_noun = json.load(f)
            
            # 构建快速查找字典
            key_to_text = {item["key"]: item["generated_text"] for item in pre_noun}
            pre_nouns = {}
            for noun in selected_noun_keys:
                pre_nouns[noun] = key_to_text.get(noun)

            # 提取名词-动词概率
            keys = list(pre_nouns.keys())
            noun_verb = extract_noun_probabilities(keys)

            # === 阶段3: 组合动作 ===
            actions = combine_actions(frames_urls, pre_nouns, noun_verb, None, llm=2)
            print(f"组合动作的时间为：{time.time()-start_time:.2f}秒")
            print(f"挑选出来的动作是：{actions}")

            # 当前版本不启用反思机制
            reflect_dict = None

            # Base3 返回 actions 作为 action_dict
            if isinstance(actions, dict):
                action_dict = actions
            else:
                action_dict = {"actions": actions}

            return reflect_dict, action_dict, selected_noun_keys
        
        # 如果所有迭代都失败，返回空结果
        return None, {"actions": []}, None
