"""
Base2 识别器模块（base=2）

中等复杂度识别器：选名词 → 组合动作，不更新知识库、不评分
"""

import time
from typing import List, Dict, Optional, Tuple

# 导入 moduls.py 中的工具函数
from moduls import select_nouns, combine_actions

from .base_recognizer import BaseRecognizer


class Base2Recognizer(BaseRecognizer):
    """
    Base2 识别器（base=2）
    
    特点：
    - 选择名词
    - 使用名词组合动作
    - 不更新知识库，不进行动作评分
    
    使用示例：
        recognizer = Base2Recognizer()
        reflect, actions, nouns = recognizer.recognize(frame_paths)
    """
    
    @property
    def name(self) -> str:
        return "Base2Recognizer (base=2)"
    
    def recognize(
        self,
        frames_urls: List[str],
        max_iterations: int = 5
    ) -> Tuple[Optional[Dict], Dict, Optional[List[str]]]:
        """
        执行中等复杂度的动作识别流程
        
        流程：
        1. 选择名词（最多5个）
        2. 使用名词组合动作（llm=2）
        
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
            noun_keys = []  # Base2 不使用已知名词列表
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

            # === 阶段2: 组合动作 ===
            pre_nouns = selected_noun_keys
            noun_verb = None

            actions = combine_actions(frames_urls, pre_nouns, noun_verb, None, llm=2)
            print(f"组合动作的时间为：{time.time()-start_time:.2f}秒")
            print(f"挑选出来的动作是：{actions}")

            # 当前版本不启用反思机制
            reflect_dict = None

            # Base2 返回 actions 作为 action_dict
            if isinstance(actions, dict):
                action_dict = actions
            else:
                action_dict = {"actions": actions}

            return reflect_dict, action_dict, selected_noun_keys
        
        # 如果所有迭代都失败，返回空结果
        return None, {"actions": []}, None
