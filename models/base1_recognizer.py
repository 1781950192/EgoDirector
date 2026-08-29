"""
Base1 识别器模块（base=1）

简化版识别器：直接组合动作，不选名词、不更新知识库
"""

import time
from typing import List, Dict, Optional, Tuple

# 导入 moduls.py 中的工具函数
from moduls import combine_actions

from .base_recognizer import BaseRecognizer


class Base1Recognizer(BaseRecognizer):
    """
    Base1 识别器（base=1）
    
    特点：
    - 最简化的流程
    - 不选择名词，直接使用 LLM 组合动作
    - 不更新知识库
    
    使用示例：
        recognizer = Base1Recognizer()
        reflect, actions, nouns = recognizer.recognize(frame_paths)
    """
    
    @property
    def name(self) -> str:
        return "Base1Recognizer (base=1)"
    
    def recognize(
        self,
        frames_urls: List[str],
        max_iterations: int = 5
    ) -> Tuple[Optional[Dict], Dict, Optional[List[str]]]:
        """
        执行简化的动作识别流程
        
        流程：
        1. 直接使用 LLM 组合动作（llm=2）
        
        Args:
            frames_urls: 帧文件路径列表
            max_iterations: 最大迭代次数（当前实现只执行一次）
            
        Returns:
            Tuple[Optional[Dict], Dict, Optional[List[str]]]: 
                (reflect_dict, action_dict, selected_noun_keys=None)
        """
        for iteration in range(max_iterations):
            print(f"迭代 {iteration + 1}")
            
            start_time = time.time()
            
            # 直接组合动作，不使用先验知识
            pre_nouns = None
            noun_verb = None
            actions = combine_actions(frames_urls, pre_nouns, noun_verb, None, llm=2)
            
            print(f"组合动作的时间为：{time.time()-start_time:.2f}秒")
            print(f"挑选出来的动作是：{actions}")

            # 当前版本不启用反思机制
            reflect_dict = None

            # Base1 返回 actions 作为 action_dict
            if isinstance(actions, dict):
                action_dict = actions
            else:
                action_dict = {"actions": actions}

            return reflect_dict, action_dict, None
        
        # 如果所有迭代都失败，返回空结果
        return None, {"actions": []}, None
