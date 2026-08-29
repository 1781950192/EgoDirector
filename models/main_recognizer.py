"""
主识别器模块（base=0）

实现完整的动作识别流程：选名词 → 更新知识库 → 组合动作 → 评分动作
"""

import json
import time
from typing import List, Dict, Optional, Tuple

# 导入 moduls.py 中的工具函数
from moduls import (
    select_nouns,
    combine_actions,
    select_actions,
    verb_noun_add,
    pre_noun_add,
    update_verb_noun_json,
    update_json_file,
    extract_noun_probabilities
)

from .base_recognizer import BaseRecognizer


class MainRecognizer(BaseRecognizer):
    """
    主识别器（base=0）
    
    特点：
    - 使用完整的多阶段流程
    - 动态更新知识库（verb_noun.json, pre_noun.json）
    - 结合历史经验进行动作评分
    
    使用示例：
        recognizer = MainRecognizer()
        reflect, actions, nouns = recognizer.recognize(frame_paths)
    """
    
    @property
    def name(self) -> str:
        return "MainRecognizer (base=0)"
    
    def recognize(
        self,
        frames_urls: List[str],
        max_iterations: int = 5
    ) -> Tuple[Optional[Dict], Dict, Optional[List[str]]]:
        """
        执行完整的动作识别流程
        
        流程：
        1. 选择名词（最多5个）
        2. 检查并更新知识库（未知名词）
        3. 加载名词先验信息
        4. 组合可能的动作
        5. 对动作进行评分
        
        Args:
            frames_urls: 帧文件路径列表
            max_iterations: 最大迭代次数（当前实现只执行一次）
            
        Returns:
            Tuple[Optional[Dict], Dict, Optional[List[str]]]: 
                (reflect_dict, action_dict, selected_noun_keys)
        """
        # 加载已知名词列表
        with open('context/verb_noun.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        noun_keys = data.keys()

        for iteration in range(max_iterations):
            print(f"迭代 {iteration + 1}")
            
            # === 阶段1: 选择名词 ===
            start_time = time.time()
            i = 0
            while True:
                selected_noun = select_nouns(frames_urls, "None", None)
                selected_noun_keys = selected_noun["noun"]
                i = i + 1
                if len(selected_noun_keys) == 5 or i >= 5:
                    break
            print(f"挑选名词的时间是{time.time()-start_time:.2f}秒")

            # === 阶段2: 更新知识库（处理未知名词）===
            start_time = time.time()
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
                    
                    # 收集verb_noun更新
                    verb_entry = verb_noun_add(item)  # 返回 {noun: [verbs]}
                    verb_noun_updates.append(verb_entry)

                # 批量更新知识库
                update_verb_noun_json('context/verb_noun.json', verb_noun_updates)
                update_json_file('context/pre_noun.json', pre_noun_updates)
            
            if selected_noun_keys is None:
                print("选择名词失败，跳过本次迭代")
                continue

            print(f"更新上下文的时间为：{time.time()-start_time:.2f}秒")
            start_time = time.time()
            print(f"挑选出来的名词是：{selected_noun_keys}")

            # === 阶段3: 加载名词先验信息 ===
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

            # === 阶段4: 组合动作 ===
            actions = combine_actions(frames_urls, pre_nouns, noun_verb, None)
            print(f"组合动作的时间为：{time.time()-start_time:.2f}秒")
            start_time = time.time()

            if actions is None:
                print("选择动作失败，跳过本次迭代")
                continue
            print(f"挑选出来的动作是：{actions}")

            # === 阶段5: 动作评分 ===
            action_dict = select_actions(frames_urls, actions, pre_nouns, None)
            print(f"动作评分的时间为：{time.time() - start_time:.2f}秒")

            # 当前版本不启用反思机制
            reflect_dict = None

            return reflect_dict, action_dict, selected_noun_keys
        
        # 如果所有迭代都失败，返回空结果
        return None, {"actions": []}, None
