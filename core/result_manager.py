"""
结果管理器模块

提供统一的结果保存和已处理记录管理功能。
"""

import os
from typing import Dict, List, Set, Optional


class ResultManager:
    """
    结果管理器
    
    职责：
    1. 初始化输出文件（创建表头）
    2. 加载已处理的ID集合（支持断点续传）
    3. 保存识别结果到文件
    4. 管理失败日志
    
    使用示例：
        manager = ResultManager('output.txt')
        processed = manager.load_processed_ids()
        manager.save_result(data_item, predicted_actions, selected_nouns)
    """
    
    def __init__(self, output_path: str):
        """
        初始化结果管理器
        
        Args:
            output_path: 输出文件路径
        """
        self.output_path = output_path
        self.header = "narration_id,verbs,nouns,actions,confidences,selected_noun_keys\n"
    
    def initialize_output(self) -> None:
        """
        初始化输出文件（如果不存在或为空，则写入表头）
        """
        if not os.path.exists(self.output_path) or os.path.getsize(self.output_path) == 0:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                f.write(self.header)
    
    def load_processed_ids(self) -> Set[str]:
        """
        加载已处理的ID集合（用于断点续传）
        
        Returns:
            Set[str]: 已处理的narration_id集合
        """
        processed_ids = set()
        
        if not os.path.exists(self.output_path):
            return processed_ids
        
        with open(self.output_path, 'r', encoding='utf-8') as f:
            # 跳过表头
            next(f, None)
            
            for line in f:
                if line.strip():
                    narration_id = line.split(',', 1)[0].strip()
                    processed_ids.add(narration_id)
        
        return processed_ids
    
    def save_result(
        self,
        data_item: Dict,
        predicted_action_dict: Dict,
        selected_noun_keys: Optional[List[str]] = None
    ) -> None:
        """
        保存单个数据项的识别结果
        
        Args:
            data_item: 数据项字典（必须包含 narration_id）
            predicted_action_dict: 预测的动作字典
            selected_noun_keys: 选择的名词键列表（可选）
        """
        narration_id = data_item.get('narration_id', 'unknown')
        
        # 提取动作信息
        actions = predicted_action_dict.get("actions", [])
        sorted_actions = sorted(actions, key=lambda x: x.get("confidence", 0), reverse=True)
        
        verbs = ";".join([a.get("verb", "").strip() for a in sorted_actions if a.get("verb")]) or "无动词"
        nouns = ";".join([a.get("noun", "").strip() for a in sorted_actions if a.get("noun")]) or "无名词"
        actions_str = ";".join([a.get("action", "").strip() for a in sorted_actions if a.get("action")]) or "无动作"
        confidences = ";".join([str(a.get("confidence", 0)) for a in sorted_actions if a.get("confidence")]) or "0.0"
        
        # 处理 selected_noun_keys
        if selected_noun_keys:
            if isinstance(selected_noun_keys, list):
                selected_noun_keys_str = ";".join(map(str, selected_noun_keys))
            else:
                selected_noun_keys_str = str(selected_noun_keys)
        else:
            selected_noun_keys_str = "无选择名词"
        
        # 追加到文件
        with open(self.output_path, 'a', encoding='utf-8') as f:
            f.write(f"{narration_id},{verbs},{nouns},{actions_str},{confidences},{selected_noun_keys_str}\n")
    
    def get_failure_log_path(self) -> str:
        """
        获取失败日志文件路径
        
        Returns:
            str: 失败日志文件路径（output_failures.txt）
        """
        base = self.output_path.rsplit('.', 1)[0]
        return f"{base}_failures.txt"
    
    def log_failure(self, narration_id: str, error_message: str) -> None:
        """
        记录处理失败的信息
        
        Args:
            narration_id: 失败的narration_id
            error_message: 错误信息
        """
        failure_path = self.get_failure_log_path()
        
        with open(failure_path, 'a', encoding='utf-8') as f:
            f.write(f"{narration_id}: {error_message}\n")
    
    def get_stats(self) -> Dict[str, int]:
        """
        获取处理统计信息
        
        Returns:
            Dict[str, int]: 统计信息 {'total': 总数, 'processed': 已处理}
        """
        if not os.path.exists(self.output_path):
            return {'total': 0, 'processed': 0}
        
        processed_count = 0
        with open(self.output_path, 'r', encoding='utf-8') as f:
            next(f, None)  # 跳过表头
            for line in f:
                if line.strip():
                    processed_count += 1
        
        return {
            'processed': processed_count
        }
