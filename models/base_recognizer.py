"""
识别器基类模块

定义所有识别器的统一接口。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple


class BaseRecognizer(ABC):
    """
    识别器抽象基类
    
    所有具体识别器必须实现 recognize() 方法。
    
    返回格式：
        Tuple[Optional[Dict], Dict, Optional[List[str]]]
        - reflect_dict: 反思字典（当前始终为 None）
        - action_dict: 动作字典，包含 "actions" 键
        - selected_noun_keys: 选择的名词键列表（可能为 None）
    """
    
    @abstractmethod
    def recognize(
        self,
        frames_urls: List[str],
        max_iterations: int = 5
    ) -> Tuple[Optional[Dict], Dict, Optional[List[str]]]:
        """
        执行动作识别
        
        Args:
            frames_urls: 帧文件路径列表
            max_iterations: 最大迭代次数
            
        Returns:
            Tuple[Optional[Dict], Dict, Optional[List[str]]]: 
                (reflect_dict, action_dict, selected_noun_keys)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """返回识别器名称"""
        pass
