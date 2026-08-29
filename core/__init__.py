"""
action_agent_vllm 核心抽象层

提供统一的数据集接口、帧提取器、HOS处理器和结果管理器。
"""

from .frame_extractor import FrameExtractor
from .dataset_interface import DatasetInterface, EGTEADataset, EK100Dataset, GTEADataset
from .hos_processor import HOSProcessor
from .result_manager import ResultManager

__all__ = [
    'FrameExtractor',
    'DatasetInterface',
    'EGTEADataset',
    'EK100Dataset', 
    'GTEADataset',
    'HOSProcessor',
    'ResultManager'
]
