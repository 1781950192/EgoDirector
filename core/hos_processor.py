"""
HOS图像处理器模块

提供统一的HOS图像路径构建和过滤功能。
"""

import os
from typing import List, Dict


class HOSProcessor:
    """
    HOS图像处理器
    
    职责：
    1. 根据数据集类型构建HOS图像路径
    2. 过滤不存在的文件
    3. 处理不同数据集的路径格式差异
    
    使用示例：
        processor = HOSProcessor('egtea', '/path/to/hos')
        paths = processor.build_paths(data_item, positions)
    """
    
    def __init__(self, dataset_type: str, hos_base_path: str):
        """
        初始化HOS处理器
        
        Args:
            dataset_type: 数据集类型 ('egtea', 'ek100', 'gtea')
            hos_base_path: HOS图像基础目录
        """
        self.dataset_type = dataset_type.lower()
        self.hos_base_path = hos_base_path
        
        if self.dataset_type not in ['egtea', 'ek100', 'gtea']:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    def build_paths(
        self,
        data_item: Dict,
        positions: List[int]
    ) -> List[str]:
        """
        构建HOS图像路径列表（只返回存在的文件）
        
        Args:
            data_item: 数据项字典
            positions: HOS图像位置索引列表
            
        Returns:
            List[str]: 存在的HOS图像路径列表
        """
        if self.dataset_type == 'egtea':
            return self._build_egtea_paths(data_item, positions)
        elif self.dataset_type == 'ek100':
            return self._build_ek100_paths(data_item, positions)
        elif self.dataset_type == 'gtea':
            return self._build_gtea_paths(data_item, positions)
        else:
            return []
    
    def _build_egtea_paths(
        self,
        data_item: Dict,
        positions: List[int]
    ) -> List[str]:
        """构建EGTEA HOS路径"""
        video_id = data_item['video_id']
        parts = video_id.split('-')
        if len(parts) < 2:
            return []

        # 尝试两种路径格式
        paths_v1 = [
            os.path.join(self.hos_base_path, video_id,
                       f"{video_id}_vis_hos_{pos:03d}.jpg")
            for pos in positions
        ]

        existing = [p for p in paths_v1 if os.path.exists(p)]
        if existing:
            return existing

        # 备用路径格式
        subdir = '-'.join(parts[:3])
        paths_v2 = [
            os.path.join(self.hos_base_path, subdir, video_id,
                       f"{video_id}_vis_hos_{pos:03d}.jpg")
            for pos in positions
        ]

        return [p for p in paths_v2 if os.path.exists(p)]
    
    def _build_ek100_paths(
        self,
        data_item: Dict,
        positions: List[int]
    ) -> List[str]:
        """构建EK100 HOS路径"""
        participant_id = data_item['participant_id']
        video_id = data_item['video_id']
        
        base_path = os.path.join(
            self.hos_base_path, 
            participant_id, 
            'rgb_frames', 
            video_id
        )
        
        if not os.path.exists(base_path):
            return []
        
        paths = [
            os.path.join(base_path, f"frame_{idx:010d}_pred.jpg") 
            for idx in positions
        ]
        
        return [p for p in paths if os.path.exists(p)]
    
    def _build_gtea_paths(
        self,
        data_item: Dict,
        positions: List[int]
    ) -> List[str]:
        """构建GTEA HOS路径"""
        video_id = data_item['video_id']
        
        paths = [
            os.path.join(self.hos_base_path, video_id, 
                       f"{video_id}_vis_hos_{pos:03d}.jpg")
            for pos in positions
        ]
        
        return [p for p in paths if os.path.exists(p)]
    
    @staticmethod
    def filter_existing(paths: List[str]) -> List[str]:
        """
        过滤存在的文件路径
        
        Args:
            paths: 路径列表
            
        Returns:
            List[str]: 存在的路径列表
        """
        return [p for p in paths if os.path.exists(p)]
