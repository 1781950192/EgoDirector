"""
数据集接口模块

提供统一的数据集抽象层，消除 EGTEA/EK100/GTEA 三个数据集的重复代码。
"""

import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd


class DatasetInterface(ABC):
    """
    数据集抽象接口
    
    所有具体数据集类必须实现以下方法：
    - load_data(): 加载数据列表
    - get_video_path(): 获取视频路径
    - get_frame_paths(): 获取帧路径列表
    - get_hos_paths(): 获取HOS图像路径列表
    - get_narration_id(): 获取唯一标识符
    """
    
    @abstractmethod
    def load_data(self) -> List[Dict]:
        """
        加载数据列表
        
        Returns:
            List[Dict]: 数据项列表，每个项包含视频ID、帧范围等信息
        """
        pass
    
    @abstractmethod
    def get_video_path(self, data_item: Dict) -> Optional[str]:
        """
        获取视频文件路径
        
        Args:
            data_item: 数据项字典
            
        Returns:
            Optional[str]: 视频路径，如果不存在返回 None
        """
        pass
    
    @abstractmethod
    def get_frame_paths(
        self, 
        data_item: Dict, 
        indices: List[int],
        frame_base: str
    ) -> List[str]:
        """
        构建帧文件路径列表
        
        Args:
            data_item: 数据项字典
            indices: 帧索引列表
            frame_base: 帧文件基础目录
            
        Returns:
            List[str]: 帧文件路径列表
        """
        pass
    
    @abstractmethod
    def get_hos_paths(
        self,
        data_item: Dict,
        positions: List[int],
        hos_base: str
    ) -> List[str]:
        """
        构建HOS图像路径列表（只返回存在的文件）
        
        Args:
            data_item: 数据项字典
            positions: HOS图像位置索引列表
            hos_base: HOS图像基础目录
            
        Returns:
            List[str]: 存在的HOS图像路径列表
        """
        pass
    
    @abstractmethod
    def get_narration_id(self, data_item: Dict) -> str:
        """
        获取数据项的唯一标识符
        
        Args:
            data_item: 数据项字典
            
        Returns:
            str: 唯一标识符（用于去重和日志）
        """
        pass


class EGTEADataset(DatasetInterface):
    """
    EGTEA 数据集实现
    
    特点：
    - 标签格式：video_id action_label（每行一个视频）
    - 视频路径：{base}/{video_id[:2段]}/{video_id}.mp4
    - HOS路径：{base}/{video_id}/{video_id}/{video_id}_vis_hos_{pos:03d}.jpg
    """
    
    def __init__(self, label_file: str, video_base_dir: str):
        """
        初始化 EGTEA 数据集
        
        Args:
            label_file: 标签文件路径（如 test_split1.txt）
            video_base_dir: 视频文件基础目录
        """
        self.label_file = label_file
        self.video_base_dir = video_base_dir
    
    def load_data(self) -> List[Dict]:
        """解析 EGTEA 标签文件"""
        data_list = []
        with open(self.label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                video_id = parts[0]
                action_label = parts[1] if len(parts) > 1 else "unknown"
                data_list.append({
                    'video_id': video_id,
                    'action_label': action_label,
                    'narration_id': video_id,
                    'start_frame': 0,
                    'stop_frame': 0,
                })
        return data_list
    
    def get_video_path(self, data_item: Dict) -> Optional[str]:
        """获取 EGTEA 视频路径"""
        video_id = data_item['video_id']
        parts = video_id.split('-')
        if len(parts) < 2:
            return None
        subdir = '-'.join(parts[:3])  # 使用前三段作为子目录 (e.g., P04-R06-GreekSalad)
        video_path = os.path.join(self.video_base_dir, subdir, video_id + '.mp4')
        return video_path if os.path.exists(video_path) else None
    
    def get_frame_paths(
        self,
        data_item: Dict,
        indices: List[int],
        frame_base: str
    ) -> List[str]:
        """
        EGTEA 不使用预提取帧，此方法返回空列表
        帧直接从视频中提取为临时文件
        """
        return []
    
    def get_hos_paths(
        self,
        data_item: Dict,
        positions: List[int],
        hos_base: str
    ) -> List[str]:
        """构建 EGTEA HOS 图像路径"""
        video_id = data_item['video_id']
        parts = video_id.split('-')
        if len(parts) < 2:
            return []

        # 尝试两种路径格式
        paths_v1 = [
            os.path.join(hos_base, video_id,
                       f"{video_id}_vis_hos_{pos:03d}.jpg")
            for pos in positions
        ]

        # 过滤存在的路径
        existing = [p for p in paths_v1 if os.path.exists(p)]

        if existing:
            return existing

        # 备用路径格式（带子目录）
        subdir = '-'.join(parts[:3])
        paths_v2 = [
            os.path.join(hos_base, subdir, video_id,
                       f"{video_id}_vis_hos_{pos:03d}.jpg")
            for pos in positions
        ]

        return [p for p in paths_v2 if os.path.exists(p)]
    
    def get_narration_id(self, data_item: Dict) -> str:
        """EGTEA 使用 video_id 作为 narration_id"""
        return data_item['video_id']


class EK100Dataset(DatasetInterface):
    """
    EPIC-KITCHENS (EK100) 数据集实现
    
    特点：
    - 标签格式：CSV 文件
    - 帧路径：{base}/{participant_id}/rgb_frames/{video_id}/frame_{idx:010d}.jpg
    - HOS路径：{base}/{participant_id}/rgb_frames/{video_id}/frame_{idx:010d}_pred.jpg
    """
    
    def __init__(self, csv_file: str, frame_base: str):
        """
        初始化 EK100 数据集
        
        Args:
            csv_file: CSV 标签文件路径
            frame_base: 帧文件基础目录
        """
        self.csv_file = csv_file
        self.frame_base = frame_base
    
    def load_data(self) -> List[Dict]:
        """加载 EK100 CSV 数据"""
        df = pd.read_csv(self.csv_file, on_bad_lines='skip')
        return df.to_dict('records')
    
    def get_video_path(self, data_item: Dict) -> Optional[str]:
        """
        EK100 使用预提取帧，不需要视频路径
        返回 None
        """
        return None
    
    def get_frame_paths(
        self,
        data_item: Dict,
        indices: List[int],
        frame_base: str = None
    ) -> List[str]:
        """构建 EK100 帧路径"""
        base = frame_base or self.frame_base
        participant_id = data_item['participant_id']
        video_id = data_item['video_id']
        
        base_path = os.path.join(base, participant_id, 'rgb_frames', video_id)
        if not os.path.exists(base_path):
            return []
        
        return [os.path.join(base_path, f"frame_{idx:010d}.jpg") for idx in indices]
    
    def get_hos_paths(
        self,
        data_item: Dict,
        positions: List[int],
        hos_base: str
    ) -> List[str]:
        """构建 EK100 HOS 图像路径"""
        participant_id = data_item['participant_id']
        video_id = data_item['video_id']
        
        base_path = os.path.join(hos_base, participant_id, 'rgb_frames', video_id)
        if not os.path.exists(base_path):
            return []
        
        # EK100 HOS 使用相同的帧索引
        paths = [os.path.join(base_path, f"frame_{idx:010d}_pred.jpg") for idx in positions]
        return [p for p in paths if os.path.exists(p)]
    
    def get_narration_id(self, data_item: Dict) -> str:
        """EK100 使用 narration_id 字段"""
        return data_item['narration_id']


class GTEADataset(DatasetInterface):
    """
    GTEA 数据集实现
    
    特点：
    - 标签格式：txt 文件，每行 <verb><noun> (start-end)
    - 视频路径：{base}/{video_name}.mp4
    - HOS路径：{base}/{video_id}/{video_id}_vis_hos_{pos:03d}.jpg
    """
    
    def __init__(self, labels_dir: str, videos_base_path: str):
        """
        初始化 GTEA 数据集
        
        Args:
            labels_dir: 标签文件目录
            videos_base_path: 视频文件基础目录
        """
        self.labels_dir = labels_dir
        self.videos_base_path = videos_base_path
    
    def load_data(self) -> List[Dict]:
        """解析 GTEA 标签文件（只导入 verb+noun 动作）"""
        data_list = []
        label_files = [f for f in os.listdir(self.labels_dir) if f.endswith('.txt')]
        
        for label_file in label_files:
            label_path = os.path.join(self.labels_dir, label_file)
            video_name = label_file.replace('.txt', '')
            
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 提取动作部分
                if ' (' in line:
                    action_part = line.split(' (')[0].strip()
                else:
                    action_part = line.split('(')[0].strip()
                
                # 跳过单一物体标注（没有 >< 连接符）
                if '><' not in action_part:
                    continue
                
                time_part = line.split('(')[1].split(')')[0].strip()
                parts = action_part.split('><')
                
                verb = parts[0].replace('<', '').strip()
                noun = parts[1].replace('>', '').strip()
                
                start_frame, end_frame = map(int, time_part.split('-'))
                
                narration_id = f"{video_name}_{start_frame}_{end_frame}"
                action = f"{verb}_{noun}" if noun else verb
                
                data_list.append({
                    'narration_id': narration_id,
                    'video_id': video_name,
                    'start_frame': start_frame,
                    'stop_frame': end_frame,
                    'verb': verb,
                    'noun': noun,
                    'action': action,
                })
        
        return data_list
    
    def get_video_path(self, data_item: Dict) -> Optional[str]:
        """获取 GTEA 视频路径"""
        video_name = data_item['video_id']
        path = os.path.join(self.videos_base_path, video_name + '.mp4')
        return path if os.path.exists(path) else None
    
    def get_frame_paths(
        self,
        data_item: Dict,
        indices: List[int],
        frame_base: str = None
    ) -> List[str]:
        """
        GTEA 不使用预提取帧，此方法返回空列表
        帧直接从视频中提取为临时文件
        """
        return []
    
    def get_hos_paths(
        self,
        data_item: Dict,
        positions: List[int],
        hos_base: str
    ) -> List[str]:
        """构建 GTEA HOS 图像路径"""
        video_id = data_item['video_id']
        
        paths = [
            os.path.join(hos_base, video_id, 
                       f"{video_id}_vis_hos_{pos:03d}.jpg")
            for pos in positions
        ]
        
        return [p for p in paths if os.path.exists(p)]
    
    def get_narration_id(self, data_item: Dict) -> str:
        """GTEA 使用 video_start_end 格式"""
        return f"{data_item['video_id']}_{data_item['start_frame']}_{data_item['stop_frame']}"
