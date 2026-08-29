"""
帧提取器模块

提供统一的视频帧提取和选择功能，消除三个主脚本中的重复代码。
"""

import os
import tempfile
from typing import List, Tuple
import cv2
import numpy as np


class FrameExtractor:
    """
    统一的帧提取器
    
    职责：
    1. 从视频中均匀采样指定数量的帧
    2. 从源帧索引中选择目标数量的帧
    3. 清理临时文件
    
    使用示例：
        extractor = FrameExtractor()
        indices, paths = extractor.extract_uniform(video_path, num_frames=32)
        selected = extractor.select_from_source(indices, target_num=16)
    """
    
    @staticmethod
    def extract_uniform_with_indices(
        video_path: str, 
        num_frames: int = 32
    ) -> Tuple[List[int], List[str]]:
        """
        从视频中均匀采样指定数量的帧
        
        Args:
            video_path: 视频文件路径
            num_frames: 要提取的帧数量
            
        Returns:
            Tuple[List[int], List[str]]: (帧索引列表, 临时文件路径列表)
            
        Raises:
            FileNotFoundError: 视频文件不存在
            ValueError: 视频无法打开或帧数为0
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError(f"视频帧数为0: {video_path}")
            
            # 计算均匀采样的帧索引
            indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int).tolist()
            temp_files = []
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                    cv2.imwrite(temp_file.name, frame)
                    temp_files.append(temp_file.name)
            
            return indices, temp_files
            
        finally:
            cap.release()
    
    @staticmethod
    def extract_with_range(
        video_path: str,
        start_frame: int,
        stop_frame: int,
        num_frames: int = 32
    ) -> Tuple[List[int], List[str]]:
        """
        从视频的指定范围均匀采样帧
        
        Args:
            video_path: 视频文件路径
            start_frame: 起始帧索引
            stop_frame: 结束帧索引（不包含）
            num_frames: 要提取的帧数量
            
        Returns:
            Tuple[List[int], List[str]]: (帧索引列表, 临时文件路径列表)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stop_frame = min(stop_frame, total_frames)
            
            # 计算要提取的帧索引
            indices = np.linspace(start_frame, stop_frame - 1, num=num_frames, dtype=int).tolist()
            temp_files = []
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                    cv2.imwrite(temp_file.name, frame)
                    temp_files.append(temp_file.name)
            
            return indices, temp_files
            
        finally:
            cap.release()
    
    @staticmethod
    def select_from_source(
        source_indices: List[int],
        target_num: int
    ) -> List[int]:
        """
        从源索引中选择目标数量的帧（保持相对顺序）
        
        Args:
            source_indices: 源帧索引列表（如32帧的索引）
            target_num: 目标帧数（如16帧）
            
        Returns:
            List[int]: 选择的帧索引列表
            
        Example:
            >>> source = [0, 2, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30, 33, 35, 38]
            >>> FrameExtractor.select_from_source(source, 8)
            [0, 5, 10, 15, 20, 25, 30, 38]
        """
        if target_num >= len(source_indices):
            return source_indices
        
        # 计算理想的目标帧位置
        ideal_positions = np.linspace(0, len(source_indices) - 1, num=target_num, dtype=int)
        
        # 从源索引中选择最接近理想位置的帧
        selected_indices = [source_indices[pos] for pos in ideal_positions]
        
        return selected_indices
    
    @staticmethod
    def cleanup_temp_files(temp_files: List[str]) -> None:
        """
        清理临时文件
        
        Args:
            temp_files: 临时文件路径列表
        """
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.unlink(f)
                except Exception as e:
                    print(f"警告: 无法删除临时文件 {f}: {e}")
