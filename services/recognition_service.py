"""
识别服务编排器模块

整合 core/ 和 models/ 模块，提供端到端的识别流程。
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple

from core.dataset_interface import DatasetInterface
from core.frame_extractor import FrameExtractor
from core.hos_processor import HOSProcessor
from core.result_manager import ResultManager
from models.base_recognizer import BaseRecognizer


class RecognitionService:
    """
    识别服务编排器
    
    职责：
    1. 整合数据集、帧提取器、HOS处理器、识别器和结果管理器
    2. 编排完整的识别流程
    3. 支持批量处理和断点续传
    
    使用示例：
        dataset = EGTEADataset(label_file, video_base)
        recognizer = MainRecognizer()
        service = RecognitionService(
            dataset=dataset,
            recognizer=recognizer,
            use_hos=True,
            num_frames=32
        )
        service.process_batch(output_path, batch_size=20)
    """
    
    def __init__(
        self,
        dataset: DatasetInterface,
        recognizer: BaseRecognizer,
        use_hos: bool = False,
        num_frames: int = 32,
        hos_base_path: str = None
    ):
        """
        初始化识别服务
        
        Args:
            dataset: 数据集接口实例
            recognizer: 识别器实例
            use_hos: 是否使用HOS图像
            num_frames: 每视频采样的帧数
            hos_base_path: HOS图像基础路径（use_hos=True时必需）
        """
        self.dataset = dataset
        self.recognizer = recognizer
        self.use_hos = use_hos
        self.num_frames = num_frames
        self.hos_base_path = hos_base_path
        
        # 初始化组件
        self.frame_extractor = FrameExtractor()
        
        if use_hos and hos_base_path:
            # 推断数据集类型
            dataset_type = self._infer_dataset_type(dataset)
            self.hos_processor = HOSProcessor(dataset_type, hos_base_path)
        else:
            self.hos_processor = None
    
    def _infer_dataset_type(self, dataset: DatasetInterface) -> str:
        """
        推断数据集类型
        
        Args:
            dataset: 数据集实例
            
        Returns:
            str: 数据集类型 ('egtea', 'ek100', 'gtea')
        """
        dataset_class_name = type(dataset).__name__
        
        if 'EGTEA' in dataset_class_name:
            return 'egtea'
        elif 'EK100' in dataset_class_name:
            return 'ek100'
        elif 'GTEA' in dataset_class_name:
            return 'gtea'
        else:
            raise ValueError(f"无法推断数据集类型: {dataset_class_name}")
    
    def process_single_item(self, data_item: Dict) -> Optional[Dict]:
        """
        处理单个数据项
        
        流程：
        1. 获取视频路径或帧路径
        2. 提取帧（从视频或预提取帧）
        3. 构建HOS路径（如果使用）
        4. 执行识别
        5. 返回结果
        
        Args:
            data_item: 数据项字典
            
        Returns:
            Optional[Dict]: 识别结果字典，失败返回 None
        """
        narration_id = self.dataset.get_narration_id(data_item)
        
        try:
            # === 步骤1: 准备帧路径 ===
            frame_paths = []
            
            # 尝试从预提取帧获取（EK100）
            video_path = self.dataset.get_video_path(data_item)
            
            if video_path is None:
                # EK100: 使用预提取帧，根据 start_frame/stop_frame 计算索引
                start_frame = data_item.get('start_frame', 0)
                stop_frame = data_item.get('stop_frame', 1)

                # 在 [start_frame, stop_frame-1] 范围内均匀采样 num_frames 帧
                frame_indices = np.linspace(
                    start_frame,
                    stop_frame - 1,
                    num=self.num_frames,
                    dtype=int
                ).tolist()

                frame_paths = self.dataset.get_frame_paths(
                    data_item,
                    frame_indices,
                    frame_base=None
                )

                # 过滤存在的帧文件，同时保持索引和路径的对应关系
                index_path_pairs = list(zip(frame_indices, frame_paths))
                valid_pairs = [(idx, p) for idx, p in index_path_pairs if os.path.exists(p)]
                frame_indices = [idx for idx, _ in valid_pairs]
                frame_paths = [p for _, p in valid_pairs]

                if not frame_paths:
                    print(f"警告: 未找到帧文件 {narration_id}")
                    return None
            else:
                # EGTEA/GTEA: 从视频中提取帧
                start_frame = data_item.get('start_frame', 0)
                stop_frame = data_item.get('stop_frame', 0)

                if start_frame > 0 and stop_frame > start_frame:
                    # 有明确的帧范围
                    indices, temp_files = self.frame_extractor.extract_with_range(
                        video_path,
                        start_frame,
                        stop_frame,
                        self.num_frames
                    )
                else:
                    # 全视频均匀采样
                    indices, temp_files = self.frame_extractor.extract_uniform_with_indices(
                        video_path,
                        self.num_frames
                    )

                # 对于从视频提取的帧，HOS使用相对位置索引 (0, 1, 2, ...)
                frame_indices = list(range(len(temp_files)))
                frame_paths = temp_files

            # === 步骤2: 构建HOS路径（如果使用）===
            if self.use_hos and self.hos_processor:
                # EK100: frame_indices 是实际帧号
                # EGTEA/GTEA: frame_indices 是相对位置 (0, 1, 2, ...)
                hos_paths = self.hos_processor.build_paths(data_item, frame_indices)

                if hos_paths:
                    # 将HOS路径添加到帧路径中
                    frame_paths.extend(hos_paths)
            
            # === 步骤3: 执行识别 ===
            reflect_dict, action_dict, selected_noun_keys = self.recognizer.recognize(
                frame_paths,
                max_iterations=5
            )
            
            # === 步骤4: 清理临时文件 ===
            if video_path:  # 只有从视频提取的才需要清理
                self.frame_extractor.cleanup_temp_files(frame_paths)
            
            # === 步骤5: 返回结果 ===
            return {
                'narration_id': narration_id,
                'reflect_dict': reflect_dict,
                'action_dict': action_dict,
                'selected_noun_keys': selected_noun_keys,
                'frame_count': len(frame_paths)
            }
            
        except Exception as e:
            print(f"错误: 处理 {narration_id} 时出错: {e}")
            
            # 清理临时文件
            if video_path:
                self.frame_extractor.cleanup_temp_files(frame_paths)
            
            return None
    
    def process_batch(
        self,
        output_path: str,
        batch_size: int = 20
    ) -> None:
        """
        批量处理数据集中的所有项
        
        Args:
            output_path: 输出文件路径
            batch_size: 批处理大小（当前未使用，保留接口）
        """
        # 加载数据
        print(f"正在加载数据集...")
        data_list = self.dataset.load_data()
        print(f"加载完成，共 {len(data_list)} 条数据")
        
        # 初始化结果管理器
        result_manager = ResultManager(output_path)
        result_manager.initialize_output()
        
        # 加载已处理的ID（断点续传）
        processed_ids = result_manager.load_processed_ids()
        print(f"已处理 {len(processed_ids)} 条，跳过这些项")
        
        # 批量处理
        total = len(data_list)
        processed_count = 0
        skipped_count = len(processed_ids)
        failed_count = 0
        
        for idx, data_item in enumerate(data_list):
            narration_id = self.dataset.get_narration_id(data_item)
            
            # 跳过已处理的
            if narration_id in processed_ids:
                continue
            
            # 打印进度
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"\n进度: [{idx + 1}/{total}] ({(idx + 1) / total * 100:.1f}%)")
                print(f"  已处理: {processed_count}, 跳过: {skipped_count}, 失败: {failed_count}")
            
            # 处理单个项
            result = self.process_single_item(data_item)
            
            if result is None:
                # 处理失败
                failed_count += 1
                result_manager.log_failure(narration_id, "处理失败")
            else:
                # 保存结果
                result_manager.save_result(
                    data_item,
                    result['action_dict'],
                    result['selected_noun_keys']
                )
                processed_count += 1
        
        # 打印最终统计
        print(f"\n{'='*60}")
        print(f"处理完成！")
        print(f"  总数据量: {total}")
        print(f"  成功处理: {processed_count}")
        print(f"  跳过: {skipped_count}")
        print(f"  失败: {failed_count}")
        print(f"{'='*60}")
