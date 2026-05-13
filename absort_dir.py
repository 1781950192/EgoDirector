import pandas as pd
import os
import shutil
import numpy as np


def extract_frames_to_new_folder(csv_file_path: str, original_root: str, new_root: str):
    """
    根据CSV文件抽取所需的图片，并复制到新文件夹中，按照原文件夹的格式组织。

    参数:
    - csv_file_path: CSV文件的路径，例如 '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_val_sampled_dataset_300.csv'
    - original_root: 原数据集的根目录，例如 '/home/will/Mydata/EPIC-KITCHENS'
    - new_root: 新文件夹的根目录，例如 '/home/will/Mydata/EPIC-KITCHENS_EXTRACTED'

    过程:
    1. 读取CSV文件，获取每个条目的participant_id, video_id, start_frame, stop_frame。
    2. 对于每个条目，计算需要抽取的帧（这里假设抽取8个均匀分布的帧，与原脚本一致）。
    3. 从原路径复制这些帧到新路径，保持相同的目录结构：new_root/participant_id/rgb_frames/video_id/frame_xxxxxxxxxx.jpg
    4. 如果目标目录不存在，会自动创建。
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path, on_bad_lines='skip')
    data_list = df.to_dict(orient='records')

    for data in data_list:
        # 原帧目录
        original_base_path = os.path.join(original_root, data['participant_id'], 'rgb_frames', data['video_id'])

        # 新帧目录
        new_base_path = os.path.join(new_root, data['participant_id'], 'rgb_frames', data['video_id'])

        # 创建新目录如果不存在
        os.makedirs(new_base_path, exist_ok=True)

        # 计算需要抽取的帧索引（与原脚本一致：从start_frame到stop_frame-1均匀取8个）
        indices = np.linspace(data['start_frame'], data['stop_frame'] - 1, num=8).astype(int)

        for idx in indices:
            # 原帧路径
            original_frame_path = os.path.join(original_base_path, f"frame_{idx:010d}.jpg")

            # 新帧路径
            new_frame_path = os.path.join(new_base_path, f"frame_{idx:010d}.jpg")

            # 如果原文件存在，复制到新路径
            if os.path.exists(original_frame_path):
                shutil.copy(original_frame_path, new_frame_path)
                print(f"复制: {original_frame_path} -> {new_frame_path}")
            else:
                print(f"跳过: {original_frame_path} 不存在")


# 示例使用
if __name__ == '__main__':
    csv_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_val_sampled_dataset_300_test.csv'
    original_root = '/home/will/Mydata/EPIC-KITCHENS'
    new_root = '/home/will/Mydata/EPIC-KITCHENS_EXTRACTED_test'  # 可以修改为所需的路径

    extract_frames_to_new_folder(csv_file_path, original_root, new_root)