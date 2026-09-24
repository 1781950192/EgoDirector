import os
from typing import List, Dict, Any

import pandas as pd
from pipeline import extract_frames_with_indices, HOSPathBuilder


# ==================== EK100 Adapter ====================

class EK100Adapter:
    """EK100 dataset adapter: load the data from a CSV file and parse the frame paths."""

    def __init__(self, csv_file_path: str, base_path_root: str, hos_path_root: str):
        self.csv_file_path = csv_file_path
        self.base_path_root = base_path_root
        self.hos_path_root = hos_path_root

    def load_data(self) -> List[Dict[str, Any]]:
        df = pd.read_csv(self.csv_file_path, on_bad_lines='skip')
        data_list = df.to_dict('records')
        print(f"{len(data_list)} EK100 samples in total")
        return data_list

    def get_video_path(self, data: Dict[str, Any]) -> str:
        return os.path.join(
            self.base_path_root,
            data['participant_id'],
            'rgb_frames',
            data['video_id']
        )

    def get_frame_paths(self, video_path: str, start_frame: int = 0, stop_frame: int = 1, num_frames: int = 32):
        """Build the EK100 frame paths: take num_frames frames uniformly from start_frame to stop_frame."""
        return self.get_frame_paths_with_indices(video_path, start_frame, stop_frame, num_frames)

    def get_hos_paths(self, data: Dict[str, Any], selected_indices: List[int], source_indices: List[int]) -> List[str]:
        return HOSPathBuilder.build_ek100_paths(
            data['participant_id'],
            data['video_id'],
            self.hos_path_root,
            selected_indices
        )

    @staticmethod
    def make_narration_id(data: Dict[str, Any]) -> str:
        return data.get('narration_id', '')

    def get_frame_paths_with_indices(self, video_path: str, start_frame: int, stop_frame: int, num_frames: int = 32):
        """Build the EK100 frame paths: take num_frames frames uniformly from start_frame to stop_frame."""
        import numpy as np
        indices32 = np.linspace(start_frame, stop_frame - 1, num=32).astype(int).tolist()
        ideal = np.linspace(start_frame, stop_frame - 1, num=num_frames).astype(int)
        indices = []
        for v in ideal:
            idx = int(indices32[np.argmin(np.abs(np.array(indices32) - v))])
            indices.append(idx)
        paths = [os.path.join(video_path, f"frame_{idx:010d}.jpg") for idx in indices]
        return indices, paths


# ==================== GTEA Adapter ====================

class GTEAAdapter:
    """GTEA dataset adapter: parse the label files and keep only the verb+noun format."""

    def __init__(self, labels_dir: str, videos_base_path: str, hos_path: str = ''):
        self.labels_dir = labels_dir
        self.videos_base_path = videos_base_path
        self.hos_path = hos_path

    def load_data(self) -> List[Dict[str, Any]]:
        data_list = []
        label_files = [f for f in os.listdir(self.labels_dir) if f.endswith('.txt')]
        print(f"Found {len(label_files)} GTEA label files")

        for label_file in label_files:
            label_path = os.path.join(self.labels_dir, label_file)
            video_name = os.path.basename(label_file).replace('.txt', '')

            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                action_part = line.split(' (')[0].strip() if ' (' in line else line.split('(')[0].strip()
                if '><' not in action_part:
                    continue  # Skip single <noun> annotations

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

        print(f"Parsed {len(data_list)} GTEA action instances in total")
        return data_list

    def get_video_path(self, data: Dict[str, Any]) -> str:
        path = os.path.join(self.videos_base_path, data['video_id'] + '.mp4')
        return path if os.path.exists(path) else None

    def get_hos_paths(self, data: Dict[str, Any], selected_indices: List[int], source_indices: List[int]) -> List[str]:
        # Find the position of the selected frame in source_indices
        selected_positions = [source_indices.index(idx) for idx in selected_indices]
        return HOSPathBuilder.build_gtea_paths(
            data['video_id'],
            self.hos_path,
            selected_positions
        )

    @staticmethod
    def make_narration_id(data: Dict[str, Any]) -> str:
        return f"{data['video_id']}_{data['start_frame']}_{data['stop_frame']}"


# ==================== EGTEA Adapter ====================

class EGTEAAdapter:
    """EGTEA dataset adapter: parse the split txt file."""

    def __init__(self, label_file_path: str, video_base_dir: str, hos_path: str = ''):
        self.label_file_path = label_file_path
        self.video_base_dir = video_base_dir
        self.hos_path = hos_path

    def load_data(self) -> List[Dict[str, Any]]:
        data_list = []
        with open(self.label_file_path, 'r') as f:
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
        print(f"Parsed {len(data_list)} videos from {self.label_file_path}")
        return data_list

    def get_video_path(self, data: Dict[str, Any]) -> str:
        parts = data['video_id'].split('-')
        if len(parts) < 3:
            return None
        subdir = '-'.join(parts[:3])
        video_path = os.path.join(self.video_base_dir, subdir, data['video_id'] + '.mp4')
        return video_path if os.path.exists(video_path) else None

    def get_hos_paths(self, data: Dict[str, Any], selected_indices: List[int], source_indices: List[int]) -> List[str]:
        selected_positions = [source_indices.index(idx) for idx in selected_indices]
        return HOSPathBuilder.build_egtea_paths(
            data['video_id'],
            self.hos_path,
            selected_positions
        )

    @staticmethod
    def make_narration_id(data: Dict[str, Any]) -> str:
        return data.get('narration_id', '')
