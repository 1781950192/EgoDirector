import os
import tempfile
import time
from typing import List, Tuple, Callable, Dict, Any

import cv2
import numpy as np


# ==================== Frame Extraction ====================

def extract_frames_with_indices(
    video_path: str,
    start_frame: int = 0,
    stop_frame: int = -1,
    num_frames: int = 16
) -> Tuple[List[int], List[str]]:
    """Uniformly extract num_frames frames and return (frame index list, temp file path list).

    start_frame / stop_frame default to 0 / -1, which means the whole video is read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return [], []

    actual_stop = min(stop_frame, total_frames) if stop_frame > 0 else total_frames
    indices = np.linspace(start_frame, actual_stop - 1, num=num_frames, dtype=int).tolist()

    temp_files = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            cv2.imwrite(temp_file.name, frame)
            temp_files.append(temp_file.name)

    cap.release()
    return indices, temp_files


def select_sub_frames(frame_indices: List[int], target_num_frames: int) -> List[int]:
    """Uniformly pick target_num_frames indices from a set of frame indices."""
    if target_num_frames >= len(frame_indices):
        return frame_indices

    ideal_positions = np.linspace(0, len(frame_indices) - 1, num=target_num_frames, dtype=int)
    return [frame_indices[pos] for pos in ideal_positions]


def map_indices_to_paths(
    selected_indices: List[int],
    source_indices: List[int],
    source_paths: List[str]
) -> List[str]:
    """Map the selected indices to the corresponding temp file paths."""
    index_to_path = dict(zip(source_indices, source_paths))
    return [index_to_path[idx] for idx in selected_indices]


def cleanup_temp_files(temp_files: List[str]):
    for f in temp_files:
        if os.path.exists(f):
            os.unlink(f)


# ==================== HOS Path Builder ====================

class HOSPathBuilder:
    """Build the HOS (Hand-Object Segmentation) image paths, with several fallback strategies."""

    @staticmethod
    def build_egtea_paths(
        video_id: str,
        hos_base: str,
        selected_positions: List[int]
    ) -> List[str]:
        """EGTEA style: try video_id/video_id/ first, then fall back to subdir/video_id/."""
        parts = video_id.split('-')
        subdir = '-'.join(parts[:3]) if len(parts) >= 3 else ''

        # Strategy A: hos_base / video_id / video_id /
        paths_a = [
            os.path.join(hos_base, video_id, video_id,
                        f"{video_id}_vis_hos_{pos:03d}.jpg")
            for pos in selected_positions
        ]
        existing_a = [p for p in paths_a if os.path.exists(p)]
        if existing_a:
            return existing_a

        # Strategy B: hos_base / subdir / video_id /
        if subdir:
            paths_b = [
                os.path.join(hos_base, subdir, video_id,
                            f"{video_id}_vis_hos_{pos:03d}.jpg")
                for pos in selected_positions
            ]
            existing_b = [p for p in paths_b if os.path.exists(p)]
            if existing_b:
                return existing_b

        return []

    @staticmethod
    def build_gtea_paths(
        video_id: str,
        hos_base: str,
        selected_positions: List[int]
    ) -> List[str]:
        """GTEA style: hos_base / video_id / video_id_vis_hos_NNN.jpg."""
        paths = [
            os.path.join(hos_base, video_id,
                        f"{video_id}_vis_hos_{pos:03d}.jpg")
            for pos in selected_positions
        ]
        return [p for p in paths if os.path.exists(p)]

    @staticmethod
    def build_ek100_paths(
        participant_id: str,
        video_id: str,
        hos_base: str,
        indices: List[int]
    ) -> List[str]:
        """EK100 style: hos_base / participant_id / rgb_frames / video_id / frame_IDX_pred.jpg."""
        base = os.path.join(hos_base, participant_id, 'rgb_frames', video_id)
        paths = [
            os.path.join(base, f"frame_{idx:010d}_pred.jpg")
            for idx in indices
        ]
        return [p for p in paths if os.path.exists(p)]


# ==================== Result Writer ====================

class ResultWriter:
    """Append the recognition result to a CSV file."""

    HEADER = "narration_id,verbs,nouns,actions,confidences,selected_noun_keys\n"

    @staticmethod
    def write_result(
        output_path: str,
        data: Dict[str, Any],
        predicted_action_dict: Dict[str, Any],
        selected_noun_keys,
        narration_id: str = None
    ):
        """Append one result row to the output file."""
        if narration_id is None:
            narration_id = data.get('narration_id', '')

        with open(output_path, 'a', encoding='utf-8') as f:
            actions = predicted_action_dict.get("actions", [])
            try:
                sorted_actions = sorted(actions, key=lambda x: x.get("confidence", 0), reverse=True)
            except Exception:
                sorted_actions = []

            verbs = ";".join([a.get("verb", "").strip() for a in sorted_actions if a.get("verb")]) or "no verb"
            nouns = ";".join([a.get("noun", "").strip() for a in sorted_actions if a.get("noun")]) or "no noun"
            actions_str = ";".join([a.get("action", "").strip() for a in sorted_actions if a.get("action")]) or "no action"
            confidences = ";".join([str(a.get("confidence", 0)) for a in sorted_actions if a.get("confidence")]) or "0.0"

            if isinstance(selected_noun_keys, list):
                selected_noun_keys_str = ";".join(map(str, selected_noun_keys))
            elif selected_noun_keys:
                selected_noun_keys_str = str(selected_noun_keys)
            else:
                selected_noun_keys_str = "no selected noun"

            f.write(f"{narration_id},{verbs},{nouns},{actions_str},{confidences},{selected_noun_keys_str}\n")

    @staticmethod
    def init_output_file(output_path: str):
        """Initialize the output file (write the header)."""
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(ResultWriter.HEADER)

    @staticmethod
    def load_processed_ids(output_path: str) -> set:
        """Read the set of already processed narration_id values."""
        processed = set()
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                next(f, None)  # Skip the header
                for line in f:
                    if line.strip():
                        processed.add(line.split(',', 1)[0].strip())
        return processed


# ==================== Shared Processing Loop ====================

def process_all_sequentially(
    data_list: list,
    output_txt_path: str,
    base: int = 0,
    use_hos: int = 0,
    num_frames: int = 16,
    get_video_path_fn=None,
    get_frame_paths_fn=None,
    get_hos_paths_fn=None,
    get_narration_id_fn=None,
    action_recognition_fn=None,
):
    """Generic sequential processing loop.

    Args:
        data_list: List of samples to process.
        output_txt_path: Path of the output CSV file.
        base: 0=main model, 1=base1, 2=base2, 3=base3
        use_hos: 0=use HOS, 1=do not use
        num_frames: Target number of frames.
        get_video_path_fn: (data) -> video_path | None
        get_frame_paths_fn: (video_path) -> (indices, temp_paths)
        get_hos_paths_fn: (data, frame_indices, source_indices) -> [hos_path]
        get_narration_id_fn: (data) -> narration_id
        action_recognition_fn: Recognition function imported from action_pipeline.
    """
    from action_pipeline import (
        action_recognition,
        action_recognition_base1,
        action_recognition_base2,
        action_recognition_base3,
    )

    start_time = time.time()
    if not data_list:
        print("No data to process")
        return

    processed_ids = ResultWriter.load_processed_ids(output_txt_path)
    ResultWriter.init_output_file(output_txt_path)

    if action_recognition_fn is None:
        action_fns = {
            0: action_recognition,
            1: action_recognition_base1,
            2: action_recognition_base2,
            3: action_recognition_base3,
        }
        action_recognition_fn = action_fns.get(base, action_recognition)

    for idx, data in enumerate(data_list):
        narration_id = get_narration_id_fn(data) if get_narration_id_fn else data.get('narration_id', '')
        if narration_id in processed_ids:
            continue

        if idx % 20 == 0:
            print(f"Progress: {idx}/{len(data_list)} elapsed {(time.time() - start_time):.1f}s current: {narration_id}")

        # Get the video path
        video_path = get_video_path_fn(data) if get_video_path_fn else None
        if not video_path:
            print(f"Skip {narration_id}: video not found")
            continue

        # Extract frames (first extract 32 frames as the base)
        if get_frame_paths_fn:
            frame_indices_all, frame_paths_all = get_frame_paths_fn(video_path)
        else:
            frame_indices_all, frame_paths_all = extract_frames_with_indices(
                video_path, num_frames=32
            )

        # Select the target frames
        selected_indices = select_sub_frames(frame_indices_all, num_frames)
        frame_paths = map_indices_to_paths(selected_indices, frame_indices_all, frame_paths_all)

        # Build the HOS paths
        hos_image_paths = []
        if use_hos == 0 and get_hos_paths_fn:
            hos_image_paths = get_hos_paths_fn(data, selected_indices, frame_indices_all)

        frames_urls = frame_paths + hos_image_paths if use_hos == 0 else frame_paths
        print(f"Total inputs: {len(frames_urls)} (video frames: {len(frame_paths)}, HOS: {len(hos_image_paths)}) -> {narration_id}")

        # Action recognition
        reflect_dict, predicted_action_dict, selected_noun_keys = action_recognition_fn(
            frames_urls, None)

        ResultWriter.write_result(output_txt_path, data, predicted_action_dict, selected_noun_keys, narration_id)
        cleanup_temp_files(frame_paths)

    print(f"Processing finished! Total time: {(time.time() - start_time):.2f} s")
