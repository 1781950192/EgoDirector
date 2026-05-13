import os
import time
import argparse

import numpy as np

from dataset_adapters import EGTEAAdapter
from pipeline import (
    extract_frames_with_indices,
    select_sub_frames,
    map_indices_to_paths,
    cleanup_temp_files,
    ResultWriter,
)
from action_pipeline import (
    action_recognition,
    action_recognition_base1,
    action_recognition_base2,
)
from context_manager import overlay_context_base


def process_all_sequentially_egtea(
    label_file_path,
    video_base_dir,
    hos_path,
    output_txt_path,
    base=0,
    use_hos=0,
    num_frames=16,
):
    adapter = EGTEAAdapter(label_file_path, video_base_dir, hos_path)
    data_list = adapter.load_data()
    if not data_list:
        print("No data to process")
        return

    processed_ids = ResultWriter.load_processed_ids(output_txt_path)
    ResultWriter.init_output_file(output_txt_path)

    action_fns = {
        0: action_recognition,
        1: action_recognition_base1,
        2: action_recognition_base2,
    }
    action_fn = action_fns.get(base, action_recognition)

    start_time = time.time()
    for idx, data in enumerate(data_list):
        narration_id = EGTEAAdapter.make_narration_id(data)
        if narration_id in processed_ids:
            continue

        if idx % 20 == 0:
            print(f"Progress: {idx}/{len(data_list)} elapsed {(time.time() - start_time):.1f}s current: {narration_id}")

        video_path = adapter.get_video_path(data)
        if not video_path:
            print(f"Skip {narration_id}: video not found")
            continue

        # First extract 32 frames as the base
        frame_indices_32, frame_paths_32 = extract_frames_with_indices(
            video_path, num_frames=32
        )

        # Select the target frames from the 32 frames
        selected_frame_indices = select_sub_frames(frame_indices_32, num_frames)
        frame_paths = map_indices_to_paths(selected_frame_indices, frame_indices_32, frame_paths_32)

        # Build the HOS paths
        hos_image_paths = []
        if use_hos == 0:
            hos_image_paths = adapter.get_hos_paths(data, selected_frame_indices, frame_indices_32)

        frames_urls = frame_paths + hos_image_paths if use_hos == 0 else frame_paths
        print(f"Total inputs: {len(frames_urls)} (video frames: {len(frame_paths)}, HOS: {len(hos_image_paths)}) -> {narration_id}")

        reflect_dict, predicted_action_dict, selected_noun_keys = action_fn(frames_urls)
        ResultWriter.write_result(output_txt_path, data, predicted_action_dict, selected_noun_keys, narration_id)
        cleanup_temp_files(frame_paths)

    print(f"EGTEA processing finished! Total time: {(time.time() - start_time):.2f} s")


def main():
    parser = argparse.ArgumentParser(description="EGTEA+ action recognition")
    parser.add_argument('--label_file', type=str,
                        default='/mnt/data/xgl/mydata/EGTEA++/EGTEA/Action_Annotations/test_split1.txt')
    parser.add_argument('--video_base_dir', type=str,
                        default='/mnt/data/xgl/mydata/EGTEA++/EGTEA/Trimmed_Action_Clips/trimmed_actions/cropped_clips')
    parser.add_argument('--hos_path', type=str,
                        default='/mnt/data/xgl/mydata/EGTEA++/egtea_hos/32frames')
    parser.add_argument('--output_txt_path', type=str, default=None,
                        help='Path of the output file; if not given, it is generated from num_frames')
    parser.add_argument('--base', type=int, default=0, choices=[0, 1, 2],
                        help="0: main model, 1: 1llm, 2: 3llm")
    parser.add_argument('--use_hos', type=int, default=0, choices=[0, 1],
                        help="0: use HOS, 1: do not use HOS")
    parser.add_argument('--num_frames', type=int, default=32, help='Number of frames')
    args = parser.parse_args()

    args.output_txt_path = 'egtea_results_32frames.txt'

    print(f"Starting the EGTEA experiment - frames: {args.num_frames}, output file: {args.output_txt_path}")

    process_all_sequentially_egtea(
        label_file_path=args.label_file,
        video_base_dir=args.video_base_dir,
        hos_path=args.hos_path,
        output_txt_path=args.output_txt_path,
        base=args.base,
        use_hos=args.use_hos,
        num_frames=args.num_frames,
    )

    overlay_context_base()


if __name__ == '__main__':
    import faulthandler
    faulthandler.enable()
    main()
