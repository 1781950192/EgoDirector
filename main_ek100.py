import os
import time
import argparse

from dataset_adapters import EK100Adapter
from pipeline import (
    extract_frames_with_indices,
    select_sub_frames,
    map_indices_to_paths,
    cleanup_temp_files,
    process_all_sequentially,
    ResultWriter,
)
from action_pipeline import (
    action_recognition,
    action_recognition_base1,
    action_recognition_base2,
    action_recognition_base3,
)
from context_manager import overlay_context_base


def process_all_sequentially_ek100(
    csv_file_path,
    output_txt_path,
    base=0,
    use_hos=0,
    num_frames=32,
):
    adapter = EK100Adapter(
        csv_file_path=csv_file_path,
        base_path_root='/mnt/data/xgl/mydata/ek100',
        hos_path_root='/mnt/data/xgl/mydata/ek100_hos/32frames',
    )
    data_list = adapter.load_data()
    if not data_list:
        return

    processed_ids = ResultWriter.load_processed_ids(output_txt_path)
    ResultWriter.init_output_file(output_txt_path)

    action_fns = {
        0: action_recognition,
        1: action_recognition_base1,
        2: action_recognition_base2,
        3: action_recognition_base3,
    }
    action_fn = action_fns.get(base, action_recognition)

    start_time = time.time()
    for idx, data in enumerate(data_list):
        narration_id = EK100Adapter.make_narration_id(data)
        if narration_id in processed_ids:
            continue

        if idx % 20 == 0:
            print(f"Progress: {idx}/{len(data_list)} elapsed {(time.time() - start_time):.1f}s processing: {narration_id}")

        video_path = adapter.get_video_path(data)
        if not video_path or not os.path.exists(video_path):
            print(f"Path does not exist: {video_path}")
            continue

        start_frame = int(data['start_frame'])
        stop_frame = int(data['stop_frame'])
        frame_indices, frame_paths = adapter.get_frame_paths_with_indices(
            video_path, start_frame, stop_frame, num_frames=num_frames
        )

        hos_paths = adapter.get_hos_paths(data, frame_indices, frame_indices)
        valid_frame_paths = [p for p in frame_paths if os.path.exists(p)]

        if use_hos == 0:
            frames_urls = valid_frame_paths + hos_paths
        else:
            frames_urls = valid_frame_paths

        print(f"Input {len(frames_urls)} images")

        reflect_dict, predicted_action_dict, selected_noun_keys = action_fn(frames_urls)
        ResultWriter.write_result(output_txt_path, data, predicted_action_dict, selected_noun_keys, narration_id)


def main():
    parser = argparse.ArgumentParser(description="Action recognition - single-threaded")
    parser.add_argument('--base', type=int, default=0, help="0: main model, 1: 1llm, 2: 2llm, 3: 3llm")
    parser.add_argument('--use_hos', type=int, default=0, help="0: use HOS, 1: do not use HOS")
    parser.add_argument('--num_frames', type=int, default=32)
    args = parser.parse_args()

    csv_file_path = '/mnt/data/xgl/mydata/ek100/EPIC_100_validation.csv'
    output_txt_path = 'ek100_val_8B_32frames.txt'

    process_all_sequentially_ek100(
        csv_file_path, output_txt_path,
        args.base, args.use_hos,
        num_frames=args.num_frames
    )

    overlay_context_base()


if __name__ == '__main__':
    import faulthandler
    faulthandler.enable()
    main()
