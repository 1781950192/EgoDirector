import pandas as pd
import os
import numpy as np
import json
import argparse
from utils import *
from moduls import *
from VISOR_HOS.visor_det import vis_hos
from multiprocessing import Pool, Manager
from functools import partial

wrong_set = Manager().list()  # 使用Manager创建进程安全的共享列表

def process_row(data, noun_file_path, verb_file_path, output_txt_path, mode):
    """
    处理单行数据，执行动作识别并保存结果。
    """
    verb_matches = []
    noun_matches = []
    overall_matches = []
    local_wrong_set = []  # 每个进程的局部错误集合

    # 创建进程特定的输出文件
    output_txt_path_proc = f"{output_txt_path.rsplit('.', 1)[0]}_proc_{os.getpid()}.txt"
    with open(output_txt_path_proc, 'w', encoding='utf-8') as f:
        base_path = os.path.join('/home/will/Mydata/EPIC-KITCHENS', data['participant_id'], 'rgb_frames', data['video_id'])
        indices = np.linspace(data['start_frame'], data['stop_frame'] - 1, num=8).astype(int)
        frame_paths = [os.path.join(base_path, f"frame_{idx:010d}.jpg") for idx in indices]
        frames_urls = [image_to_base64(path) for path in frame_paths if os.path.exists(path)]
        images_box = vis_hos(frame_paths)
        frames_urls = frames_urls + images_box

        if not frames_urls:
            print(f"跳过 {data['video_id']}：没有有效的帧文件")
            return [], [], [], []

        predicted_action_dict, noun_reason, action_reason = action_recognition(frames_urls, noun_file_path, verb_file_path)

        predicted_action = predicted_action_dict.get("action", "")
        pred_noun = predicted_action_dict.get("noun", "")
        pred_verb = predicted_action_dict.get("verb", "")

        print(f"预测的动作：{predicted_action_dict}")
        narration = data['narration'].strip().lower()
        print(f"真实标签：{narration}")
        try:
            last_space_idx = narration.rindex(' ')
            gt_verb = data["verb"]
            gt_noun = data["noun"]
        except ValueError:
            gt_verb, gt_noun = narration, ''

        verb_match = 0
        noun_match = 0
        overall_match = 0
        best_action = predicted_action or "无有效动作预测"

        if predicted_action:
            try:
                pred_verb = pred_verb.strip()
                pred_noun = pred_noun.strip()

                verb_match = compare_text_similarity_v3(gt_verb, pred_verb)
                noun_match = compare_text_similarity_v3(gt_noun, pred_noun) if gt_noun else 0
                overall_match = 1 if verb_match and noun_match else 0
            except ValueError:
                print(f"动作格式错误：{predicted_action}")

        f.write(f"{narration},{best_action},{verb_match},{noun_match},{overall_match}\n")

        # 训练模式下收集错误预测
        if mode == "train" and overall_match == 0:
            selected_reason = predicted_action_dict.get("reason", "未知原因")
            selected_reason = {
                "noun_reason": noun_reason,
                "action_reason": action_reason,
                "selected_reason": selected_reason
            }
            error_reason = judge_error_reason(frames_urls, selected_reason, predicted_action, narration)
            print(error_reason)

            wrong_record = {
                "predicted_action": predicted_action,
                "correct_action": narration,
                "selected_reason": selected_reason,
                "error_reason": error_reason
            }
            local_wrong_set.append(wrong_record)

        verb_matches.append(verb_match)
        noun_matches.append(noun_match)
        overall_matches.append(overall_match)

    return verb_matches, noun_matches, overall_matches, local_wrong_set

def merge_results(output_files, final_output_path, verb_matches_all, noun_matches_all, overall_matches_all):
    """
    合并所有进程的输出文件并计算总体准确率。
    """
    with open(final_output_path, 'w', encoding='utf-8') as f_out:
        for output_file in output_files:
            with open(output_file, 'r', encoding='utf-8') as f_in:
                f_out.write(f_in.read())
            os.remove(output_file)  # 删除临时文件

        # 计算并写入总体准确率
        verb_accuracy = sum(verb_matches_all) / len(verb_matches_all) if verb_matches_all else 0
        noun_accuracy = sum(noun_matches_all) / len(noun_matches_all) if noun_matches_all else 0
        overall_accuracy = sum(overall_matches_all) / len(overall_matches_all) if overall_matches_all else 0
        f_out.write(f"准确率：动词={verb_accuracy:.4f}, 名词={noun_accuracy:.4f}, 总体={overall_accuracy:.4f}\n")
        print(f"准确率：动词={verb_accuracy:.4f}, 名词={noun_accuracy:.4f}, 总体={overall_accuracy:.4f}\n")

def save_and_compare_results(csv_file_path: str, output_txt_path: str, noun_file_path: str, verb_file_path: str, mode: str = "test", num_processes: int = 4):
    """
    保存并比较动作识别结果，支持训练和测试模式，使用多进程处理。

    参数：
        csv_file_path (str): 输入 CSV 文件路径。
        output_txt_path (str): 输出结果保存路径。
        noun_file_path (str): 名词类别 CSV 文件路径。
        verb_file_path (str): 动词类别 CSV 文件路径。
        mode (str): 'train' 表示训练模式（5 次迭代，保存提示），'test' 表示测试模式（1 次迭代，不保存提示）。
        num_processes (int): 使用的进程数。
    """
    # 根据模式确定迭代次数
    iterations = 5 if mode == "train" else 1

    for item in range(1, iterations + 1):
        # 训练模式下为每次迭代修改输出文件路径
        if mode == "train":
            output_txt_path_1, output_txt_path_2 = output_txt_path.split(".")
            output_txt_path_iter = output_txt_path_1 + f"{item}." + output_txt_path_2
        else:
            output_txt_path_iter = output_txt_path

        # 读取 CSV 文件
        df = pd.read_csv(csv_file_path, on_bad_lines='skip')
        data_list = df.to_dict(orient='records')

        # 将数据分成指定数量的份数
        n = len(data_list)
        chunk_size = n // num_processes + (1 if n % num_processes else 0)
        data_chunks = [data_list[i:i + chunk_size] for i in range(0, n, chunk_size)]

        # 创建进程池
        with Pool(processes=num_processes) as pool:
            # 准备多进程参数
            process_func = partial(process_row, noun_file_path=noun_file_path, verb_file_path=verb_file_path, output_txt_path=output_txt_path_iter, mode=mode)
            results = pool.map(process_func, data_list)  # 直接处理每行数据

        # 合并结果
        verb_matches_all = []
        noun_matches_all = []
        overall_matches_all = []
        output_files = []
        for i, (verb_matches, noun_matches, overall_matches, local_wrong_set) in enumerate(results):
            verb_matches_all.extend(verb_matches)
            noun_matches_all.extend(noun_matches)
            overall_matches_all.extend(overall_matches)
            wrong_set.extend(local_wrong_set)  # 合并错误集合
            output_files.append(f"{output_txt_path_iter.rsplit('.', 1)[0]}_proc_{i}.txt")

        # 合并输出文件并计算准确率
        merge_results(output_files, output_txt_path_iter, verb_matches_all, noun_matches_all, overall_matches_all)

        # 训练模式下保存错误预测到 JSON
        if mode == "train":
            with open(f"json_epic/wrong_set_{item}.json", 'w', encoding='utf-8') as json_f:
                json.dump(list(wrong_set), json_f, ensure_ascii=False, indent=4)

            # 训练模式下优化并保存提示
            with open(f"json_epic/wrong_set_{item}.json", 'r', encoding='utf-8') as json_wrong:
                data = json.load(json_wrong)
                print("开始总结错误...")
                out = conclusion_incorrect(data)
                print("开始优化提示...")
                out = optimize_prompts_using_wrong_set(out)

                # 保存优化后的提示
                prompt_dict = {}
                for prompt_type, prompt_text in [
                    ("select_noun_prompt", out["select_noun_prompt"]),
                    ("combine_actions_prompt", out["combine_actions_prompt"]),
                    ("select_actions_prompt", out["select_actions_prompt"])
                ]:
                    file_path = f"prompt/prompt_{item}/{prompt_type}.txt"
                    prompt_dict[f"{prompt_type}.txt"] = file_path
                    dir_path = os.path.dirname(file_path)
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(prompt_text)

            wrong_set[:] = []  # 清空 wrong_set 为下一次迭代做准备

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="运行动作识别的训练或测试模式")
    parser.add_argument('--train', action='store_true', help="以训练模式运行（5 次迭代，保存提示）")
    parser.add_argument('--num-processes', type=int, default=4, help="指定使用的进程数（默认：4）")
    args = parser.parse_args()

    # 文件路径
    csv_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_val_sampled_dataset_300.csv'
    noun_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_noun_classes.csv'
    verb_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_verb_classes.csv'
    output_txt_path = 'val_comparison_results_frames8.txt'

    # 根据命令行参数选择模式
    mode = "train" if args.train else "test"
    print(f"运行模式：{'训练模式' if mode == 'train' else '测试模式'}")
    print(f"使用进程数：{args.num_processes}")

    # 调用主函数
    save_and_compare_results(csv_file_path, output_txt_path, noun_file_path, verb_file_path, mode=mode, num_processes=args.num_processes)
if __name__ == '__main__':
    main()