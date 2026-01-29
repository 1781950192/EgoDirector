import pandas as pd
from utils import *
from moduls import *
# from GroundDino_api import *
from VISOR_HOS.visor_det import vis_hos

wrong_set = []

# 保存和比较结果
# 保存和比较结果（修改后的函数，添加错题集逻辑）
def save_and_compare_results(csv_file_path: str, output_txt_path: str, noun_file_path: str, verb_file_path: str):
    for item in range(1, 6):
        output_txt_path_1, output_txt_path_2 = output_txt_path.split(".")
        output_txt_path = output_txt_path_1 + f"{item}." + output_txt_path_2

        df = pd.read_csv(csv_file_path, on_bad_lines='skip')
        data_list = df.to_dict(orient='records')

        verb_matches = []
        noun_matches = []
        overall_matches = []

        with open(output_txt_path, 'w', encoding='utf-8') as f:
            for data in data_list:
                base_path = os.path.join('/home/will/Mydata/EPIC-KITCHENS', data['participant_id'], 'rgb_frames',
                                         data['video_id'])
                indices = np.linspace(data['start_frame'], data['stop_frame'] - 1, num=8).astype(int)
                frame_paths = [os.path.join(base_path, f"frame_{idx:010d}.jpg") for idx in indices]
                frames_urls = [image_to_base64(path) for path in frame_paths if os.path.exists(path)]
                images_box = vis_hos(frame_paths)
                frames_urls = frames_urls + images_box

                if not frames_urls:
                    print(f"跳过 {data['video_id']}：没有有效的帧文件")
                    continue

                predicted_action_dict, noun_reason, action_reason = action_recognition(frames_urls, noun_file_path,
                                                                                       verb_file_path)

                predicted_action = predicted_action_dict.get("action", "")
                pred_noun = predicted_action_dict.get("noun", "")
                pred_verb = predicted_action_dict.get("verb", "")

                print(f"预测的动作是：{predicted_action_dict}")
                narration = data['narration'].strip().lower()
                print(f"真实的标签是：{narration}")
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
                        print(f"动作格式错误: {predicted_action}")

                f.write(f"{narration},{best_action},{verb_match},{noun_match},{overall_match}\n")

                verb_matches.append(verb_match)
                noun_matches.append(noun_match)
                overall_matches.append(overall_match)

                # if overall_match == 0:
                #     selected_reason = predicted_action_dict.get("reason", "未知原因")
                #     selected_reason = {"挑选名词的原因": noun_reason, "组合动作的原因": action_reason,
                #                        "挑选动作的原因": selected_reason}
                #     error_reason = judge_error_reason(frames_urls, selected_reason, predicted_action, narration)
                #     print(error_reason)
                #
                #     wrong_record = {
                #         "predicted_action": predicted_action,
                #         "correct_action": narration,
                #         "selected_reason": selected_reason,
                #         "error_reason": error_reason
                #     }
                #     wrong_set.append(wrong_record)

            # 在 with 语句内写入准确率
            verb_accuracy = sum(verb_matches) / len(verb_matches) if verb_matches else 0
            noun_accuracy = sum(noun_matches) / len(noun_matches) if noun_matches else 0
            overall_accuracy = sum(overall_matches) / len(overall_matches) if overall_matches else 0
            f.write(f"准确率: 动词={verb_accuracy:.4f}, 名词={noun_accuracy:.4f}, 总体={overall_accuracy:.4f}\n")
            print(f"准确率: 动词={verb_accuracy:.4f}, 名词={noun_accuracy:.4f}, 总体={overall_accuracy:.4f}\n")

        with open("json_epic/wrong_set.json", 'w', encoding='utf-8') as json_f:
            json.dump(wrong_set, json_f, ensure_ascii=False, indent=4)

        # # 在处理完所有数据后，可以优化提示
        # with open("json_epic/wrong_set.json", 'r', encoding='utf-8') as json_wrong:
        #     data = json.load(json_wrong)
        #     print("开始总结错误")
        #     out = conclusion_incorrect(data)
        #     print("开始优化提示")
        #     out = optimize_prompts_using_wrong_set(out)
        #
        #     text = out["select_noun_prompt"]
        #     file_path = f"prompt/prompt_{item}/select_noun_prompt.txt"  # 文件路径（可以是相对路径或绝对路径）
        #     prompt_dict["select_noun_prompt.txt"] = file_path
        #     # 递归创建目录（仅目录部分）
        #     dir_path = os.path.dirname(file_path)
        #     if not os.path.exists(dir_path):
        #         os.makedirs(dir_path)
        #     with open(file_path, 'w', encoding='utf-8') as file:  # 'w' 表示写入模式
        #         file.write(text)
        #
        #     text = out["combine_actions_prompt"]
        #     file_path = f"prompt/prompt_{item}/combine_actions_prompt.txt"  # 文件路径（可以是相对路径或绝对路径）
        #     prompt_dict["combine_actions_prompt.txt"] = file_path
        #     with open(file_path, 'w', encoding='utf-8') as file:  # 'w' 表示写入模式
        #         file.write(text)
        #
        #     text = out["select_actions_prompt"]
        #     file_path = f"prompt/prompt_{item}/select_actions_prompt.txt"  # 文件路径（可以是相对路径或绝对路径）
        #     prompt_dict["select_actions_prompt.txt"] = file_path
        #     with open(file_path, 'w', encoding='utf-8') as file:  # 'w' 表示写入模式
        #         file.write(text)
        wrong_set.clear()

# 示例使用
if __name__ == '__main__':
    csv_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_val_sampled_dataset_300.csv'
    noun_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_noun_classes.csv'
    verb_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_verb_classes.csv'
    output_txt_path = 'val_comparison_results_frames8_new_prompt5.txt'

    save_and_compare_results(csv_file_path, output_txt_path, noun_file_path, verb_file_path)


# import pandas as pd
# import os
# import numpy as np
# import json
# from pathlib import Path
# from utils import *
# from moduls_2 import *
# from VISOR_HOS.visor_det import vis_hos
#
# wrong_set = []
# prompt_dict = {}
#
# def save_and_compare_results(csv_file_path: str, output_txt_path: str, noun_file_path: str, verb_file_path: str):
#     for item in range(1, 6):
#         output_txt_path_1, output_txt_path_2 = output_txt_path.split(".")
#         output_txt_path = output_txt_path_1 + f"{item}." + output_txt_path_2
#
#         df = pd.read_csv(csv_file_path, on_bad_lines='skip')
#         data_list = df.to_dict(orient='records')
#
#         verb_matches = []
#         noun_matches = []
#         overall_matches = []
#
#         with open(output_txt_path, 'w', encoding='utf-8') as f:
#             for data in data_list:
#                 base_path = os.path.join('/home/will/Mydata/EPIC-KITCHENS', data['participant_id'], 'rgb_frames',
#                                        data['video_id'])
#                 indices = np.linspace(data['start_frame'], data['stop_frame'] - 1, num=8).astype(int)
#                 frame_paths = [os.path.join(base_path, f"frame_{idx:010d}.jpg") for idx in indices]
#                 frames_urls = [image_to_base64(path) for path in frame_paths if os.path.exists(path)]
#                 images_box = vis_hos(frame_paths)
#                 frames_urls = frames_urls + images_box
#
#                 if not frames_urls:
#                     print(f"跳过 {data['video_id']}：没有有效的帧文件")
#                     continue
#
#                 predicted_action_dict, noun_reason, action_reason = action_recognition(frames_urls, noun_file_path,
#                                                                                      verb_file_path)
#
#                 predicted_action = predicted_action_dict.get("action", "")
#                 pred_noun = predicted_action_dict.get("noun", "")
#                 pred_verb = predicted_action_dict.get("verb", "")
#
#                 print(f"预测的动作是：{predicted_action_dict}")
#                 narration = data['narration'].strip().lower()
#                 print(f"真实的标签是：{narration}")
#                 try:
#                     last_space_idx = narration.rindex(' ')
#                     gt_verb = data["verb"]
#                     gt_noun = data["noun"]
#                 except ValueError:
#                     gt_verb, gt_noun = narration, ''
#
#                 verb_match = 0
#                 noun_match = 0
#                 overall_match = 0
#                 best_action = predicted_action or "无有效动作预测"
#
#                 if predicted_action:
#                     try:
#                         pred_verb = pred_verb.strip()
#                         pred_noun = pred_noun.strip()
#
#                         verb_match = compare_text_similarity_v3(gt_verb, pred_verb)
#                         noun_match = compare_text_similarity_v3(gt_noun, pred_noun) if gt_noun else 0
#                         overall_match = 1 if verb_match and noun_match else 0
#                     except ValueError:
#                         print(f"动作格式错误: {predicted_action}")
#
#                 f.write(f"{narration},{best_action},{verb_match},{noun_match},{overall_match}\n")
#
#                 verb_matches.append(verb_match)
#                 noun_matches.append(noun_match)
#                 overall_matches.append(overall_match)
#
#                 if overall_match == 0:
#                     selected_reason = predicted_action_dict.get("reason", "未知原因")
#                     selected_reason = {
#                         "挑选名词的原因": noun_reason,
#                         "组合动作的原因": action_reason,
#                         "挑选动作的原因": selected_reason
#                     }
#                     error_reason = judge_error_reason(frames_urls, selected_reason, predicted_action, narration)
#                     print(error_reason)
#
#                     wrong_record = {
#                         "predicted_action": predicted_action,
#                         "correct_action": narration,
#                         "selected_reason": selected_reason,
#                         "error_reason": error_reason
#                     }
#                     wrong_set.append(wrong_record)
#
#                     # 当错误集达到30个时，触发提示优化
#                     if len(wrong_set) >= 30:
#                         print("错误集达到30个，开始优化提示")
#                         with open("json_epic/wrong_set.json", 'w', encoding='utf-8') as json_f:
#                             json.dump(wrong_set, json_f, ensure_ascii=False, indent=4)
#
#                         with open("json_epic/wrong_set.json", 'r', encoding='utf-8') as json_wrong:
#                             data = json.load(json_wrong)
#                             print("开始总结错误")
#                             out = conclusion_incorrect(data)
#                             print("开始优化提示")
#                             out = optimize_prompts_using_wrong_set(out)
#
#                             # 保存优化后的提示
#                             for prompt_key, prompt_text in [
#                                 ("select_noun_prompt", out["select_noun_prompt"]),
#                                 ("combine_actions_prompt", out["combine_actions_prompt"]),
#                                 ("select_actions_prompt", out["select_actions_prompt"])
#                             ]:
#                                 file_path = f"prompt/prompt_{item}/{prompt_key}.txt"
#                                 prompt_dict[f"{prompt_key}.txt"] = file_path
#                                 dir_path = os.path.dirname(file_path)
#                                 if not os.path.exists(dir_path):
#                                     os.makedirs(dir_path)
#                                 with open(file_path, 'w', encoding='utf-8') as file:
#                                     file.write(prompt_text)
#
#                         # 清空错误集以继续收集新的错误
#                         wrong_set.clear()
#
#             # 在 with 语句内写入准确率
#             verb_accuracy = sum(verb_matches) / len(verb_matches) if verb_matches else 0
#             noun_accuracy = sum(noun_matches) / len(noun_matches) if noun_matches else 0
#             overall_accuracy = sum(overall_matches) / len(overall_matches) if overall_matches else 0
#             f.write(f"准确率: 动词={verb_accuracy:.4f}, 名词={noun_accuracy:.4f}, 总体={overall_accuracy:.4f}\n")
#             print(f"准确率: 动词={verb_accuracy:.4f}, 名词={noun_accuracy:.4f}, 总体={overall_accuracy:.4f}\n")
#
#         # 处理剩余的错误（如果错误少于30个）
#         if wrong_set:
#             print("处理剩余的错误集，开始优化提示")
#             with open("json_epic/wrong_set.json", 'w', encoding='utf-8') as json_f:
#                 json.dump(wrong_set, json_f, ensure_ascii=False, indent=4)
#
#             with open("json_epic/wrong_set.json", 'r', encoding='utf-8') as json_wrong:
#                 data = json.load(json_wrong)
#                 print("开始总结错误")
#                 out = conclusion_incorrect(data)
#                 print("开始优化提示")
#                 out = optimize_prompts_using_wrong_set(out)
#
#                 # 保存优化后的提示
#                 for prompt_key, prompt_text in [
#                     ("select_noun_prompt", out["select_noun_prompt"]),
#                     ("combine_actions_prompt", out["combine_actions_prompt"]),
#                     ("select_actions_prompt", out["select_actions_prompt"])
#                 ]:
#                     file_path = f"prompt/prompt_{item}/{prompt_key}.txt"
#                     prompt_dict[f"{prompt_key}.txt"] = file_path
#                     dir_path = os.path.dirname(file_path)
#                     if not os.path.exists(dir_path):
#                         os.makedirs(dir_path)
#                     with open(file_path, 'w', encoding='utf-8') as file:
#                         file.write(prompt_text)
#
#             # 清空错误集
#             wrong_set.clear()
#
# if __name__ == '__main__':
#     csv_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_val_sampled_dataset_300.csv'
#     noun_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_noun_classes.csv'
#     verb_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_verb_classes.csv'
#     detections_root = Path('/home/will/Mycode/MCP_action/epic_det/hand-objects/hand-objects')
#     output_txt_path = 'val_comparison_results_frames8_new.txt'
#
#     save_and_compare_results(csv_file_path, output_txt_path, noun_file_path, verb_file_path)