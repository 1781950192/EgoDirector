# import pandas as pd
# import os
# import numpy as np
# from multiprocessing import Pool, Manager
# from functools import partial
# import glob
# import argparse
#
# # 假设这些函数在原有的 'utils' 和 'moduls' 模块中定义
# from utils import image_to_base64
# from moduls import action_recognition
# from VISOR_HOS.visor_det import vis_hos
#
# def process_row(data, noun_file_path, verb_file_path, output_txt_path):
#     """
#     处理单行数据，执行动作识别并保存结果，将所有动作、动词和名词按置信度排序写入同一行。
#     """
#     # 创建进程特定的输出文件
#     output_txt_path_proc = f"{output_txt_path.rsplit('.', 1)[0]}_proc_{os.getpid()}.txt"
#     with open(output_txt_path_proc, 'a', encoding='utf-8') as f:
#         base_path = os.path.join('/home/will/Mydata/EPIC-KITCHENS', data['participant_id'], 'rgb_frames', data['video_id'])
#         indices = np.linspace(data['start_frame'], data['stop_frame'] - 1, num=8).astype(int)
#         frame_paths = [os.path.join(base_path, f"frame_{idx:010d}.jpg") for idx in indices]
#         frames_urls = [image_to_base64(path) for path in frame_paths if os.path.exists(path)]
#         images_box = vis_hos(frame_paths)
#         frames_urls = frames_urls + images_box
#
#         if not frames_urls:
#             print(f"跳过 {data['video_id']}：没有有效的帧文件")
#             return output_txt_path_proc
#
#         try:
#             reflect_dict, predicted_action_dict, noun_reason, action_reason = action_recognition(frames_urls, noun_file_path, verb_file_path)
#         except Exception as e:
#             print(f"action_recognition 错误：{e}")
#             return output_txt_path_proc
#
#         correct_approach_one = reflect_dict["correct_approach_one"],
#         correct_approach_two = reflect_dict["correct_approach_two"],
#         correct_approach_three = reflect_dict["correct_approach_three"],
#
#         # 按置信度排序动作
#         actions = predicted_action_dict.get("actions", [])
#         sorted_actions = sorted(actions, key=lambda x: x.get("confidence", 0), reverse=True)
#
#         # 将所有动作、动词、名词和置信度写入同一行
#         verbs = ";".join([action.get("verb", "").strip() for action in sorted_actions]) or "无动词"
#         nouns = ";".join([action.get("noun", "").strip() for action in sorted_actions]) or "无名词"
#         actions_str = ";".join([action.get("action", "").strip() for action in sorted_actions]) or "无动作"
#         confidences = ";".join([str(action.get("confidence", 0.0)) for action in sorted_actions]) or "0.0"
#         f.write(f"{data['narration_id']},{data['participant_id']},{data['video_id']},{data['narration_timestamp']},{data['start_timestamp']},{data['stop_timestamp']},{data['start_frame']},{data['stop_frame']},{verbs},{nouns},{actions_str},{confidences}\n")
#
#     return output_txt_path_proc
#
# def merge_results(output_files, final_output_path):
#     """
#     合并所有进程的输出文件。
#     """
#     with open(final_output_path, 'w', encoding='utf-8') as f_out:
#         for output_file in output_files:
#             if os.path.exists(output_file):
#                 with open(output_file, 'r', encoding='utf-8') as f_in:
#                     f_out.write(f_in.read())
#                 os.remove(output_file)
#             else:
#                 print(f"警告：文件 {output_file} 不存在，跳过。")
#
# def save_and_test(csv_file_path, output_txt_path, noun_file_path, verb_file_path, num_processes=4):
#     """
#     在测试模式下保存动作识别结果（无标签模式）。
#     """
#     print("使用无标签 CSV 模式")
#
#     # 读取 CSV 文件
#     df = pd.read_csv(csv_file_path, on_bad_lines='skip')
#     data_list = df.to_dict(orient='records')
#
#     output_files_set = set()
#
#     try:
#         # 创建进程池
#         with Pool(processes=num_processes) as pool:
#             process_func = partial(process_row, noun_file_path=noun_file_path, verb_file_path=verb_file_path,
#                                   output_txt_path=output_txt_path)
#             results = pool.map(process_func, data_list)
#
#         # 收集输出文件路径
#         for output_txt_path_proc in results:
#             if output_txt_path_proc:
#                 output_files_set.add(output_txt_path_proc)
#
#     except Exception as e:
#         print(f"处理过程中发生错误：{e}")
#         temp_pattern = f"{output_txt_path.rsplit('.', 1)[0]}_proc_*.txt"
#         found_files = glob.glob(temp_pattern)
#         output_files_set.update(found_files)
#         print(f"找到 {len(found_files)} 个已生成的临时文件，将尝试合并。")
#
#     finally:
#         output_files = list(output_files_set)
#         merge_results(output_files, output_txt_path)
#
# def main():
#     parser = argparse.ArgumentParser(description="运行动作识别的测试模式（无标签）")
#     parser.add_argument('--num_processes', type=int, default=9, help="指定使用的进程数（默认：1）")
#     args = parser.parse_args()
#
#     # 文件路径
#     csv_file_path = '/home/will/Mycode/action_agent/EPIC-KITCHENS_EXTRACTED/EPIC_100_val_sampled_dataset_300.csv'
#     noun_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_noun_classes.csv'
#     verb_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_verb_classes.csv'
#     output_txt_path = 'test_comparison_results_frames8.txt'
#
#     print(f"以测试模式运行，使用 {args.num_processes} 个进程")
#     save_and_test(csv_file_path, output_txt_path, noun_file_path, verb_file_path, num_processes=args.num_processes)
#
# if __name__ == '__main__':
#     main()


# import os
# import json
# import time
# from filelock import FileLock
# from utils import image_to_base64, get_text_embedding_v3
# from moduls import action_recognition
# from VISOR_HOS.visor_det import vis_hos
# import numpy as np
# from multiprocessing import Pool, Manager
# from functools import partial
# import glob
# import argparse
# import pandas as pd
# from scipy.spatial.distance import cosine
# from datetime import datetime
# import threading
# from concurrent.futures import ThreadPoolExecutor, as_completed
#
#
# def merge_strategies(playbook_data):
#     """使用文本嵌入计算相似度合并策略，限制总策略数小于50，添加调试信息"""
#     max_strategies = 50
#     similarity_threshold = 0.85
#     print(f"开始合并策略，阈值: {similarity_threshold}")
#
#     for key in ['one', 'two', 'three']:
#         strategies = playbook_data.get(key, {})
#         if len(strategies) <= max_strategies:
#             print(f"{key} 策略数 {len(strategies)} <= {max_strategies}, 跳过合并")
#             continue
#
#         embeddings = {}
#         for strategy_id, strategy in strategies.items():
#             content = strategy['content']
#             embeddings[strategy_id] = get_text_embedding_v3(content) if content != "N/A" else np.zeros(1)
#
#         merged = {}
#         used_ids = set()
#         merges_count = 0
#         for sid1, s1 in strategies.items():
#             if sid1 in used_ids:
#                 continue
#             emb1 = embeddings[sid1]
#             if np.all(emb1 == 0):
#                 merged[sid1] = s1
#                 used_ids.add(sid1)
#                 continue
#
#             freq_sum = int(s1['frequency'])
#             for sid2, s2 in strategies.items():
#                 if sid2 in used_ids or sid1 == sid2:
#                     continue
#                 emb2 = embeddings[sid2]
#                 if np.all(emb2 == 0):
#                     continue
#                 similarity = 1 - cosine(emb1, emb2)
#                 if similarity >= similarity_threshold and s1['category'].lower() == s2['category'].lower():
#                     freq_sum += int(s2['frequency'])
#                     used_ids.add(sid2)
#                     merges_count += 1
#                     print(f"合并策略 {sid1} 和 {sid2}, 相似度: {similarity:.4f}")
#             merged[sid1] = {
#                 'category': s1['category'],
#                 'content': s1['content'],
#                 'frequency': str(freq_sum)
#             }
#             used_ids.add(sid1)
#
#         print(f"{key} 合并完成，发生 {merges_count} 次合并")
#         sorted_strategies = sorted(merged.items(), key=lambda x: int(x[1]['frequency']), reverse=True)
#         playbook_data[key] = dict(sorted_strategies[:max_strategies])
#
#     return playbook_data
#
#
# def load_local_playbook(playbook_path):
#     """每个进程加载自己的playbook副本"""
#     if os.path.exists(playbook_path):
#         with open(playbook_path, 'r', encoding='utf-8') as f:
#             try:
#                 return json.load(f)
#             except json.JSONDecodeError:
#                 print(f"警告：{playbook_path} 解析错误，返回空playbook")
#     return {'one': {}, 'two': {}, 'three': {}}
#
#
# def get_top_strategies_local(playbook_data):
#     """从本地playbook数据获取top策略"""
#     top_strategies = {'one': {}, 'two': {}, 'three': {}}
#     for key in ['one', 'two', 'three']:
#         if playbook_data[key]:
#             sorted_strategies = sorted(playbook_data[key].items(),
#                                        key=lambda x: int(x[1]['frequency']),
#                                        reverse=True)
#             top_3 = sorted_strategies[:3]
#             for strategy_id, strategy in top_3:
#                 top_strategies[key][strategy_id] = {
#                     'category': strategy['category'],
#                     'content': strategy['content']
#                 }
#         else:
#             top_strategies[key] = {'None': None}
#     return top_strategies
#
#
# def save_result_locally(output_path, data, predicted_action_dict, reflect_dict):
#     """将结果保存到本地文件"""
#     try:
#         with open(output_path, 'a', encoding='utf-8') as f:
#             actions = predicted_action_dict.get("actions", [])
#             sorted_actions = sorted(actions, key=lambda x: x.get("confidence", 0), reverse=True)
#             verbs = ";".join([action.get("verb", "").strip() for action in sorted_actions]) or "无动词"
#             nouns = ";".join([action.get("noun", "").strip() for action in sorted_actions]) or "无名词"
#             actions_str = ";".join([action.get("action", "").strip() for action in sorted_actions]) or "无动作"
#             confidences = ";".join([str(action.get("confidence", 0.0)) for action in sorted_actions]) or "0.0"
#
#             f.write(
#                 f"{data['narration_id']},{data['participant_id']},{data['video_id']},{data['narration_timestamp']},"
#                 f"{data['start_timestamp']},{data['stop_timestamp']},{data['start_frame']},{data['stop_frame']},"
#                 f"{verbs},{nouns},{actions_str},{confidences}\n"
#             )
#     except Exception as e:
#         print(f"保存结果到 {output_path} 失败: {e}")
#
#
# def process_single_item(data, noun_file_path, verb_file_path, output_txt_path,
#                         local_playbook, process_id, item_index):
#     """处理单个数据项"""
#     start_time = time.time()
#
#     try:
#         base_path = os.path.join('/home/will/Mydata/EPIC-KITCHENS',
#                                  data['participant_id'], 'rgb_frames', data['video_id'])
#         if not os.path.exists(base_path):
#             print(f"进程 {process_id} 跳过 {data['video_id']}：路径不存在")
#             return None
#
#         # 处理帧数据
#         indices = np.linspace(data['start_frame'], data['stop_frame'] - 1, num=8).astype(int)
#         frame_paths = [os.path.join(base_path, f"frame_{idx:010d}.jpg") for idx in indices]
#         frames_urls = [image_to_base64(path) for path in frame_paths if os.path.exists(path)]
#
#         if not frames_urls:
#             print(f"进程 {process_id} 跳过 {data['video_id']}：没有有效的帧文件")
#             return None
#
#         # 视觉检测
#         images_box = vis_hos(frame_paths)
#         frames_urls = frames_urls + images_box
#
#         # 获取top策略（本地操作，无锁）
#         top_strategies = get_top_strategies_local(local_playbook)
#
#         # 动作识别（无锁操作）
#         try:
#             reflect_dict, predicted_action_dict, noun_reason, action_reason = action_recognition(
#                 frames_urls, noun_file_path, verb_file_path, top_strategies
#             )
#         except Exception as e:
#             print(f"进程 {process_id} action_recognition 错误：{e}, narration_id: {data['narration_id']}")
#             return None
#
#         # 生成唯一ID
#         timestamp = int(time.time() * 1000)
#         new_id = f"id-{data['narration_id']}-{process_id}-{item_index}-{timestamp}"
#
#         # 保存结果到本地文件
#         output_txt_path_proc = f"{output_txt_path.rsplit('.', 1)[0]}_proc_{process_id}.txt"
#         save_result_locally(output_txt_path_proc, data, predicted_action_dict, reflect_dict)
#
#         # 返回新策略条目用于批量更新
#         result_entry = {
#             'new_id': new_id,
#             'reflect_dict': reflect_dict,
#             'process_id': process_id,
#             'narration_id': data['narration_id'],
#             'timestamp': datetime.now().isoformat()
#         }
#
#         processing_time = time.time() - start_time
#         print(f"进程 {process_id} 处理 {data['narration_id']} 完成，耗时: {processing_time:.2f}秒")
#
#         return result_entry
#
#     except Exception as e:
#         print(f"进程 {process_id} 处理 {data['narration_id']} 失败：{e}")
#         return None
#
#
# def update_playbook_batch(all_results, playbook_path):
#     """批量更新playbook，减少锁竞争"""
#     if not all_results:
#         print("没有新结果需要更新到playbook")
#         return
#
#     lock_path = f"{playbook_path}.lock"
#     with FileLock(lock_path):
#         # 加载现有playbook
#         if os.path.exists(playbook_path):
#             with open(playbook_path, 'r', encoding='utf-8') as f:
#                 playbook_data = json.load(f)
#         else:
#             playbook_data = {'one': {}, 'two': {}, 'three': {}}
#
#         # 批量应用更新
#         update_count = 0
#         for result in all_results:
#             if result is None:
#                 continue
#
#             new_id = result['new_id']
#             reflect_dict = result['reflect_dict']
#
#             # 更新三个策略类别
#             playbook_data['one'][new_id] = {
#                 "category": reflect_dict.get("category_one", "N/A"),
#                 "content": reflect_dict.get("correct_approach_one", "N/A"),
#                 "frequency": "1"
#             }
#             playbook_data['two'][new_id] = {
#                 "category": reflect_dict.get("category_two", "N/A"),
#                 "content": reflect_dict.get("correct_approach_two", "N/A"),
#                 "frequency": "1"
#             }
#             playbook_data['three'][new_id] = {
#                 "category": reflect_dict.get("category_three", "N/A"),
#                 "content": reflect_dict.get("correct_approach_three", "N/A"),
#                 "frequency": "1"
#             }
#             update_count += 1
#
#         # 合并策略（如果数量过多）
#         playbook_data = merge_strategies(playbook_data)
#
#         # 保存更新后的playbook
#         with open(playbook_path, 'w', encoding='utf-8') as f:
#             json.dump(playbook_data, f, ensure_ascii=False, indent=2)
#
#         print(f"批量更新完成，共更新 {update_count} 个策略，当前策略数 - "
#               f"one: {len(playbook_data['one'])}, "
#               f"two: {len(playbook_data['two'])}, "
#               f"three: {len(playbook_data['three'])}")
#
#
# def process_chunk(args, noun_file_path, verb_file_path, output_txt_path, playbook_path):
#     """处理一个数据块"""
#     chunk_id, chunk_data = args
#     process_id = f"{os.getpid()}_{chunk_id}"
#     local_results = []
#
#     print(f"进程 {process_id} 开始处理数据块，包含 {len(chunk_data)} 个数据项")
#
#     # 加载本地playbook副本
#     local_playbook = load_local_playbook(playbook_path)
#
#     for i, data in enumerate(chunk_data):
#         if i % 10 == 0:  # 每处理10个数据项打印一次进度
#             print(f"进程 {process_id} 处理进度: {i + 1}/{len(chunk_data)}")
#
#         result = process_single_item(data, noun_file_path, verb_file_path,
#                                      output_txt_path, local_playbook, process_id, i)
#         if result:
#             local_results.append(result)
#
#     print(f"进程 {process_id} 完成数据块处理，生成 {len(local_results)} 个结果")
#     return local_results
#
#
# def merge_temp_files(output_txt_path, num_processes):
#     """合并所有临时文件"""
#     temp_pattern = f"{output_txt_path.rsplit('.', 1)[0]}_proc_*.txt"
#     temp_files = glob.glob(temp_pattern)
#
#     if not temp_files:
#         print("未找到临时文件进行合并")
#         return
#
#     try:
#         with open(output_txt_path, 'w', encoding='utf-8') as final_file:
#             # 写入表头
#             final_file.write("narration_id,participant_id,video_id,narration_timestamp,"
#                              "start_timestamp,stop_timestamp,start_frame,stop_frame,"
#                              "verbs,nouns,actions,confidences\n")
#
#             for temp_file in temp_files:
#                 if os.path.exists(temp_file):
#                     with open(temp_file, 'r', encoding='utf-8') as f:
#                         content = f.read()
#                         final_file.write(content)
#                     os.remove(temp_file)
#                     print(f"合并并删除临时文件: {temp_file}")
#     except Exception as e:
#         print(f"合并文件失败: {e}")
#
#
# def save_and_test_optimized(csv_file_path, output_txt_path, noun_file_path, verb_file_path,
#                             num_processes=4, playbook_path='ace_playbook.json'):
#     """优化版本的主函数"""
#     print(f"使用优化多进程模式，进程数: {num_processes}")
#     start_time = time.time()
#
#     try:
#         df = pd.read_csv(csv_file_path, on_bad_lines='skip')
#         data_list = df.to_dict(orient='records')
#         print(f"成功读取 {len(data_list)} 条数据")
#     except Exception as e:
#         print(f"读取CSV文件失败：{e}")
#         return
#
#     # 使用数据分块减少进程间通信
#     chunk_size = max(1, len(data_list) // num_processes)
#     chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]
#     print(f"将数据分成 {len(chunks)} 个块，每块约 {chunk_size} 条数据")
#
#     all_results = []
#
#     # 使用进程池处理数据块
#     with Pool(processes=num_processes) as pool:
#         print("启动进程池...")
#
#         # 准备参数
#         process_func = partial(process_chunk,
#                                noun_file_path=noun_file_path,
#                                verb_file_path=verb_file_path,
#                                output_txt_path=output_txt_path,
#                                playbook_path=playbook_path)
#
#         # 为每个块分配ID
#         chunk_args = [(i, chunk) for i, chunk in enumerate(chunks)]
#
#         # 并行处理所有块
#         chunk_results = pool.map(process_func, chunk_args)
#
#         # 收集所有结果
#         for result in chunk_results:
#             all_results.extend(result)
#
#     # 批量更新playbook
#     update_playbook_batch(all_results, playbook_path)
#
#     # 合并结果文件
#     merge_temp_files(output_txt_path, num_processes)
#
#     total_time = time.time() - start_time
#     print(f"处理完成！总耗时: {total_time:.2f}秒, 共处理 {len(all_results)} 个有效结果")
#
#
# def save_and_test_threaded(csv_file_path, output_txt_path, noun_file_path, verb_file_path,
#                            num_processes=4, playbook_path='ace_playbook.json'):
#     """备选方案：使用线程池+进程隔离（如果上面方案仍有问题）"""
#     print(f"使用线程池+进程隔离模式，工作线程数: {num_processes}")
#
#     try:
#         df = pd.read_csv(csv_file_path, on_bad_lines='skip')
#         data_list = df.to_dict(orient='records')
#         print(f"成功读取 {len(data_list)} 条数据")
#     except Exception as e:
#         print(f"读取CSV文件失败：{e}")
#         return
#
#     all_results = []
#
#     def worker_wrapper(data):
#         """包装工作函数，用于线程池"""
#         process_id = threading.get_ident()
#         local_playbook = load_local_playbook(playbook_path)
#
#         result = process_single_item(data, noun_file_path, verb_file_path,
#                                      output_txt_path, local_playbook, f"thread_{process_id}", 0)
#         return result
#
#     # 使用线程池
#     with ThreadPoolExecutor(max_workers=num_processes) as executor:
#         print("启动线程池...")
#
#         # 提交所有任务
#         future_to_data = {executor.submit(worker_wrapper, data): data for data in data_list}
#
#         # 收集结果
#         completed = 0
#         for future in as_completed(future_to_data):
#             data = future_to_data[future]
#             try:
#                 result = future.result()
#                 if result:
#                     all_results.append(result)
#                 completed += 1
#                 if completed % 10 == 0:
#                     print(f"处理进度: {completed}/{len(data_list)}")
#             except Exception as e:
#                 print(f"处理 {data.get('narration_id', 'unknown')} 失败: {e}")
#                 completed += 1
#
#     # 批量更新playbook
#     update_playbook_batch(all_results, playbook_path)
#
#     # 合并结果文件（线程版本不需要合并临时文件）
#     print("线程池模式处理完成")
#
#
# def main():
#     parser = argparse.ArgumentParser(description="运行动作识别的测试模式（优化多进程版本）")
#     parser.add_argument('--num_processes', type=int, default=min(4, os.cpu_count() - 1),
#                         help="指定使用的进程数（默认：min(4, CPU核心数-1）")
#     parser.add_argument('--mode', type=str, choices=['process', 'thread'], default='process',
#                         help="选择并发模式：process（进程池）或 thread（线程池）")
#     args = parser.parse_args()
#
#     csv_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_validation.csv'
#     noun_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_noun_classes.csv'
#     verb_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_verb_classes.csv'
#     output_txt_path = 'test_comparison_results_frames8_optimized.txt'
#     playbook_path = 'ace_playbook_optimized.json'
#
#     print(f"使用优化模式运行，模式: {args.mode}, 并发数: {args.num_processes}")
#
#     if args.mode == 'process':
#         save_and_test_optimized(csv_file_path, output_txt_path, noun_file_path, verb_file_path,
#                                 num_processes=args.num_processes, playbook_path=playbook_path)
#     else:
#         save_and_test_threaded(csv_file_path, output_txt_path, noun_file_path, verb_file_path,
#                                num_processes=args.num_processes, playbook_path=playbook_path)
#
#
# if __name__ == '__main__':
#     main()


# import os
# import json
# import time
# from filelock import FileLock
# import numpy as np
# from multiprocessing import Pool
# from functools import partial
# import glob
# import argparse
# import pandas as pd
# from datetime import datetime
#
# # 假设这些导入存在
# from utils import image_to_base64, get_text_embedding_v3
# from moduls import action_recognition
# from VISOR_HOS.visor_det import vis_hos
# from scipy.spatial.distance import cosine
#
#
# def merge_strategies_to_limit(playbook_data, max_strategies=50):
#     """合并策略确保每个类别不超过max_strategies条"""
#     print(f"开始合并策略，确保每个类别不超过 {max_strategies} 条")
#
#     for key in ['one', 'two', 'three']:
#         strategies = playbook_data.get(key, {})
#         current_count = len(strategies)
#
#         if current_count <= max_strategies:
#             print(f"{key} 策略数 {current_count} <= {max_strategies}, 跳过合并")
#             continue
#
#         print(f"{key} 策略数 {current_count} > {max_strategies}, 开始合并...")
#
#         # 计算文本嵌入
#         embeddings = {}
#         for strategy_id, strategy in strategies.items():
#             content = strategy['content']
#             embeddings[strategy_id] = get_text_embedding_v3(content) if content != "N/A" else np.zeros(1)
#
#         # 按频率排序
#         sorted_strategies = sorted(strategies.items(), key=lambda x: int(x[1]['frequency']), reverse=True)
#
#         # 保留高频策略，合并相似的低频策略
#         merged = {}
#         used_ids = set()
#         merges_count = 0
#
#         # 先保留前N/2个高频策略
#         high_freq_count = max_strategies // 2
#         for i, (sid, strategy) in enumerate(sorted_strategies[:high_freq_count]):
#             merged[sid] = strategy
#             used_ids.add(sid)
#
#         # 合并剩余的策略
#         similarity_threshold = 0.85
#         for sid1, s1 in sorted_strategies[high_freq_count:]:
#             if sid1 in used_ids:
#                 continue
#
#             emb1 = embeddings[sid1]
#             if np.all(emb1 == 0):
#                 # 如果无法获取嵌入，直接保留
#                 if len(merged) < max_strategies:
#                     merged[sid1] = s1
#                     used_ids.add(sid1)
#                 continue
#
#             # 寻找相似策略进行合并
#             merged_with_existing = False
#             for sid2 in list(merged.keys()):
#                 if sid2 in used_ids:
#                     emb2 = embeddings[sid2]
#                     if np.all(emb2 == 0):
#                         continue
#
#                     similarity = 1 - cosine(emb1, emb2)
#                     if similarity >= similarity_threshold and s1['category'].lower() == merged[sid2][
#                         'category'].lower():
#                         # 合并策略
#                         merged[sid2]['frequency'] = str(int(merged[sid2]['frequency']) + int(s1['frequency']))
#                         used_ids.add(sid1)
#                         merges_count += 1
#                         merged_with_existing = True
#                         break
#
#             # 如果没有合并到现有策略，且还有空间，则添加
#             if not merged_with_existing and len(merged) < max_strategies:
#                 merged[sid1] = s1
#                 used_ids.add(sid1)
#
#         # 如果合并后仍然超过限制，按频率排序并截断
#         if len(merged) > max_strategies:
#             sorted_merged = sorted(merged.items(), key=lambda x: int(x[1]['frequency']), reverse=True)
#             merged = dict(sorted_merged[:max_strategies])
#
#         playbook_data[key] = merged
#         print(f"{key} 合并完成，从 {current_count} 条合并到 {len(merged)} 条，发生 {merges_count} 次合并")
#
#     return playbook_data
#
#
# def ensure_playbook_limit(playbook_data, max_strategies=50):
#     """确保playbook每个类别不超过max_strategies条策略"""
#     for key in ['one', 'two', 'three']:
#         strategies = playbook_data.get(key, {})
#         if len(strategies) > max_strategies:
#             # 按频率排序并保留前max_strategies条
#             sorted_strategies = sorted(strategies.items(), key=lambda x: int(x[1]['frequency']), reverse=True)
#             playbook_data[key] = dict(sorted_strategies[:max_strategies])
#             print(f"{key} 类别策略数超过 {max_strategies}，已截断为 {len(playbook_data[key])} 条")
#
#     return playbook_data
#
#
# def load_local_playbook(playbook_path, max_strategies=50):
#     """加载playbook副本，并确保不超过策略限制"""
#     if os.path.exists(playbook_path):
#         with open(playbook_path, 'r', encoding='utf-8') as f:
#             try:
#                 playbook_data = json.load(f)
#                 # 确保加载的playbook不超过策略限制
#                 return ensure_playbook_limit(playbook_data, max_strategies)
#             except json.JSONDecodeError:
#                 print(f"警告：{playbook_path} 解析错误，返回空playbook")
#     return {'one': {}, 'two': {}, 'three': {}}
#
#
# def get_top_strategies_local(playbook_data, top_n=3):
#     """从playbook数据获取top策略"""
#     top_strategies = {'one': {}, 'two': {}, 'three': {}}
#     for key in ['one', 'two', 'three']:
#         if playbook_data[key]:
#             sorted_strategies = sorted(playbook_data[key].items(),
#                                        key=lambda x: int(x[1]['frequency']),
#                                        reverse=True)
#             top_n_strategies = sorted_strategies[:top_n]
#             for strategy_id, strategy in top_n_strategies:
#                 top_strategies[key][strategy_id] = {
#                     'category': strategy['category'],
#                     'content': strategy['content']
#                 }
#         else:
#             top_strategies[key] = {'None': None}
#     return top_strategies
#
#
# def save_result_locally(output_path, data, predicted_action_dict):
#     """将结果保存到本地文件"""
#     try:
#         with open(output_path, 'a', encoding='utf-8') as f:
#             actions = predicted_action_dict.get("actions", [])
#             sorted_actions = sorted(actions, key=lambda x: x.get("confidence", 0), reverse=True)
#             verbs = ";".join([action.get("verb", "").strip() for action in sorted_actions]) or "无动词"
#             nouns = ";".join([action.get("noun", "").strip() for action in sorted_actions]) or "无名词"
#             actions_str = ";".join([action.get("action", "").strip() for action in sorted_actions]) or "无动作"
#             confidences = ";".join([str(action.get("confidence", 0.0)) for action in sorted_actions]) or "0.0"
#
#             f.write(
#                 f"{data['narration_id']},{data['participant_id']},{data['video_id']},{data['narration_timestamp']},"
#                 f"{data['start_timestamp']},{data['stop_timestamp']},{data['start_frame']},{data['stop_frame']},"
#                 f"{verbs},{nouns},{actions_str},{confidences}\n"
#             )
#     except Exception as e:
#         print(f"保存结果到 {output_path} 失败: {e}")
#
#
# def process_single_item(data, noun_file_path, verb_file_path, output_txt_path,
#                         playbook_manager, process_id, item_index):
#     """处理单个数据项，使用playbook管理器"""
#     start_time = time.time()
#
#     try:
#         base_path = os.path.join('/home/will/Mydata/EPIC-KITCHENS',
#                                  data['participant_id'], 'rgb_frames', data['video_id'])
#         if not os.path.exists(base_path):
#             print(f"进程 {process_id} 跳过 {data['video_id']}：路径不存在")
#             return None
#
#         # 处理帧数据
#         indices = np.linspace(data['start_frame'], data['stop_frame'] - 1, num=8).astype(int)
#         frame_paths = [os.path.join(base_path, f"frame_{idx:010d}.jpg") for idx in indices]
#         frames_urls = [image_to_base64(path) for path in frame_paths if os.path.exists(path)]
#
#         if not frames_urls:
#             print(f"进程 {process_id} 跳过 {data['video_id']}：没有有效的帧文件")
#             return None
#
#         # 视觉检测
#         images_box = vis_hos(frame_paths)
#         frames_urls = frames_urls + images_box
#
#         # 从playbook管理器获取最新的top策略
#         current_playbook = playbook_manager.get_playbook()
#         top_strategies = get_top_strategies_local(current_playbook)
#
#         # 动作识别
#         try:
#             reflect_dict, predicted_action_dict, noun_reason, action_reason = action_recognition(
#                 frames_urls, noun_file_path, verb_file_path, top_strategies
#             )
#         except Exception as e:
#             print(f"进程 {process_id} action_recognition 错误：{e}, narration_id: {data['narration_id']}")
#             return None
#
#         # 生成唯一ID
#         timestamp = int(time.time() * 1000)
#         new_id = f"id-{data['narration_id']}-{process_id}-{item_index}-{timestamp}"
#
#         # 保存结果到本地文件
#         output_txt_path_proc = f"{output_txt_path.rsplit('.', 1)[0]}_proc_{process_id}.txt"
#         save_result_locally(output_txt_path_proc, data, predicted_action_dict)
#
#         # 将新策略添加到playbook管理器
#         playbook_manager.add_strategy(new_id, reflect_dict)
#
#         # 返回新策略条目用于批量更新
#         result_entry = {
#             'new_id': new_id,
#             'reflect_dict': reflect_dict,
#             'process_id': process_id,
#             'narration_id': data['narration_id'],
#             'timestamp': datetime.now().isoformat()
#         }
#
#         processing_time = time.time() - start_time
#         print(f"进程 {process_id} 处理 {data['narration_id']} 完成，耗时: {processing_time:.2f}秒")
#
#         return result_entry
#
#     except Exception as e:
#         print(f"进程 {process_id} 处理 {data['narration_id']} 失败：{e}")
#         return None
#
#
# class SimplePlaybookManager:
#     """简化的playbook管理器，定期从文件重新加载，确保策略不超过限制"""
#
#     def __init__(self, playbook_path, max_strategies=50, reload_interval=30):
#         self.playbook_path = playbook_path
#         self.max_strategies = max_strategies
#         self.reload_interval = reload_interval
#         self.last_reload = 0
#         self.local_playbook = self._load_from_file()
#
#     def _load_from_file(self):
#         """从文件加载playbook，并确保不超过策略限制"""
#         if os.path.exists(self.playbook_path):
#             with open(self.playbook_path, 'r', encoding='utf-8') as f:
#                 try:
#                     playbook_data = json.load(f)
#                     # 确保加载的playbook不超过策略限制
#                     return ensure_playbook_limit(playbook_data, self.max_strategies)
#                 except json.JSONDecodeError:
#                     print(f"警告：{self.playbook_path} 解析错误，返回空playbook")
#         return {'one': {}, 'two': {}, 'three': {}}
#
#     def get_playbook(self):
#         """获取playbook，必要时重新加载"""
#         current_time = time.time()
#         if current_time - self.last_reload > self.reload_interval:
#             self.local_playbook = self._load_from_file()
#             self.last_reload = current_time
#         return self.local_playbook
#
#     def add_strategy(self, new_id, reflect_dict):
#         """添加新策略到本地playbook，确保不超过策略限制"""
#         # 添加新策略
#         self.local_playbook['one'][new_id] = {
#             "category": reflect_dict.get("category_one", "N/A"),
#             "content": reflect_dict.get("correct_approach_one", "N/A"),
#             "frequency": "1"
#         }
#         self.local_playbook['two'][new_id] = {
#             "category": reflect_dict.get("category_two", "N/A"),
#             "content": reflect_dict.get("correct_approach_two", "N/A"),
#             "frequency": "1"
#         }
#         self.local_playbook['three'][new_id] = {
#             "category": reflect_dict.get("category_three", "N/A"),
#             "content": reflect_dict.get("correct_approach_three", "N/A"),
#             "frequency": "1"
#         }
#
#         # 检查是否超过限制，如果超过则合并策略
#         for key in ['one', 'two', 'three']:
#             if len(self.local_playbook[key]) > self.max_strategies:
#                 print(f"本地playbook {key} 类别策略数超过 {self.max_strategies}，进行合并...")
#                 self.local_playbook = merge_strategies_to_limit(self.local_playbook, self.max_strategies)
#                 break
#
#
# def process_chunk_with_manager(args, noun_file_path, verb_file_path, output_txt_path, playbook_path, max_strategies=50):
#     """使用playbook管理器处理数据块"""
#     chunk_id, chunk_data = args
#     process_id = f"{os.getpid()}_{chunk_id}"
#     local_results = []
#
#     print(f"进程 {process_id} 开始处理数据块，包含 {len(chunk_data)} 个数据项")
#
#     # 创建playbook管理器
#     playbook_manager = SimplePlaybookManager(playbook_path, max_strategies=max_strategies, reload_interval=30)
#
#     for i, data in enumerate(chunk_data):
#         if i % 10 == 0:  # 每处理10个数据项打印一次进度
#             print(f"进程 {process_id} 处理进度: {i + 1}/{len(chunk_data)}")
#
#         result = process_single_item(data, noun_file_path, verb_file_path,
#                                      output_txt_path, playbook_manager, process_id, i)
#         if result:
#             local_results.append(result)
#
#     print(f"进程 {process_id} 完成数据块处理，生成 {len(local_results)} 个结果")
#     return local_results
#
#
# def update_playbook_batch(all_results, playbook_path, max_strategies=50):
#     """批量更新playbook，确保不超过策略限制"""
#     if not all_results:
#         print("没有新结果需要更新到playbook")
#         return
#
#     lock_path = f"{playbook_path}.lock"
#     with FileLock(lock_path):
#         # 加载现有playbook
#         if os.path.exists(playbook_path):
#             with open(playbook_path, 'r', encoding='utf-8') as f:
#                 playbook_data = json.load(f)
#         else:
#             playbook_data = {'one': {}, 'two': {}, 'three': {}}
#
#         # 批量应用更新
#         update_count = 0
#         for result in all_results:
#             if result is None:
#                 continue
#
#             new_id = result['new_id']
#             reflect_dict = result['reflect_dict']
#
#             # 更新三个策略类别
#             playbook_data['one'][new_id] = {
#                 "category": reflect_dict.get("category_one", "N/A"),
#                 "content": reflect_dict.get("correct_approach_one", "N/A"),
#                 "frequency": "1"
#             }
#             playbook_data['two'][new_id] = {
#                 "category": reflect_dict.get("category_two", "N/A"),
#                 "content": reflect_dict.get("correct_approach_two", "N/A"),
#                 "frequency": "1"
#             }
#             playbook_data['three'][new_id] = {
#                 "category": reflect_dict.get("category_three", "N/A"),
#                 "content": reflect_dict.get("correct_approach_three", "N/A"),
#                 "frequency": "1"
#             }
#             update_count += 1
#
#         # 检查并确保每个类别不超过max_strategies条策略
#         for key in ['one', 'two', 'three']:
#             if len(playbook_data[key]) > max_strategies:
#                 print(f"playbook {key} 类别策略数 {len(playbook_data[key])} > {max_strategies}，进行合并...")
#                 playbook_data = merge_strategies_to_limit(playbook_data, max_strategies)
#                 break
#
#         # 保存更新后的playbook
#         with open(playbook_path, 'w', encoding='utf-8') as f:
#             json.dump(playbook_data, f, ensure_ascii=False, indent=2)
#
#         print(f"批量更新完成，共更新 {update_count} 个策略，当前策略数 - "
#               f"one: {len(playbook_data['one'])}, "
#               f"two: {len(playbook_data['two'])}, "
#               f"three: {len(playbook_data['three'])}")
#
#
# def merge_temp_files(output_txt_path, num_processes):
#     """合并所有临时文件"""
#     temp_pattern = f"{output_txt_path.rsplit('.', 1)[0]}_proc_*.txt"
#     temp_files = glob.glob(temp_pattern)
#
#     if not temp_files:
#         print("未找到临时文件进行合并")
#         return
#
#     try:
#         with open(output_txt_path, 'w', encoding='utf-8') as final_file:
#             # 写入表头
#             final_file.write("narration_id,participant_id,video_id,narration_timestamp,"
#                              "start_timestamp,stop_timestamp,start_frame,stop_frame,"
#                              "verbs,nouns,actions,confidences\n")
#
#             for temp_file in temp_files:
#                 if os.path.exists(temp_file):
#                     with open(temp_file, 'r', encoding='utf-8') as f:
#                         content = f.read()
#                         final_file.write(content)
#                     os.remove(temp_file)
#                     print(f"合并并删除临时文件: {temp_file}")
#     except Exception as e:
#         print(f"合并文件失败: {e}")
#
#
# def save_and_test_with_limited_playbook(csv_file_path, output_txt_path, noun_file_path, verb_file_path,
#                                         num_processes=4, playbook_path='ace_playbook.json', max_strategies=50):
#     """使用有限策略playbook的主函数"""
#     print(f"使用有限策略playbook多进程模式，进程数: {num_processes}，每个类别最多 {max_strategies} 条策略")
#     start_time = time.time()
#
#     try:
#         df = pd.read_csv(csv_file_path, on_bad_lines='skip')
#         data_list = df.to_dict(orient='records')
#         print(f"成功读取 {len(data_list)} 条数据")
#     except Exception as e:
#         print(f"读取CSV文件失败：{e}")
#         return
#
#     # 创建初始playbook文件（如果不存在）
#     if not os.path.exists(playbook_path):
#         initial_data = {'one': {}, 'two': {}, 'three': {}}
#         with open(playbook_path, 'w', encoding='utf-8') as f:
#             json.dump(initial_data, f, ensure_ascii=False, indent=2)
#         print(f"创建初始playbook文件: {playbook_path}")
#     else:
#         # 确保现有playbook不超过策略限制
#         with open(playbook_path, 'r', encoding='utf-8') as f:
#             existing_data = json.load(f)
#
#         needs_cleanup = False
#         for key in ['one', 'two', 'three']:
#             if len(existing_data[key]) > max_strategies:
#                 needs_cleanup = True
#                 break
#
#         if needs_cleanup:
#             print("现有playbook超过策略限制，进行清理...")
#             existing_data = merge_strategies_to_limit(existing_data, max_strategies)
#             with open(playbook_path, 'w', encoding='utf-8') as f:
#                 json.dump(existing_data, f, ensure_ascii=False, indent=2)
#             print("playbook清理完成")
#
#     # 使用数据分块
#     chunk_size = max(1, len(data_list) // num_processes)
#     chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]
#     print(f"将数据分成 {len(chunks)} 个块，每块约 {chunk_size} 条数据")
#
#     all_results = []
#
#     # 使用进程池处理数据块
#     with Pool(processes=num_processes) as pool:
#         print("启动进程池...")
#
#         # 准备参数
#         process_func = partial(process_chunk_with_manager,
#                                noun_file_path=noun_file_path,
#                                verb_file_path=verb_file_path,
#                                output_txt_path=output_txt_path,
#                                playbook_path=playbook_path,
#                                max_strategies=max_strategies)
#
#         # 为每个块分配ID
#         chunk_args = [(i, chunk) for i, chunk in enumerate(chunks)]
#
#         # 并行处理所有块
#         chunk_results = pool.map(process_func, chunk_args)
#
#         # 收集所有结果
#         for result in chunk_results:
#             all_results.extend(result)
#
#     # 批量更新playbook（最终一致性）
#     update_playbook_batch(all_results, playbook_path, max_strategies)
#
#     # 合并结果文件
#     merge_temp_files(output_txt_path, num_processes)
#
#     total_time = time.time() - start_time
#     print(f"处理完成！总耗时: {total_time:.2f}秒, 共处理 {len(all_results)} 个有效结果")
#
#     # 最终验证playbook策略数量
#     with open(playbook_path, 'r', encoding='utf-8') as f:
#         final_playbook = json.load(f)
#         for key in ['one', 'two', 'three']:
#             count = len(final_playbook[key])
#             print(f"最终playbook {key} 类别策略数: {count}")
#             if count > max_strategies:
#                 print(f"警告: {key} 类别策略数 {count} 超过限制 {max_strategies}")
#
#
# def main():
#     parser = argparse.ArgumentParser(description="运行动作识别（有限策略playbook版本）")
#     parser.add_argument('--num_processes', type=int, default=min(10, os.cpu_count() - 1))
#     parser.add_argument('--max_strategies', type=int, default=50, help="每个类别最多保留的策略数")
#     args = parser.parse_args()
#
#     csv_file_path = '/home/will/Mycode/action_agent/unrecognized_val.txt'
#     noun_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_noun_classes.csv'
#     verb_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_verb_classes.csv'
#     output_txt_path = 'test_comparison_results_frames8_limited_2.txt'
#     playbook_path = 'ace_playbook_limited.json'
#
#     save_and_test_with_limited_playbook(csv_file_path, output_txt_path, noun_file_path, verb_file_path,
#                                         num_processes=args.num_processes,
#                                         playbook_path=playbook_path,
#                                         max_strategies=args.max_strategies)
#
#
# if __name__ == '__main__':
#     main()


import os
import json
import time
from filelock import FileLock
import numpy as np
from multiprocessing import Pool
from functools import partial
import glob
import argparse
import pandas as pd
from datetime import datetime

# 假设这些导入存在
from utils import image_to_base64, get_text_embedding_v3
from moduls import action_recognition
from VISOR_HOS.visor_det import vis_hos
from scipy.spatial.distance import cosine


def merge_strategies_to_limit(playbook_data, max_strategies=50):
    """合并策略确保每个类别不超过max_strategies条"""
    print(f"开始合并策略，确保每个类别不超过 {max_strategies} 条")

    for key in ['one', 'two', 'three']:
        strategies = playbook_data.get(key, {})
        current_count = len(strategies)

        if current_count <= max_strategies:
            print(f"{key} 策略数 {current_count} <= {max_strategies}, 跳过合并")
            continue

        print(f"{key} 策略数 {current_count} > {max_strategies}, 开始合并...")

        # 计算文本嵌入
        embeddings = {}
        for strategy_id, strategy in strategies.items():
            content = strategy['content']
            embeddings[strategy_id] = get_text_embedding_v3(content) if content != "N/A" else np.zeros(1)

        # 按频率排序
        sorted_strategies = sorted(strategies.items(), key=lambda x: int(x[1]['frequency']), reverse=True)

        # 保留高频策略，合并相似的低频策略
        merged = {}
        used_ids = set()
        merges_count = 0

        # 先保留前N/2个高频策略
        high_freq_count = max_strategies // 2
        for i, (sid, strategy) in enumerate(sorted_strategies[:high_freq_count]):
            merged[sid] = strategy
            used_ids.add(sid)

        # 合并剩余的策略
        similarity_threshold = 0.85
        for sid1, s1 in sorted_strategies[high_freq_count:]:
            if sid1 in used_ids:
                continue

            emb1 = embeddings[sid1]
            if np.all(emb1 == 0):
                # 如果无法获取嵌入，直接保留
                if len(merged) < max_strategies:
                    merged[sid1] = s1
                    used_ids.add(sid1)
                continue

            # 寻找相似策略进行合并
            merged_with_existing = False
            for sid2 in list(merged.keys()):
                if sid2 in used_ids:
                    emb2 = embeddings[sid2]
                    if np.all(emb2 == 0):
                        continue

                    similarity = 1 - cosine(emb1, emb2)
                    if similarity >= similarity_threshold and s1['category'].lower() == merged[sid2][
                        'category'].lower():
                        # 合并策略
                        merged[sid2]['frequency'] = str(int(merged[sid2]['frequency']) + int(s1['frequency']))
                        used_ids.add(sid1)
                        merges_count += 1
                        merged_with_existing = True
                        break

            # 如果没有合并到现有策略，且还有空间，则添加
            if not merged_with_existing and len(merged) < max_strategies:
                merged[sid1] = s1
                used_ids.add(sid1)

        # 如果合并后仍然超过限制，按频率排序并截断
        if len(merged) > max_strategies:
            sorted_merged = sorted(merged.items(), key=lambda x: int(x[1]['frequency']), reverse=True)
            merged = dict(sorted_merged[:max_strategies])

        playbook_data[key] = merged
        print(f"{key} 合并完成，从 {current_count} 条合并到 {len(merged)} 条，发生 {merges_count} 次合并")

    return playbook_data


def ensure_playbook_limit(playbook_data, max_strategies=50):
    """确保playbook每个类别不超过max_strategies条策略"""
    for key in ['one', 'two', 'three']:
        strategies = playbook_data.get(key, {})
        if len(strategies) > max_strategies:
            # 按频率排序并保留前max_strategies条
            sorted_strategies = sorted(strategies.items(), key=lambda x: int(x[1]['frequency']), reverse=True)
            playbook_data[key] = dict(sorted_strategies[:max_strategies])
            print(f"{key} 类别策略数超过 {max_strategies}，已截断为 {len(playbook_data[key])} 条")

    return playbook_data


def load_local_playbook(playbook_path):
    """加载playbook副本，不进行强制截断"""
    if os.path.exists(playbook_path):
        with open(playbook_path, 'r', encoding='utf-8') as f:
            try:
                playbook_data = json.load(f)
                return playbook_data
            except json.JSONDecodeError:
                print(f"警告：{playbook_path} 解析错误，返回空playbook")
    return {'one': {}, 'two': {}, 'three': {}}


def get_real_time_playbook(playbook_path):
    """实时获取最新playbook，使用文件锁确保一致性，不进行截断"""
    lock_path = f"{playbook_path}.lock"
    with FileLock(lock_path):
        return load_local_playbook(playbook_path)


def check_and_merge_playbook(playbook_path, max_strategies=50, merge_threshold=200):
    """检查playbook是否需要合并，如果策略数大于阈值则合并"""
    lock_path = f"{playbook_path}.lock"
    with FileLock(lock_path):
        # 加载最新playbook
        playbook_data = load_local_playbook(playbook_path)

        # 检查是否需要合并（严格大于阈值）
        need_merge = False
        for key in ['one', 'two', 'three']:
            if len(playbook_data[key]) > merge_threshold:
                need_merge = True
                print(f"检测到 {key} 类别策略数 {len(playbook_data[key])} > {merge_threshold}，触发合并")
                break

        # 如果需要合并，则执行合并
        if need_merge:
            print(f"开始合并playbook，阈值: {merge_threshold}，目标: {max_strategies}")
            playbook_data = merge_strategies_to_limit(playbook_data, max_strategies)

            # 保存合并后的playbook
            with open(playbook_path, 'w', encoding='utf-8') as f:
                json.dump(playbook_data, f, ensure_ascii=False, indent=2)

            print(f"playbook合并完成，当前策略数 - one: {len(playbook_data['one'])}, "
                  f"two: {len(playbook_data['two'])}, three: {len(playbook_data['three'])}")

        return need_merge


def update_strategy_with_threshold(new_id, reflect_dict, playbook_path, max_strategies=50, merge_threshold=200):
    """更新策略到playbook，并在策略数大于阈值时合并"""
    lock_path = f"{playbook_path}.lock"
    with FileLock(lock_path):
        # 加载最新playbook
        playbook_data = load_local_playbook(playbook_path)

        # 先检查当前是否需要合并（在添加新策略之前）
        need_merge = False
        for key in ['one', 'two', 'three']:
            if len(playbook_data[key]) > merge_threshold:
                need_merge = True
                break

        # 如果需要合并，则先执行合并
        if need_merge:
            print(f"检测到策略数超过阈值 {merge_threshold}，开始合并...")
            playbook_data = merge_strategies_to_limit(playbook_data, max_strategies)

        # 添加新策略
        playbook_data['one'][new_id] = {
            "category": reflect_dict.get("category_one", "N/A"),
            "content": reflect_dict.get("correct_approach_one", "N/A"),
            "frequency": "1"
        }
        playbook_data['two'][new_id] = {
            "category": reflect_dict.get("category_two", "N/A"),
            "content": reflect_dict.get("correct_approach_two", "N/A"),
            "frequency": "1"
        }
        playbook_data['three'][new_id] = {
            "category": reflect_dict.get("category_three", "N/A"),
            "content": reflect_dict.get("correct_approach_three", "N/A"),
            "frequency": "1"
        }

        # 保存更新后的playbook
        with open(playbook_path, 'w', encoding='utf-8') as f:
            json.dump(playbook_data, f, ensure_ascii=False, indent=2)

        print(f"策略更新完成，当前策略数 - one: {len(playbook_data['one'])}, "
              f"two: {len(playbook_data['two'])}, three: {len(playbook_data['three'])}")

        return need_merge


def get_top_strategies_local(playbook_data, top_n=3):
    """从playbook数据获取top策略"""
    top_strategies = {'one': {}, 'two': {}, 'three': {}}
    for key in ['one', 'two', 'three']:
        if playbook_data[key]:
            sorted_strategies = sorted(playbook_data[key].items(),
                                       key=lambda x: int(x[1]['frequency']),
                                       reverse=True)
            top_n_strategies = sorted_strategies[:top_n]
            for strategy_id, strategy in top_n_strategies:
                top_strategies[key][strategy_id] = {
                    'category': strategy['category'],
                    'content': strategy['content']
                }
        else:
            top_strategies[key] = {'None': None}
    return top_strategies


def save_result_locally(output_path, data, predicted_action_dict, selected_noun_keys):
    """将结果保存到本地文件，包括selected_noun_keys在最后一列"""
    try:
        with open(output_path, 'a', encoding='utf-8') as f:
            actions = predicted_action_dict.get("actions", [])
            sorted_actions = sorted(actions, key=lambda x: x.get("confidence", 0), reverse=True)
            verbs = ";".join([action.get("verb", "").strip() for action in sorted_actions]) or "无动词"
            nouns = ";".join([action.get("noun", "").strip() for action in sorted_actions]) or "无名词"
            actions_str = ";".join([action.get("action", "").strip() for action in sorted_actions]) or "无动作"
            confidences = ";".join([str(action.get("confidence", 0.0)) for action in sorted_actions]) or "0.0"

            # 处理selected_noun_keys，确保是字符串格式
            if selected_noun_keys is None:
                selected_noun_keys_str = "无选择名词"
            elif isinstance(selected_noun_keys, list):
                selected_noun_keys_str = ";".join([str(key) for key in selected_noun_keys]) or "无选择名词"
            else:
                selected_noun_keys_str = str(selected_noun_keys) or "无选择名词"

            f.write(
                f"{data['narration_id']},{data['participant_id']},{data['video_id']},{data['narration_timestamp']},"
                f"{data['start_timestamp']},{data['stop_timestamp']},{data['start_frame']},{data['stop_frame']},"
                f"{verbs},{nouns},{actions_str},{confidences},{selected_noun_keys_str}\n"
            )
    except Exception as e:
        print(f"保存结果到 {output_path} 失败: {e}")


def safe_action_recognition(frames_urls, noun_file_path, verb_file_path, top_strategies):
    """安全地调用 action_recognition，处理 None 值情况"""
    try:
        # 如果 top_strategies 为 None，创建一个空的策略结构
        if top_strategies is None:
            top_strategies = {'one': {'None': None}, 'two': {'None': None}, 'three': {'None': None}}

        reflect_dict, predicted_action_dict, noun_reason, action_reason, selected_noun_keys = action_recognition(
            frames_urls, noun_file_path, verb_file_path, top_strategies
        )
        return reflect_dict, predicted_action_dict, noun_reason, action_reason, selected_noun_keys
    except Exception as e:
        print(f"action_recognition 错误：{e}")
        # 返回默认值
        default_reflect = {
            "category_one": "N/A",
            "correct_approach_one": "N/A",
            "category_two": "N/A",
            "correct_approach_two": "N/A",
            "category_three": "N/A",
            "correct_approach_three": "N/A"
        }
        default_action = {"actions": []}
        return default_reflect, default_action, "N/A", "N/A", "N/A"


def process_single_item(data, noun_file_path, verb_file_path, output_txt_path,
                        playbook_path, max_strategies, merge_threshold, process_id, item_index, use_playbook=True):
    """处理单个数据项，使用阈值控制的playbook"""
    start_time = time.time()

    try:
        base_path = os.path.join('/home/will/Mydata/EPIC-KITCHENS',
                                 data['participant_id'], 'rgb_frames', data['video_id'])
        if not os.path.exists(base_path):
            print(f"进程 {process_id} 跳过 {data['video_id']}：路径不存在")
            return None

        # 处理帧数据
        indices = np.linspace(data['start_frame'], data['stop_frame'] - 1, num=8).astype(int)
        frame_paths = [os.path.join(base_path, f"frame_{idx:010d}.jpg") for idx in indices]
        frames_urls = [image_to_base64(path) for path in frame_paths if os.path.exists(path)]

        if not frames_urls:
            print(f"进程 {process_id} 跳过 {data['video_id']}：没有有效的帧文件")
            return None

        # 视觉检测
        images_box = vis_hos(frame_paths)
        frames_urls = frames_urls + images_box

        # 根据 use_playbook 标志决定是否使用 playbook
        if use_playbook:
            # 获取最新的playbook
            current_playbook = get_real_time_playbook(playbook_path)
            top_strategies = get_top_strategies_local(current_playbook)

            print(f"进程 {process_id} 获取到playbook，策略数 - "
                  f"one: {len(current_playbook['one'])}, "
                  f"two: {len(current_playbook['two'])}, "
                  f"three: {len(current_playbook['three'])}")
        else:
            top_strategies = None
            print(f"进程 {process_id} 不使用playbook，策略为None")

        # 动作识别 - 使用安全的包装函数
        try:
            reflect_dict, predicted_action_dict, noun_reason, action_reason, selected_noun_keys = safe_action_recognition(
                frames_urls, noun_file_path, verb_file_path, top_strategies
            )
        except Exception as e:
            print(f"进程 {process_id} action_recognition 错误：{e}, narration_id: {data['narration_id']}")
            return None

        # 生成唯一ID
        timestamp = int(time.time() * 1000)
        new_id = f"id-{data['narration_id']}-{process_id}-{item_index}-{timestamp}"

        # 保存结果到本地文件，包括selected_noun_keys
        output_txt_path_proc = f"{output_txt_path.rsplit('.', 1)[0]}_proc_{process_id}.txt"
        save_result_locally(output_txt_path_proc, data, predicted_action_dict, selected_noun_keys)

        # 只有在使用 playbook 时才更新策略
        if use_playbook:
            # 更新策略到playbook，使用阈值控制
            update_strategy_with_threshold(new_id, reflect_dict, playbook_path, max_strategies, merge_threshold)

        # 返回新策略条目用于统计
        result_entry = {
            'new_id': new_id,
            'reflect_dict': reflect_dict,
            'process_id': process_id,
            'narration_id': data['narration_id'],
            'timestamp': datetime.now().isoformat(),
            'use_playbook': use_playbook,
            'selected_noun_keys': selected_noun_keys
        }

        processing_time = time.time() - start_time
        print(
            f"进程 {process_id} 处理 {data['narration_id']} 完成，耗时: {processing_time:.2f}秒，使用playbook: {use_playbook}")

        return result_entry

    except Exception as e:
        print(f"进程 {process_id} 处理 {data['narration_id']} 失败：{e}")
        return None


def create_sample_strategies(playbook_path, num_samples=5):
    """创建示例策略，确保playbook不为空"""
    sample_data = {
        'one': {},
        'two': {},
        'three': {}
    }

    # 创建一些示例策略
    for i in range(num_samples):
        sample_id = f"sample_{i + 1}"
        sample_data['one'][sample_id] = {
            "category": "基本动作",
            "content": f"示例策略 {i + 1} - 基础操作",
            "frequency": "1"
        }
        sample_data['two'][sample_id] = {
            "category": "中级技巧",
            "content": f"示例策略 {i + 1} - 进阶技巧",
            "frequency": "1"
        }
        sample_data['three'][sample_id] = {
            "category": "高级策略",
            "content": f"示例策略 {i + 1} - 高级策略",
            "frequency": "1"
        }

    # 保存示例策略
    with open(playbook_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    print(f"创建了 {num_samples} 条示例策略到 {playbook_path}")


def process_chunk_with_threshold(args, noun_file_path, verb_file_path, output_txt_path,
                                 playbook_path, max_strategies, merge_threshold, use_playbook=True):
    """使用阈值控制的playbook处理数据块"""
    chunk_id, chunk_data = args
    process_id = f"{os.getpid()}_{chunk_id}"
    local_results = []

    print(f"进程 {process_id} 开始处理数据块，包含 {len(chunk_data)} 个数据项，使用playbook: {use_playbook}")

    for i, data in enumerate(chunk_data):
        if i % 10 == 0:  # 每处理10个数据项打印一次进度
            print(f"进程 {process_id} 处理进度: {i + 1}/{len(chunk_data)}")

        result = process_single_item(data, noun_file_path, verb_file_path,
                                     output_txt_path, playbook_path, max_strategies,
                                     merge_threshold, process_id, i, use_playbook)
        if result:
            local_results.append(result)

    print(f"进程 {process_id} 完成数据块处理，生成 {len(local_results)} 个结果")
    return local_results


def merge_temp_files(output_txt_path, num_processes):
    """合并所有临时文件，包括新的selected_noun_keys列"""
    temp_pattern = f"{output_txt_path.rsplit('.', 1)[0]}_proc_*.txt"
    temp_files = glob.glob(temp_pattern)

    if not temp_files:
        print("未找到临时文件进行合并")
        return

    try:
        with open(output_txt_path, 'w', encoding='utf-8') as final_file:
            # 写入表头，包括新的selected_noun_keys列
            final_file.write("narration_id,participant_id,video_id,narration_timestamp,"
                             "start_timestamp,stop_timestamp,start_frame,stop_frame,"
                             "verbs,nouns,actions,confidences,selected_noun_keys\n")

            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    with open(temp_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        final_file.write(content)
                    os.remove(temp_file)
                    print(f"合并并删除临时文件: {temp_file}")
    except Exception as e:
        print(f"合并文件失败: {e}")


def save_and_test_with_threshold_playbook(csv_file_path, output_txt_path, noun_file_path, verb_file_path,
                                          num_processes=4, playbook_path='ace_playbook.json',
                                          max_strategies=50, merge_threshold=200, use_playbook=True):
    """使用阈值控制策略playbook的主函数"""
    print(f"使用阈值控制策略playbook多进程模式，进程数: {num_processes}")
    print(f"每个类别最多 {max_strategies} 条策略，合并阈值: {merge_threshold}")
    print(f"使用playbook: {use_playbook}")
    start_time = time.time()

    try:
        df = pd.read_csv(csv_file_path, on_bad_lines='skip')
        data_list = df.to_dict(orient='records')
        print(f"成功读取 {len(data_list)} 条数据")
    except Exception as e:
        print(f"读取CSV文件失败：{e}")
        return

    # 只有在使用 playbook 时才需要初始化 playbook
    if use_playbook:
        # 确保playbook文件存在且有内容
        if not os.path.exists(playbook_path) or os.path.getsize(playbook_path) < 50:
            print("检测到空playbook或文件过小，创建示例策略...")
            create_sample_strategies(playbook_path, num_samples=5)

        # 验证playbook文件可读
        try:
            initial_data = load_local_playbook(playbook_path)
            print(f"初始playbook加载成功，策略数 - one: {len(initial_data['one'])}, "
                  f"two: {len(initial_data['two'])}, three: {len(initial_data['three'])}")
        except Exception as e:
            print(f"playbook文件验证失败: {e}，重新创建...")
            create_sample_strategies(playbook_path, num_samples=5)
    else:
        print("不使用playbook，跳过playbook初始化")

    # 使用数据分块
    chunk_size = max(1, len(data_list) // num_processes)
    chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]
    print(f"将数据分成 {len(chunks)} 个块，每块约 {chunk_size} 条数据")

    all_results = []

    # 使用进程池处理数据块
    with Pool(processes=num_processes) as pool:
        print("启动进程池...")

        # 准备参数
        process_func = partial(process_chunk_with_threshold,
                               noun_file_path=noun_file_path,
                               verb_file_path=verb_file_path,
                               output_txt_path=output_txt_path,
                               playbook_path=playbook_path,
                               max_strategies=max_strategies,
                               merge_threshold=merge_threshold,
                               use_playbook=use_playbook)

        # 为每个块分配ID
        chunk_args = [(i, chunk) for i, chunk in enumerate(chunks)]

        # 并行处理所有块
        chunk_results = pool.map(process_func, chunk_args)

        # 收集所有结果
        for result in chunk_results:
            all_results.extend(result)

    # 只有在使用 playbook 时才需要最终检查合并
    if use_playbook:
        check_and_merge_playbook(playbook_path, max_strategies, merge_threshold)

    # 合并结果文件
    merge_temp_files(output_txt_path, num_processes)

    total_time = time.time() - start_time
    print(f"处理完成！总耗时: {total_time:.2f}秒, 共处理 {len(all_results)} 个有效结果，使用playbook: {use_playbook}")

    # 只有在使用 playbook 时才需要最终验证playbook策略数量
    if use_playbook:
        try:
            final_playbook = get_real_time_playbook(playbook_path)
            for key in ['one', 'two', 'three']:
                count = len(final_playbook[key])
                print(f"最终playbook {key} 类别策略数: {count}")
                if count > max_strategies:
                    print(f"警告: {key} 类别策略数 {count} 超过限制 {max_strategies}")
        except Exception as e:
            print(f"最终playbook验证失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="运行动作识别（阈值控制策略playbook版本）")
    parser.add_argument('--num_processes', type=int, default=min(6, os.cpu_count() - 1))
    parser.add_argument('--max_strategies', type=int, default=50, help="每个类别最多保留的策略数")
    parser.add_argument('--merge_threshold', type=int, default=200, help="触发合并的策略数量阈值")
    parser.add_argument('--use_playbook', action='store_true', help="是否使用playbook")
    args = parser.parse_args()

    csv_file_path = '/home/will/Mycode/action_agent/EPIC-KITCHENS_EXTRACTED/EPIC_100_val_sampled_dataset_300.csv'
    noun_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_noun_classes.csv'
    verb_file_path = '/home/will/Mydata/EPIC-KITCHENS/EPIC_100_verb_classes.csv'
    output_txt_path = 't_300/test_comparison_results_frames8_threshold.txt'
    playbook_path = 'ace_playbook_threshold.json'

    save_and_test_with_threshold_playbook(csv_file_path, output_txt_path, noun_file_path, verb_file_path,
                                          num_processes=args.num_processes,
                                          playbook_path=playbook_path,
                                          max_strategies=args.max_strategies,
                                          merge_threshold=args.merge_threshold,
                                          use_playbook=args.use_playbook)


if __name__ == '__main__':
    main()