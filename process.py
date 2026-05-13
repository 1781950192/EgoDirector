import pandas as pd


def extract_unrecognized_rows(recognized_txt_path, full_csv_path, output_csv_path='unrecognized.csv'):
    """
    Extracts rows from the full CSV that do not exist in the recognized TXT/CSV based on the first column (ID).
    Assumes both files are comma-separated, even if the recognized file has a .txt extension.

    Args:
    recognized_txt_path (str): Path to the file with recognized actions (can be .txt or .csv).
    full_csv_path (str): Path to the full CSV file without actions.
    output_csv_path (str, optional): Path to save the output CSV. Defaults to 'unrecognized.csv'.

    Returns:
    str: Path to the output CSV file.
    """
    # Read the recognized file (TXT or CSV, assuming comma-separated)
    recognized_df = pd.read_csv(recognized_txt_path, header=None)

    # Read the full CSV
    full_df = pd.read_csv(full_csv_path, header=None)

    # Extract the set of IDs from the recognized file (first column)
    recognized_ids = set(recognized_df.iloc[:, 0])

    # Filter the full DF to keep only rows where ID is not in recognized_ids
    unrecognized_df = full_df[~full_df.iloc[:, 0].isin(recognized_ids)]

    # Save the unrecognized rows to a new CSV
    unrecognized_df.to_csv(output_csv_path, index=False, header=False)

    return output_csv_path


def merge_and_sort_txt_files(file_paths, output_path):
    """
    合并多个TXT文件的内容，根据ID的三个数字部分（P后的第一个数字、第二个下划线前的数字、最后一个数字）进行分层排序，
    并将排序后的内容保存到指定的输出文件中。

    :param file_paths: 字符串列表，每个是TXT文件的路径。
    :param output_path: 字符串，合并并排序后的内容保存路径。
    """
    all_lines = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as file:
            all_lines.extend(line.strip() for line in file if line.strip())

    def sort_key(line):
        id_part = line.split(',')[0]
        parts = id_part.split('_')
        int1 = int(parts[0][1:])  # P后的数字
        int2 = int(parts[1])
        int3 = int(parts[2])
        return (int1, int2, int3)

    sorted_lines = sorted(all_lines, key=sort_key)
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write('\n'.join(sorted_lines) + '\n')

def add_none_to_lines(input_path, output_path):
    """
    在输入的 txt 文件的每一行末尾添加 ",None,None,None"，
    并将结果写入新的输出 txt 文件。

    参数:
        input_path (str): 输入文件的路径。
        output_path (str): 输出文件的路径。
    """
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # 去掉行尾换行符（如果有的话），加上 ",None,None,None" 再写入
            line = line.rstrip('\n\r')  # 保留原始内容，只去掉换行符
            outfile.write(f"{line},None,None,None\n")

if __name__ == '__main__':
    # add_none_to_lines("unrecognized_3.txt","val_comparison_results_frames8_4.txt")
    merge_and_sort_txt_files(["/home/will/Mycode/action_agent/test_comparison_results_frames8_threshold_proc_377762_0.txt","/home/will/Mycode/action_agent/test_comparison_results_frames8_threshold_proc_377763_1.txt","/home/will/Mycode/action_agent/test_comparison_results_frames8_threshold_proc_377764_2.txt","/home/will/Mycode/action_agent/test_comparison_results_frames8_threshold_proc_377765_3.txt","/home/will/Mycode/action_agent/test_comparison_results_frames8_threshold_proc_377766_4.txt","/home/will/Mycode/action_agent/test_comparison_results_frames8_threshold_proc_377767_5.txt"],'val_all.txt')
    # extract_unrecognized_rows("test_comparison_results.txt",'/home/will/Mydata/EPIC-KITCHENS/EPIC_100_validation.csv', output_csv_path='unrecognized_val.txt')