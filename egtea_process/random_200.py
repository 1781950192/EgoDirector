import random


def random_lines_from_file(filename, num_lines=200):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    total_lines = len(lines)
    if total_lines <= num_lines:
        selected_lines = lines
    else:
        selected_lines = random.sample(lines, num_lines)

    return selected_lines


# 使用示例
if __name__ == "__main__":
    filename = "/home/will/Mydata/EGTEA++/EGTEA/Action_Annotations/test_split1.txt"  # 替换为你的文件名
    output_filename = "egtea_200_lines.txt"

    random_lines = random_lines_from_file(filename, 200)

    with open(output_filename, 'w', encoding='utf-8') as out_file:
        out_file.writelines(random_lines)

    print(f"已从 {filename} 中随机选取 {len(random_lines)} 行，保存到 {output_filename}")