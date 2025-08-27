import os
import csv

# 源文件夹路径
source_folder = r"D:\PycharmProjects\2025A\山东高速\2025-01-22"
# 目标文件夹路径
target_folder = os.path.join(os.path.dirname(source_folder), "2025-01-22-new")

# 如果目标文件夹不存在，则创建它
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 存储所有分类文件的写入器
writers = {}

def get_writer(category):
    """根据分类获取对应的 CSV 文件写入器，若不存在则创建"""
    if category not in writers:
        category_file = os.path.join(target_folder, f"{category}.csv")
        outfile = open(category_file, "a", newline='', encoding="utf-8")
        writer = csv.writer(outfile)
        writers[category] = (writer, outfile)
    return writers[category][0]

# 遍历源文件夹下的所有文件
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    # 按逗号分割获取各列数据
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        # 第三列（索引为 2）作为分类依据
                        category = parts[2]
                        # 移除第三列，将剩余部分作为其他信息
                        other_info = parts[:2] + parts[3:]
                        writer = get_writer(category)
                        writer.writerow(other_info)
                    elif len(parts) < 3:
                        # 若不足三列，写入空行
                        writer = get_writer('insufficient_columns')
                        writer.writerow([])

# 关闭所有打开的文件
for writer, outfile in writers.values():
    outfile.close()
