#%% 读取2025-01-22-new文件夹下csv文件
import pandas as pd
import os
from tqdm import tqdm

# 定义文件夹路径
folder_path = '2025-01-22-new'

Density = pd.DataFrame()
# 检查文件夹是否存在
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    # 获取所有 CSV 文件
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    # 使用 tqdm 显示进度条
    for file_path in tqdm(csv_files, desc="处理 CSV 文件", unit="file"):
        file = os.path.basename(file_path)
        try:
            # 读取 CSV 文件并添加到列表中
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
        id = file.split('.')[0]
        df.columns=['pic_id','pic_time','vehicle_plate','identify_type','reliability']
        # 设置pictime格式为时间
        df['pic_time'] = pd.to_datetime(df['pic_time'])
        # 按照时间排序
        df = df.sort_values(by='pic_time')

        # reliablility大于等于95
        df = df[df['reliability'] >= 95]

        # 计算每分钟的数据量
        df['minute'] = df['pic_time'].dt.floor('min')
        df['count'] = 1
        Density_minute = df.groupby('minute')['count'].sum()
        Density_minute *= 60
        Density_minute.name = id
        Density = pd.concat([Density,Density_minute],axis=1)
#%% 仅保留2025-01-22的数据
start_time = pd.Timestamp('2025-01-22 00:00:00')
end_time = pd.Timestamp('2025-01-23 00:00:00')
Density = Density[Density.index >= start_time]
Density = Density[Density.index <= end_time]
Density = Density.fillna(0)
Density.to_csv('./data/Density.csv')
