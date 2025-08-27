#%%
import pandas as pd
import os
from tqdm import tqdm


data51 = pd.read_feather('./2025-05-01牌识.feather')
data51 = data51[data51['reliability']>95]
data51['pic_time'] = pd.to_datetime(data51['pic_time'])
data51.to_parquet('./2025-05-02牌识-reliable.parquet')
# 设置时间间隔参数（单位：分钟），可根据需要调整（如1、5、15、30等）
interval_minutes = 15  # 此处修改为目标间隔分钟数

Density = pd.DataFrame()
# group by grantry_id
for id, group in data51.groupby('gantry_id'):
    print(id)
    # 计算指定间隔的数据量
    group['time_interval'] = group['pic_time'].dt.floor(f'{interval_minutes}min')  # 使用参数化间隔
    group['count'] = 1
    # 按指定间隔分组求和
    Density_interval = group.groupby('time_interval')['count'].sum()
    # 计算每小时流量：60分钟/间隔分钟数 = 每小时包含的间隔数
    Density_interval *= (60 / interval_minutes)
    Density_interval.name = id
    Density = pd.concat([Density,Density_interval],axis=1)
Density = Density.fillna(0)
Density.to_csv(f'./data/Density_0501_{interval_minutes}min.csv')
#%%
# 读取 流量/2025五一饱和度和速度15分钟.csv
flow = pd.read_csv('./流量/2025五一饱和度和速度15分钟.csv',encoding='gbk')
