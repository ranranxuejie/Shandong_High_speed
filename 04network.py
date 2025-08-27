import networkx as nx
import matplotlib.pyplot as plt
from tvtk.tools.visual import vector

peak = False # 是否仅保留高峰时间
import ast
import warnings
import seaborn as sns
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

import numpy as np
import os
os.makedirs('./fig',exist_ok=True)
os.makedirs('result/p_values',exist_ok=True)
warnings.filterwarnings("ignore")
import pandas as pd
net_data = pd.read_csv('./路网信息.csv')
data51 = pd.read_feather('./2025-05-01牌识.feather')
# 提取node信息


start_nodes = net_data[['start_idd', 'start_idd_loc']].drop_duplicates()
end_nodes = net_data[['end_idd', 'end_idd_loc']].drop_duplicates()
# 重命名列名以统一格式，确保合并时列名匹配
start_nodes = start_nodes.rename(columns={'start_idd': 'node_id', 'start_idd_loc': 'node_loc'})
end_nodes = end_nodes.rename(columns={'end_idd': 'node_id', 'end_idd_loc': 'node_loc'})
node_data = pd.concat([start_nodes, end_nodes], axis=0)
# 重命名列
node_data.columns = ['node_id', 'node_loc']
# 将字符串格式的坐标转换为元组类型
node_data['node_loc'] = node_data['node_loc'].apply(ast.literal_eval)
# 提取edge信息
edge_data = net_data[['iddd','start_idd', 'end_idd', 'section_length(meter)','gantry_id']]
# 重命名列
edge_data.columns = ['edge_id','start_node', 'end_node', 'length','gantry_id']
# 转换为数值类型
edge_data['length'] = pd.to_numeric(edge_data['length'], errors='coerce')
edge_gantry_dic = edge_data[['edge_id','gantry_id']].set_index('edge_id').dropna().to_dict()['gantry_id']
edge_node_dic = edge_data[['edge_id','end_node']].set_index('edge_id').to_dict()['end_node']
node_edge_dic = edge_data[['edge_id','end_node']].set_index('end_node').to_dict()['edge_id']
# 创建图
G = nx.DiGraph()
# 添加节点
for index, row in node_data.iterrows():
    G.add_node(row['node_id'], pos=row['node_loc'])
# 添加边
for index, row in edge_data.iterrows():
    G.add_edge(row['start_node'], row['end_node'], length=row['length'],edge_id=row['edge_id'],gantry_id=row['gantry_id'])

pos = nx.get_node_attributes(G, 'pos')

period = '春节' # '五一' '春节' '清明'
flow = pd.read_csv(f'./流量/2025{period}饱和度和速度15分钟.csv',encoding='gbk')
flow_iddd = list(set(flow['iddd'].apply(int).tolist()))
target_edge_id = 807
target_edge = edge_data[edge_data['edge_id'] == target_edge_id].iloc[0]
target_start_node = target_edge['start_node']
print(f"目标边(target_edge_id={int(target_edge_id)})的起点节点: {target_start_node}")
edge_end_node_map = dict(zip(edge_data['edge_id'], edge_data['end_node']))

# 计算每个flow_iddd边的终点到target_edge_id边起点的距离
def calculate_distance_to_target(flow_iddd,target_start_node):
    # 存储计算结果
    distance_results = []

    # 遍历每个flow_iddd计算距离
    if target_start_node is not None:
        for edge_id in flow_iddd:
            try:
                # 获取当前flow边的终点节点
                end_node = edge_end_node_map[edge_id]

                # 计算有向图中的最短路径距离
                distance = nx.dijkstra_path_length(
                    G,
                    source=end_node,
                    target=target_start_node,
                    weight='length'
                )

                # 记录结果
                distance_results.append({
                    'flow_iddd': edge_id,
                    'end_node': end_node,
                    'target_start_node': target_start_node,
                    'distance(m)': distance
                })
                print(f"已计算 flow_iddd={edge_id} 的距离: {distance}米")

            except KeyError:
                print(f"警告: flow_iddd={edge_id} 不存在于边数据中，已跳过")
            except nx.NetworkXNoPath:
                print(f"警告: 从flow_iddd={edge_id}的终点到目标起点无可达路径")
            except Exception as e:
                print(f"计算flow_iddd={edge_id}时出错: {str(e)}")

    # 将结果转换为DataFrame并显示
    if distance_results:
        results_df = pd.DataFrame(distance_results)
        print("\n距离计算结果汇总:")
        print(results_df)
    results_df = pd.DataFrame(distance_results)
    return results_df
if not os.path.exists(f'./result/distance.csv'):
    results_df = calculate_distance_to_target(flow_iddd,target_start_node)
    results_df.to_csv(f'./result/distance.csv',index=False)
else:
    results_df = pd.read_csv(f'./result/distance.csv')
flow_end_nodes = results_df['end_node'].tolist()
# 添加距离筛选：只保留100km以内的结果
filtered_results_df = results_df[results_df['distance(m)'] <= 50*1000]  # 1e5米 = 100km
nodes_set = set()

# 遍历filtered_results_df
for index, row in filtered_results_df.iterrows():
    flow_id = row['end_node']
    # 计算最短路经过的节点、边
    nodes = nx.shortest_path(G, source=flow_id, target=target_start_node, weight='length')
    nodes_set.update(nodes)
    distance = row['distance(m)']
    print(f"flow_id: {flow_id:.0f}, 距离: {distance:.2f} 米")
# 绘图，仅包含nodes_set
#%%
def plot_subgraph_from_nodes(G, nodes_set):
    # 设置中文显示
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 从原始图中提取只包含nodes_set的子图
    subgraph = G.subgraph(nodes_set).copy()

    # 创建画布
    plt.figure(figsize=(10, 6),dpi=300)
    ax = plt.gca()

    pos = {node: data['pos'] for node, data in G.nodes(data=True)}

    # 绘制节点
    edge_nodes = []
    flow_nodes = []
    for node in subgraph.nodes():
        if node in flow_end_nodes:
            flow_nodes.append(node)  # flow_iddd列表中的节点设为蓝色
        else:
            edge_nodes.append(node)  # 其他节点保持浅蓝色

    # 绘制节点
    nx.draw_networkx_nodes(subgraph, pos, node_size=50, node_color='lightblue')
    nx.draw_networkx_nodes(subgraph, pos, nodelist=flow_nodes, node_size=2000, node_color='yellow')


    # 突出显示目标节点
    # nx.draw_networkx_nodes(subgraph, pos, nodelist=[target_start_node],
    #                        node_size=2500, node_color='orange')
    # 添加标签：只显示flow_nodes和target_start_node的标签
    labels = {node: int(node) for node in flow_nodes+[target_start_node]}
    nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=14, font_family="SimHei")

    # 绘制边
    nx.draw_networkx_edges(subgraph, pos, edgelist=subgraph.edges(), arrowstyle='->',
                           arrowsize=10, edge_color='gray', width=1.5)

    # 添加边标签（edge_id）
    edge_labels = {(u, v): d['edge_id'] for u, v, d in subgraph.edges(data=True)}
    # nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=10)
    # 计算10km对应的经度度数（1度≈111km）
    degrees_for_10km = 10 / 111  # 约等于0.09度

    # 添加10km比例尺
    size_bar = AnchoredSizeBar(ax.transData,
                               degrees_for_10km, '10km', 'lower right',
                               pad=0.1,
                               color='black',
                               frameon=False,
                               size_vertical=0.005)
    ax.add_artist(size_bar)
    plt.title(f'Target Node {int(target_start_node)} 的上游节点关系图', fontsize=20)
    # 显示图形
    plt.axis('on')  # 显示坐标轴
    plt.tight_layout()
    plt.show()
plot_subgraph_from_nodes(G, nodes_set)
#%%

period = '春节' # '五一' '春节' '清明'
# 读取上游flow_nodes的edge_id
#flow_end_nodes nodes_set取交集

flow_end_nodes = set(flow_end_nodes).intersection(nodes_set)
# 加上target_start_node
flow_end_nodes.add(target_start_node)
flow_edge_ids = edge_data[edge_data['end_node'].isin(flow_end_nodes)]
# 计算flow_data['时间窗序号（15分钟）']groupby后flow_hour的均值

flow = pd.read_csv(f'./流量/2025{period}饱和度和速度15分钟.csv',encoding='gbk')

flow_data = flow[flow['iddd'].isin(flow_edge_ids['edge_id'])]
flow_data['date'] = pd.to_datetime(flow_data['date'])
# flow_data['d_date'] = flow_data['date'].apply(lambda x:x.day)
flow_data['time'] = flow_data['时间窗序号（15分钟）'] + (flow_data['date']-flow_data['date'].tolist()[0]).dt.days*96
day_flow = {}
for iddd,data_iddd in flow_data.groupby('iddd'):
    day_flow[iddd] = data_iddd.groupby('时间窗序号（15分钟）')['高峰小时流量'].mean()
flow_peak = flow_data.groupby(['时间窗序号（15分钟）'])['高峰小时流量'].mean()
# 选取flow_peak高峰小时流量最高的12个时间窗
peak_time = flow_peak.sort_values(ascending=False).iloc[:24].index.tolist()
peak_time = list(range(37,75))
flow_peak.plot()
plt.show()
plt.close()
flow_sequence = {}
for iddd,flow_data_iddd in flow_data.groupby('iddd'):
    print(iddd)
    flow_data_iddd.dropna(inplace=True)
    # 为时间窗相同的减去相应的均值
    day_flow = flow_data_iddd.groupby('时间窗序号（15分钟）')['高峰小时流量'].mean()
    # flow_data_iddd['高峰小时流量'] -= day_flow[flow_data_iddd['时间窗序号（15分钟）'].tolist()].tolist()
    # 高峰小时流量转为列表
    flow_hour = flow_data_iddd['高峰小时流量'].tolist()
    # 饱和度转为列表
    saturation = flow_data_iddd['饱和度（高峰小时）'].tolist()
    # 时间窗序号（15分钟）转为列表
    time_window = flow_data_iddd['time'].tolist()
    # 平均速度转为列表
    speed = flow_data_iddd['平均速度'].tolist()
    flow_sequence[iddd] = {
        'flow_hour': flow_hour,
        'saturation': saturation,
        'time_window': time_window,
        'speed': speed
    }
flow_sequence = pd.DataFrame(flow_sequence).T
flow_edge = [node_edge_dic[node] for node in flow_end_nodes]
def plot_flow_sequence(flow_sequence):
    # 绘制行937和938的序列折线图，每列一个子图
    fig, axes = plt.subplots(2, 2, figsize=(10, 6),dpi=300)
    i1 = 9
    i2 = 11
    i = 0
    x1 = flow_sequence.loc[flow_edge[i1], 'time_window']
    x2 = flow_sequence.loc[flow_edge[i2], 'time_window']
    for col in flow_sequence.columns:
        print(i)
        y1 = flow_sequence.loc[flow_edge[i1], col]
        y2 = flow_sequence.loc[flow_edge[i2], col]
        # 绘制折线图 - 使用axes.flat[i]访问扁平化的子图数组
        axes.flat[i].plot(x1, y1)
        axes.flat[i].plot(x2, y2)
        axes.flat[i].set_title(col)
        i += 1  # 在绘制完成后递增索引
    plt.savefig(f'./figure/flow_sequence{"_peak" if peak else ""}.png')
    plt.show()
plot_flow_sequence(flow_sequence)
#%%
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy.signal import butter, filtfilt


def granger_causality_test(y1, y2):
    # y1 = np.sin(np.linspace(0,10,100))
    # y2 = np.sin(np.linspace(0,10,100))
    # y1 = y1 + np.random.rand(100)*0.01
    # y2 = y2 + np.random.rand(100)*0.01

    def remove_period_band(data, low_period, high_period, fs=1.0, order=4):
        """
        去除周期在 [low_period, high_period] 之间的信号成分
        """
        low_freq = 1 / high_period  # 注意边界转换
        high_freq = 1 / low_period
        nyquist = 0.5 * fs
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = butter(order, [low, high], btype='bandstop')
        return filtfilt(b, a, data)


    y1_clean = remove_period_band(y1, 70,120, fs=1.0)
    y2_clean = remove_period_band(y2, 70,120, fs=1.0)

    # # 绘制滤波后的序列
    # plt.figure(figsize=(10, 4))
    # plt.plot(y1, label='Original Y')
    # plt.plot(y1_clean, label='Filtered Y')
    # plt.legend()
    # plt.show()
    data = pd.DataFrame({'Y': y1_clean, 'X': y2_clean})
    # data.plot()
    # plt.show()
    # 初始化差分数据为原始数据（修正：解决adf_data未定义问题）
    adf_data = data.copy()

    # 2. 平稳性检验（ADF），确保同阶单整（协整前提）
    def adf_test(series):
        result = adfuller(series, regression='c')  # 带常数项的ADF检验
        return result[1] < 0.05  # p<0.05则平稳

    adf_y = adf_test(adf_data['Y'])
    adf_x = adf_test(adf_data['X'])
    diff_i = 0

    # 循环差分，直到两个序列均平稳（修正：保证同阶差分）
    while not (adf_x and adf_y):
        diff_i += 1
        adf_data = adf_data.diff().dropna()  # 对当前数据差分
        # 重新检验平稳性
        adf_y = adf_test(adf_data['Y'])
        adf_x = adf_test(adf_data['X'])
        # 防止过度差分（若差分3次仍不平稳，可视为不平稳序列）
        if diff_i >= 3:
            print("警告：差分3次后仍不平稳，可能不适合协整分析")
            break
    print(f"使序列平稳的差分阶数: {diff_i}")

    # 3. 格兰杰因果检验（修正：用平稳后的序列，且基于最优滞后阶数）
    # 先为平稳数据选择最优滞后阶数（用于格兰杰检验）
    var_stationary = VAR(data)
    lag_stationary = var_stationary.select_order(maxlags=2).aic
    # 用最优滞后阶数做格兰杰检验
    gc_results = grangercausalitytests(data, maxlag=lag_stationary)
    # 提取最优滞后阶数的p值（修正：使用最优滞后阶数而非固定）
    p_value_gc = gc_results[lag_stationary][0]['ssr_ftest'][1]
    if diff_i == 0:
        return p_value_gc, p_value_gc, lag_stationary
    # 4. 协整检验与模型选择（协整要求同阶单整，这里默认diff_i=1，即I(1)）
    # 仅当差分阶数为1时（最常见协整情形），才做协整检验

    elif diff_i == 1:
        # 选择协整秩（基于原始数据，因为协整检验用非平稳序列）
        rank_test = select_coint_rank(
            data,
            det_order=0,  # 无确定性趋势
            k_ar_diff=lag_stationary,  # 与VAR滞后阶数一致
            method='trace',
            signif=0.05
        )
        print(f"协整秩: {rank_test.rank}")

        # 5. 拟合模型
        if rank_test.rank == 0:
            # 无协整关系，用平稳数据拟合VAR
            var_model = VAR(adf_data)
            var_fit = var_model.fit(maxlags=lag_stationary)
            # 格兰杰因果检验（Y是否受X影响）
            gc_Y = var_fit.test_causality('Y', 'X')
            gc_X = var_fit.test_causality('X', 'Y')
            return max(gc_X.pvalue, p_value_gc), gc_Y.pvalue, lag_stationary
        else:
            # 有协整关系，用原始数据拟合VECM
            vecm = VECM(
                data,
                k_ar_diff=lag_stationary,  # 短期动态滞后阶数
                coint_rank=rank_test.rank,  # 协整秩
                deterministic='ci'  # 协整方程带常数项
            )
            vecm_fit = vecm.fit()
            # 格兰杰因果检验（在VECM中检验）
            gc_X = vecm_fit.test_granger_causality(caused='Y')  # X是否引起Y
            gc_Y = vecm_fit.test_granger_causality(caused='X')  # Y是否引起X
            return max(gc_X.pvalue, p_value_gc), gc_Y.pvalue, lag_stationary
    elif diff_i > 1:
        # 若差分阶数≠1（非同阶单整或过度差分），直接用平稳数据拟合VAR
        var_model = VAR(adf_data)
        var_fit = var_model.fit(maxlags=lag_stationary)
        gc_Y = var_fit.test_causality('Y', 'X')
        gc_X = var_fit.test_causality('X', 'Y')
        return max(gc_X.pvalue, p_value_gc), gc_Y.pvalue, lag_stationary


cols = ['flow_hour', 'saturation', 'time_window', 'speed']
col = cols[0]

node_ids = [node_edge_dic[node] for node in [255,282,405,485,563,653,676,673,508,489,562]]

p_values = pd.DataFrame(index=node_ids, columns=node_ids)
lag = pd.DataFrame(index=node_ids, columns=node_ids)
for i in node_ids:
    x1 = flow_sequence.loc[i, 'time_window']

    for j in node_ids:
        if i==j:
            continue
        x2 = flow_sequence.loc[j, 'time_window']
        y1 = flow_sequence.loc[i, col]
        y2 = flow_sequence.loc[j, col]

        # 仅保留共同的x，先计算交集
        x = list(set(x1).intersection(set(x2)))
        if peak:
            x = list(set(x).intersection(set(peak_time)))
        x = list(set(x).intersection(set(range(96*3))))
        y1 = [y1[x1.index(x_)] for x_ in x]
        y2 = [y2[x2.index(x_)] for x_ in x]
        # 对y1 和y2 进行格兰杰因果检验
        p_value = granger_causality_test(y1,y2)
        p_values.loc[i, j] = p_value[0]
        lag.loc[i,j] = p_value[2]
p_values.fillna(0, inplace=True)
lag.fillna(0, inplace=True)
# 小于0.05为显著，大于为不显著
# p_values = p_values.map(lambda x:0 if x<0.05 else 1)

p_values.to_csv(f'./result/p_values/p_values{"_peak" if peak else ""}.csv')

real_node = [edge_node_dic[edge_id] for edge_id in p_values.columns]
# 合并p_values_dic的三个矩阵，取最小值，同时保留从哪个矩阵来的信息，保存为另一个矩阵

# 重新排序index和col，255,282,405,485,563,653,676,673,508,489,562

import matplotlib.patches as patches  # 导入 patches 模块
mask = np.triu(np.ones_like(p_values, dtype=bool))
plt.figure(figsize=(10, 8),dpi=300)
ax = sns.heatmap(p_values,
                 # annot=lag,
                 mask=mask,
                 fmt="d", cmap="viridis",
                 cbar_kws={'label': 'P value'},
                 vmax=0.1,
                 annot_kws={'size': 16})  # 添加字体大小设置

# 为p值小于0.05的单元格添加红框
for i in range(p_values.shape[0]):
    for j in range(p_values.shape[1]):
        if i<=j:
            continue
        if p_values.iloc[i, j] < 0.05:
            # 创建矩形边框 (x, y)为左下角坐标，width和height为1个单元格大小
            rect = patches.Rectangle((j+0.05, i+0.05), 0.9, 0.9, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(j+0.5, i+0.5, f'{lag.iloc[i, j]}', color='white', fontsize=16, ha='center', va='center')

# 设置x轴和y轴的刻度标签，字体大小16
ax.set_xticklabels(real_node, rotation=45, ha='right',fontsize=16)
ax.set_yticklabels(real_node, rotation=0,fontsize=16)
plt.xlabel('Caused',fontsize=16)
plt.ylabel('Cause',fontsize=16)
plt.title(f'Granger Causality Test During {period}',fontsize=20)
plt.savefig(f'./figure/granger_heatmap{"_peak" if peak else ""}.png')
plt.show()
