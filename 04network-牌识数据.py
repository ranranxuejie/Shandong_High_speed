import networkx as nx
import matplotlib.pyplot as plt
from tvtk.tools.visual import vector
peak = False # 是否仅保留高峰时间
import ast
import warnings
warnings.filterwarnings("ignore")
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
# data51 = pd.read_feather('./2025-05-01牌识.feather')
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

period = '五一' # '五一' '春节' '清明'
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

# 添加距离筛选：只保留100km以内的结果
results_df = results_df.sort_values('distance(m)')
filtered_results_df = results_df[results_df['distance(m)'] <= 50*1000]  # 1e5米 = 100km
flow_end_nodes = filtered_results_df['end_node'].tolist()
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
    plt.savefig(f'./figure/upstream.png')
    plt.show()
plot_subgraph_from_nodes(G, nodes_set)

#%%
subgraph = G.subgraph(nodes_set).copy()

flow_edge_ids = edge_data.set_index('end_node').loc[flow_end_nodes,:]
# 计算flow_data['时间窗序号（15分钟）']groupby后flow_hour的均值
gantry_id = flow_edge_ids['gantry_id'].dropna().tolist()

for day in [1,2,3,4,5]:
    try:
        flow_51 = pd.concat([flow_51,pd.read_csv(f'./data/Density_050{day}_1min.csv',index_col=0)],axis=0)
    except:
        flow_51 = pd.read_csv(f'./data/Density_050{day}_1min.csv',index_col=0)
        continue
gantry_id_real = []
for g in gantry_id:
    if g in flow_51.columns:
        gantry_id_real.append(g)
gantry_id = gantry_id_real
flow_51 = flow_51.loc[:,gantry_id].fillna(0)
gantry_node_dic = flow_edge_ids.reset_index().set_index('gantry_id')['end_node'].to_dict()
flow_51.columns = [gantry_node_dic[column] for column in flow_51.columns]
flow_51 = flow_51.sort_index()
flow_sequence = flow_51.copy()
flow_edge = [node_edge_dic[node] for node in flow_end_nodes]
def plot_flow_sequence(n1,n2):
    fig,ax = plt.subplots(figsize=(10,6),dpi=300)
    ax.plot(flow_sequence[n1],label=n1)
    ax.plot(flow_sequence[n2],label=n2)
    ax.legend()
    plt.show()
# plot_flow_sequence(255,282)
#%%
from statsmodels.tsa.vector_ar.var_model import VAR
n = 2
day_len = 1440//n
def granger_causality_test(y1, y2,dis):
    data = pd.DataFrame({'Y': y1, 'X': y2})
    data = data.groupby(np.arange(len(data)) // n).mean().iloc[:1440,:]
    # 移动平均
    # data = data.rolling(2).mean().dropna()
    # 一阶差分
    # data = data.diff().dropna()
    t = dis/1000
    var_stationary = VAR(data)
    min_pvalue = 1
    best_lag = 0
    for i in range(1,int(t//n)):
        # 最优滞后阶数为p值最小的
        p_values = var_stationary.fit(maxlags=i).test_causality('Y', ['X']).pvalue
        if p_values<min_pvalue:
            min_pvalue = p_values
            best_lag = i
    # best_lag = var_stationary.select_order(maxlags=int(t//n)).aic
    # model = var_stationary.fit(max(1,best_lag))
    # # F检验
    # f_test = model.test_causality('Y', ['X'])

    return min_pvalue,best_lag

node_ids = list(flow_sequence.columns)
# node_ids = [255,282,405,485,563,653,676,508,489,562]
# 计算节点之间的两两距离
distance_nodes = pd.DataFrame(index = node_ids,columns=node_ids)
for i in node_ids:
    for j in node_ids:
        # 计算两点间在subgraph之间的距离
        if i==j:
            distance_nodes.loc[i,j] = 0
            continue
        # 计算length总和
        try:
            d = nx.shortest_path_length(subgraph,source=i,target=j,weight='length')
            distance_nodes.loc[i,j] = d
        except nx.NetworkXNoPath:
            distance_nodes.loc[i,j] = np.nan

p_values = pd.DataFrame(index=node_ids, columns=node_ids)
lag = pd.DataFrame(index=node_ids, columns=node_ids)
from tqdm import tqdm
bar = tqdm(total=len(node_ids)*(len(node_ids)-1)/2)
for i in node_ids:
    for j in node_ids:
        if node_ids.index(i)<=node_ids.index(j):
            continue
        bar.update(1)
        dis = distance_nodes.loc[i,j]
        if not dis<1e10:
            continue
        y1 = flow_sequence[i]
        y2 = flow_sequence[j]
        # 对y1 和y2 进行格兰杰因果检验
        p_value = granger_causality_test(y1,y2,dis)
        p_values.loc[i, j] = p_value[0]
        lag.loc[i,j] = p_value[1]
p_values.fillna(0, inplace=True)
lag.fillna(0, inplace=True)
# 小于0.05为显著，大于为不显著
# p_values = p_values.map(lambda x:0 if x<0.05 else 1)

p_values.to_csv(f'./result/p_values/p_values{"_peak" if peak else ""}.csv')
# 合并p_values_dic的三个矩阵，取最小值，同时保留从哪个矩阵来的信息，保存为另一个矩阵

# 重新排序index和col，255,282,405,485,563,653,676,673,508,489,562

import matplotlib.patches as patches  # 导入 patches 模块
mask = np.triu(np.ones_like(p_values, dtype=bool))

plt.figure(figsize=(10, 8),dpi=300)
# distance_nodes中nan值为无限大
distance_nodes = distance_nodes.fillna(np.inf)
ax = sns.heatmap(distance_nodes,
                 # annot=lag,
                 mask=mask,
                 fmt="d", cmap="viridis",
                 cbar_kws={'label': 'P value'},
                 vmax=distance_nodes.max().iloc[0],
                 annot_kws={'size': 16})  # 添加字体大小设置

# 为p值小于0.05的单元格添加红框
for i in range(p_values.shape[0]):
    for j in range(p_values.shape[1]):
        if i<=j:
            continue
        if p_values.iloc[i, j] <0.05 :#< 0.05:
            # 创建矩形边框 (x, y)为左下角坐标，width和height为1个单元格大小
            rect = patches.Rectangle((j+0.05, i+0.05), 0.9, 0.9, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(j+0.5, i+0.5, f'{lag.iloc[i, j]*n}', color='white', fontsize=16, ha='center', va='center')

# 设置x轴和y轴的刻度标签，字体大小16
# ax.set_xticklabels(node_ids, rotation=45, ha='right',fontsize=16)
# ax.set_yticklabels(node_ids, rotation=0,fontsize=16)
plt.xlabel('Caused',fontsize=16)
plt.ylabel('Cause',fontsize=16)
plt.title(f'Granger Causality Test During {period}',fontsize=20)
plt.savefig(f'./figure/granger_heatmap{"_peak" if peak else ""}.png')
plt.show()
#%%
p = {}
for i in range(1100):
    x = np.random.random(5*i+50)
    y = x + np.random.random(5*i+50)*10
    # 模型P值
    from sklearn.feature_selection import f_regression
    f_statistic, p_values = f_regression(x.reshape(-1,1), y)
    p[5*i+50] = p_values[0]
    print(p_values)
plt.plot(p.keys(),p.values())
plt.xlabel('样本数量')
plt.ylabel('P值')
plt.show()
