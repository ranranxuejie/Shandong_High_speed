#%%
max_dis = 70
import zipfile
import networkx as nx
import matplotlib.pyplot as plt
import os
import warnings
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
import numpy as np
from matplotlib.colors import BoundaryNorm
warnings.filterwarnings("ignore")
import pandas as pd
net_data = pd.read_csv('./路网信息.csv')
start_nodes = net_data[['start_idd', 'start_idd_loc']].drop_duplicates()
end_nodes = net_data[['end_idd', 'end_idd_loc']].drop_duplicates()
start_nodes = start_nodes.rename(columns={'start_idd': 'node_id', 'start_idd_loc': 'node_loc'})
end_nodes = end_nodes.rename(columns={'end_idd': 'node_id', 'end_idd_loc': 'node_loc'})
node_data = pd.concat([start_nodes, end_nodes], axis=0)
node_data.columns = ['node_id', 'node_loc']

path_count = pd.read_csv('./data/path_count.csv',index_col=0)
path_count_stack = path_count.stack().dropna()
# 转换为DataFrame

# 提取edge信息
edge_data = net_data[['iddd','start_idd', 'end_idd', 'section_length(meter)','gantry_id']]
# 重命名列
edge_data.columns = ['edge_id','start_node', 'end_node', 'length','gantry_id']
# 转换为数值类型
edge_data['length'] = pd.to_numeric(edge_data['length'], errors='coerce')
edge_gantry_dic = edge_data[['edge_id','gantry_id']].set_index('edge_id').dropna().to_dict()['gantry_id']
gantry_edge_dic = edge_data[['edge_id','gantry_id']].set_index('gantry_id').to_dict()['edge_id']
edge_node_dic = edge_data[['edge_id','end_node']].set_index('edge_id').to_dict()['end_node']
node_edge_dic = edge_data[['edge_id','end_node']].set_index('end_node').to_dict()['edge_id']

# 创建图
G = nx.DiGraph()
# 添加节点
for index, row in node_data.iterrows():
    G.add_node(row['node_id'], pos=eval(row['node_loc']))
# 添加边
for index, row in edge_data.iterrows():
    G.add_edge(row['start_node'], row['end_node'], length=row['length'],
               edge_id=row['edge_id'],gantry_id=row['gantry_id'])
pos = nx.get_node_attributes(G, 'pos')

path_count_df = path_count_stack.reset_index()
path_count_df.columns = ['start_node', 'end_node', 'count']
path_count_df['start_node'] = path_count_df['start_node'].astype(int)
path_count_df['end_node'] = path_count_df['end_node'].astype(float)
path_count_df['distance'] = path_count_df.apply(lambda row: nx.dijkstra_path_length(G, row['start_node'], row['end_node'], weight='length'), axis=1)
path_count_df['path'] = path_count_df.apply(lambda row: len(nx.dijkstra_path(G, row['start_node'], row['end_node'], weight='length')), axis=1)

# 将count列数值附加到edge的数据上
edge_data = edge_data.merge(path_count_df, on=['start_node', 'end_node'], how='left')
edge_data['count'] = edge_data['count'].fillna(0)
# 归一化
edge_data['count_'] = (edge_data['count'] - edge_data['count'].min()) / (edge_data['count'].max() - edge_data['count'].min())
edge_data['count_'] *= 10
# 创建图
G = nx.DiGraph()
# 添加节点
for index, row in node_data.iterrows():
    G.add_node(row['node_id'], pos=eval(row['node_loc']))
# 添加边
for index, row in edge_data.iterrows():
    G.add_edge(row['start_node'], row['end_node'], length=row['length'],
               count=row['count'],
               edge_id=row['edge_id'], gantry_id=row['gantry_id'])


def plot_G():
    # 绘制图
    plt.figure(figsize=(12, 8))
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=5, node_color='lightblue')

    # 绘制边
    edge_weights = [G[u][v]['count'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=edge_weights, edge_color='gray')
    plt.title("Graph Visualization")
    plt.show()
def gephi_G():
    # 创建图
    G = nx.DiGraph()
    # 添加节点
    for index, row in node_data.iterrows():
        G.add_node(row['node_id'], pos_x=eval(row['node_loc'])[0], pos_y=eval(row['node_loc'])[1])
    # 添加边
    for index, row in edge_data.iterrows():
        G.add_edge(row['start_node'], row['end_node'], length=row['length'],
                   count=row['count'],
                   edge_id=row['edge_id'], gantry_id=row['gantry_id'])
    nx.write_gexf(G, 'network_large.gexf')
#%%
flow = pd.read_csv(f'./流量/2025五一饱和度和速度15分钟.csv',encoding='gbk')
flow_iddd = list(set(flow['iddd'].apply(int).tolist()))
edge_end_node_map = dict(zip(edge_data['edge_id'], edge_data['end_node']))
target_edges = pd.read_excel('./拥堵合并新.xlsx')
gantry_etc = pd.read_csv('./dim_mdm_etc_gantry_ri.csv')[['id$$国标ID','direction$$方向 1-上行 2-下行','opma_road_name$$所属高速公路路线名称（运营业务）','stake_num$$桩号']]
gantry_etc.columns = ['gantry_id','direction','road_name','stake_num']
gantry_etc['stake_num'] = gantry_etc['stake_num'].apply(lambda x: x.split('+')[0][1:])
target_start_nodes = []
for _,row in target_edges.iterrows():
    road = row['路线名']
    start_K,end_K = row[['开始桩号','结束桩号']]
    direction = row['上下行']
    target_gantrys = gantry_etc[(gantry_etc['road_name']==road)&(gantry_etc['direction']==direction)]
    # 选择stack_num之差绝对值最小的
    target_gantrys['stake_num_'] = target_gantrys['stake_num'].apply(lambda x: abs(int(x)-int((start_K+end_K)/2)))
    target_gantrys = target_gantrys[target_gantrys['stake_num_']==target_gantrys['stake_num_'].min()]
    target_gantry = target_gantrys['gantry_id'].tolist()[0]
    try:
        target_start_node = edge_end_node_map[gantry_edge_dic[target_gantry]]
    except KeyError:
        print(target_gantry,'No edge')
        continue
    try:
        contribution = pd.read_csv(f'./data/contribution/contribution_{target_start_node:.1f}.csv')
    except FileNotFoundError:
        print(target_start_node,'No contribution')
        continue
    target_start_nodes.append(target_start_node)

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
target_start_node = target_start_nodes[0]
for target_start_node in target_start_nodes:
    if not os.path.exists(f'./result/distance_{target_start_node:.1f}.csv'):
        results_df = calculate_distance_to_target(flow_iddd,target_start_node)
        results_df.to_csv(f'./result/distance_{target_start_node:.1f}.csv',index=False)
    else:
        results_df = pd.read_csv(f'./result/distance_{target_start_node:.1f}.csv')

    # 添加距离筛选：只保留100km以内的结果
    results_df = results_df.sort_values('distance(m)')
    filtered_results_df = results_df[results_df['distance(m)'] <= max_dis*1000]  # 1e5米 = 100km
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
    G_sub = G.subgraph(nodes_set).copy()
    edges_sub = G_sub.edges
    edge_weights = np.array([G_sub[u][v]['count'] for u, v in edges_sub])
    # # 保存G_sub为network库可以读取的格式
    # nx.write_graphml(G_sub, 'network_sub.graphml')
    # a=nx.read_graphml('network_sub.graphml')
    #%%
    transition = pd.read_csv('./data/transition_counts.csv',index_col=0)
    for _,row in transition.iterrows():
        dis = G.edges[(row['path'],row['path_end'])]['length']
    transition['distance'] = transition.apply(lambda x: G.edges[(x['path'],x['path_end'])]['length'],axis=1)
    transition['speed'] = transition['distance']/transition['time_delta']*3.6
    # 大于120则速度为120
    transition['speed'] = transition['speed'].apply(lambda x: 144 if x > 144 else x)
    transition.set_index(['path','path_end'],inplace=True)
    contribution = pd.read_csv(f'./data/contribution/contribution_{target_start_node:.1f}.csv')
    # 筛选出contribution中node在nodes_set中的数据
    contribution = contribution[contribution['path'].isin(G_sub.nodes)]
    edge_contribution = {}
    # 遍历节点
    for _,row in contribution.sort_values('count',ascending=True).iterrows():
        node = row['path']
        # 计算到目标节点经过的边
        edges = nx.shortest_path(G_sub, source=node, target=target_start_node, weight='length')
        if len(edges) >= 2:
            for i in range(len(edges)-1):
                edge = (edges[i],edges[i+1])
                edge_contribution.update({edge:row['count']})
    # 新增edge的contribution属性，更新边的contribution
    for edge in G_sub.edges:
        try:
            speed = transition.loc[(edge[0], edge[1]), 'speed']
            G_sub[edge[0]][edge[1]]['contribution'] = edge_contribution[edge]
            G_sub[edge[0]][edge[1]]['speed'] = speed
        except:
            G_sub[edge[0]][edge[1]]['contribution'] = 0
            G_sub[edge[0]][edge[1]]['speed'] = 0

    # 引入颜色映射红绿黄
    edge_weights = np.array([G_sub[u][v]['contribution'] for u, v in G_sub.edges])
    cmap = plt.get_cmap('RdYlGn')
    edge_colors_speed = np.array([G_sub[u][v]['speed'] for u, v in G_sub.edges])
    # 归一化
    edge_colors_speed = edge_colors_speed
    edge_colors = cmap(edge_colors_speed / 120)
    #%%
    # 绘制子图
    fig,ax = plt.subplots(figsize=(12, 8),dpi=300)
    # 绘制边，不画箭头，颜色为颜色映射，width越大越红
    nx.draw_networkx_edges(G_sub, pos, edgelist=edges_sub, width=edge_weights/4, edge_color=edge_colors,arrows=False,
                           connectionstyle='arc3,rad=10')
    # 绘制节点
    nx.draw_networkx_nodes(G_sub, pos, node_size=40, node_color='lightblue')
    # 突出绘制目标节点
    nx.draw_networkx_nodes(G_sub, pos, nodelist=[target_start_node], node_size=100, node_color='red')
    # 绘制图例，首先是颜色-速度的图例
    # 定义离散边界（示例：5个区间，0-24-48-72-96-120）
    boundaries = [int(i) for i in np.linspace(0,120,7)]
    # 创建边界归一化器，将数据分到指定区间
    norm = BoundaryNorm(boundaries, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) # 必须设置空数组以激活颜色条
    # 添加颜色条并设置标签和缩放比例
    cbar = plt.colorbar(sm, shrink=0.8,ax=ax)
    cbar.set_ticks(boundaries)
    cbar.set_label('速度 (km/h)')
    cbar.set_ticklabels([f'{b}' for b in boundaries])  # 基础版：直接显示边界值

    # 绘制图例，其次是宽度-流量的图例
    # 绘制一个横着的较长的等腰三角形，宽度为10，高度为1，颜色为红色，放置在图的右上角
    import matplotlib.patches as patches
    ax_lim = ax.get_xlim()
    width = 0.03
    ax.add_patch(patches.Polygon([[0.77, 0.85-width], [0.77, 0.85+width], [0.95, 0.85]], color='gray', transform=ax.transAxes))
    ax.text(0.86, 0.9, '贡献率', transform=ax.transAxes, fontsize=12, color='black',ha='center')
    ax.text(0.77,0.78,'100%',transform=ax.transAxes,ha='center')# 居中
    ax.text(0.95,0.78,'0%',transform=ax.transAxes,ha='center')
    plt.title("Graph Visualization")
    # 使用背景，网格点样式
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_points = np.linspace(x_min, x_max, 10)  # 10个x轴点
    y_points = np.linspace(y_min, y_max, 10)  # 10个y轴点
    xx, yy = np.meshgrid(x_points, y_points)
    ax.scatter(xx.ravel(), yy.ravel(), s=20, color='gray',alpha=0.3, marker='o', zorder=5,
               #不绘制轮廓
               edgecolors='none')
    plt.savefig('./figure/graph_contribution_speed/'+str(target_start_node)+'.png')
    plt.show()
