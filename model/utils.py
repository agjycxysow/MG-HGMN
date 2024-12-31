"""Data processing utilities"""
import os
from os.path import basename, isfile
from os import makedirs
from glob import glob
import json
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Data, HeteroData
from texttable import Texttable
import dgl


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

# 保存采样子图到服务器
def save_to_file(subgraphs, save_path):
    for i, subgraph in enumerate(subgraphs):
        torch.save(subgraph, f"{save_path}/{i}.pt")

def to_json(variable, file_path):
    """
    将变量写入 JSON 文件并保存。

    Args:
        variable: 要写入的变量（可以是列表、字典等支持 JSON 序列化的对象）。
        file_path: JSON 文件的保存路径。
    """
    with open(file_path, 'w') as json_file:
        json.dump(variable, json_file)

# 生成映射字典
def index_mapping(subgraph):
    nx1 = convert_to_nx(subgraph)
    # 初始化存储各节点类型的字典
    node_lists = {}
    # 遍历所有节点
    for node_type, node in nx1.nodes():
        if node_type not in node_lists:
            # 如果节点类型尚未在字典中出现，创建一个新的列表
            node_lists[node_type] = []
        # 添加节点到相应类型的列表中
        node_lists[node_type].append(node)

    map_list = {}
    # data[node_type].n_id
    for node_type, nodes in node_lists.items():
        nids = subgraph[node_type].n_id
        for node in nodes:
            if node_type not in map_list:
                # 如果节点类型尚未在字典中出现，创建一个新的列表
                map_list[node_type] = {}
            # 添加节点到相应类型的列表中，并移除 <tensor> 和括号
            map_list[node_type][node] = str(nids[nodes.index(node)]).replace('tensor(', '').replace(')', '')

    return map_list



def get_node_color(nxg):
    colors = ['#ef476f', '#ffd166', '#06d6a0', '#118ab2', '#073b4c', '#ff66ff', ] + ['#999999'] * 23
    node_colors = []
    type_list = nxg.node_type
    for i, type in enumerate(type_list):
        cur_color = colors[type]
        node_colors.append(cur_color)
    return node_colors

def get_edge_color(nxg):
    colors = ['#355070', '#6d597a', '#b56576', '#eaac8b', '#073b4c', '#3d5a80', '#ccc5b9', '#f9dcc4'] + ['#999999'] * 23
    edge_colors = []
    type_list = nxg.edge_type
    for i, type in enumerate(type_list):
        cur_color = colors[type]
        edge_colors.append(cur_color)
    return edge_colors


def save_nx_graph(graph, folder_path, file_name):

    # 修改可视化函数以接受文件夹路径和文件名参数
    pos = nx.spring_layout(graph)  # 使用 Spring layout 算法进行布局
    # plt.figure(figsize=(6, 6))  # 可根据需要调整图像大小
    nx.draw_networkx(graph, pos, with_labels=False,
                     node_color=get_node_color(graph),
                     edge_color=get_edge_color(graph),
                     node_size=500, font_size=10)
    # plt.title(f"{file_name} Visualization")
    plt.box(False)
    # plt.show()
    # 调整边距
    plt.tight_layout()
    # plt.subplots_adjust(left=0.0, right=0.0, top=0.0, bottom=0.0)
    plt.savefig(os.path.join(folder_path, f"{file_name}.png"))  # 保存图像到指定目录
    plt.close()  # 关闭当前图像，避免重复显示

# 可视化 NetworkX 图
def visualize_nx_graph(graph):
    # 将异质图数据转换为 NetworkX 图
    pos = nx.spring_layout(graph)  # 使用 Spring layout 算法进行布局
    # plt.figure(figsize=(6, 6))  # 可根据需要调整图像大小
    # nx.draw(graph, pos, with_labels=True, node_color='cyan', node_size=200, font_size=10)
    nx.draw_networkx(graph, pos, with_labels=False,
                     node_color=get_node_color(graph),
                     edge_color = get_edge_color(graph),
                     node_size=500, font_size=10)
    # plt.title(f"Graph{graph} Visualization")
    plt.show()

# PyG 图转换为 NetworkX 图
def convert_to_nx(data):
    '''
    convert PyG graph to Networkx graph for visualization
    :param data: PyG Data or HeteroData object
    :return: a Networkx graph or a MultiDiGraph if HeteroData.
    '''
    # todo: 将节点type也作为feature的一部分。
    if isinstance(data, Data):  # 处理同质图
        graph = nx.Graph()
        for i in range(data.num_nodes):
            graph.add_node(i)
        edges = data.edge_index.numpy().T.tolist()
        graph.add_edges_from(edges)
        graph = graph.to_undirected()

    elif isinstance(data, HeteroData):  # 处理异质图
        graph = nx.MultiDiGraph()
        # 确定所有节点的类型
        node_types = set()
        for edge_type in data.edge_types:
            source_type, _, target_type = edge_type
            node_types.add(source_type)
            node_types.add(target_type)

        for node_type in data.node_types:
            node_types.add(node_type)
        # 添加节点
        for node_type in node_types:
            num_nodes = data[node_type].num_nodes
            for node_index in range(num_nodes):
                graph.add_node((node_type, node_index), node_type=node_type)
        # 添加边
        for edge_type in data.edge_types:
            source_type, _, target_type = edge_type
            edge_index = data[edge_type].edge_index
            for i in range(edge_index.shape[1]):
                source = (source_type, edge_index[0, i].item())
                target = (target_type, edge_index[1, i].item())
                graph.add_edge(source, target, edge_type=edge_type)
        graph = graph.to_undirected()
    else:
        raise ValueError("Unsupported data type. Must be either Data or HeteroData.")

    return graph


def restore_tuple(transformed_tuple):
    # 从元组中提取出原始的键值对元组
    original_dict = dict(transformed_tuple[2])
    # 创建原始元组
    original_tuple = transformed_tuple[:2] + (original_dict,)
    return original_tuple

def sorted_nicely(l):
    """
    将文件名中的数字提取出来，并将其转换成整数，以便进行整数比较来对文件名进行排序。
    Sort file names in a fancy way.
    The numbers in file names are extracted and converted from str into int first,
    so file names can be sorted based on int comparison.
    :param l: A list of file names:str.
    :return: A nicely sorted file name list.
    """

    def tryint(s):
        try:
            return int(s)
        except:
            return s

    import re
    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    return sorted(l, key=alphanum_key)

def get_file_paths(dir, file_format='pt'):
    """
    Return all file paths with file_format under dir.
    :param dir: Input path.
    :param file_format: The suffix name of required files.
    :return paths: The paths of all required files.
    """
    dir = dir.rstrip('/')
    paths = sorted_nicely(glob(dir + '/*.' + file_format))
    return paths

def iterate_get_graphs(dir, file_format):
    """
    Read networkx (dict) graphs from all .gexf (.json) files under dir.
    :param dir: Input path.
    :param file_format: The suffix name of required files.
    :return graphs: Networkx (dict) graphs.
    """
    assert file_format in ['gexf', 'json', 'onehot', 'anchor', 'pt']
    graphs = []
    for file in get_file_paths(dir, file_format):
        gid = int(basename(file).split('.')[0])
        if file_format == 'pt':
            g = torch.load(file)
        elif file_format == 'gexf':
            g = nx.read_gexf(file)
            g.graph['gid'] = gid
            if not nx.is_connected(g):
                raise RuntimeError('{} not connected'.format(gid))
        elif file_format == 'json':
            # g is a dict
            g = json.load(open(file, 'r'))
            g['gid'] = gid
        elif file_format in ['onehot', 'anchor']:
            # g is a list of onehot labels
            g = json.load(open(file, 'r'))
        graphs.append(g)
    return graphs

def load_all_graphs(data_location, data_format):
    graphs = iterate_get_graphs(data_location + "/train", data_format)
    train_num = len(graphs)
    graphs += iterate_get_graphs(data_location+ "/test", data_format)
    test_num = len(graphs) - train_num
    val_num = test_num
    train_num -= val_num
    '''
    Aids split:
    700 = (420 + 140) + 140
    train : val : test = 3 : 1 : 1
    train_num = 420
    test_num = 140
    val_num = 140
    '''
    return train_num, val_num, test_num, graphs

def load_ged(ged_dict, file_name):
    '''
    list(tuple)
    ged = [(id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, [best_node_mapping])]

    id_1 and id_2 are the IDs of a graph pair, e.g., the ID of 4.json is 4.
    The given graph pairs satisfy that n1 <= n2.

    ged_value = ged_nc + ged_in + ged_ie
    (ged_nc, ged_in, ged_ie) is the type-aware ged following the setting of TaGSim.
    ged_nc: the number of node relabeling
    ged_in: the number of node insertions/deletions
    ged_ie: the number of edge insertions/deletions

    [best_node_mapping] contains 10 best matching at most.
    best_node_mapping is a list of length n1: u in g1 -> best_node_mapping[u] in g2

    return dict()
    ged_dict[(id_1, id_2)] = ((ged_value, ged_nc, ged_in, ged_ie), best_node_mapping_list)
    '''
    path = file_name
    TaGED = json.load(open(path, 'r'))
    for (id_1, id_2, ged_value) in TaGED:
        ged = (ged_value)
        ged_dict[(id_1, id_2)] = (ged)

    # for (id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, mappings) in TaGED:
    #     ta_ged = (ged_value, ged_nc, ged_in, ged_ie)
    #     ged_dict[(id_1, id_2)] = (ta_ged, mappings)

# todo aids
def load_labels(data_location, dataset_name):
    path = data_location + dataset_name + "/labels.json"
    global_labels = json.load(open(path, 'r'))
    features = iterate_get_graphs(data_location + dataset_name + "/train", "onehot") \
             + iterate_get_graphs(data_location + dataset_name + "/test", "onehot")
    print('Load one-hot label features (dim = {}) of {}.'.format(len(global_labels), dataset_name))
    return global_labels, features


def homo_onehot(graphs, num_classes):
    '''
    generate homogeneous graph index one hot features for SimGNN.
    :param graphs: homogeneous graphs
    :return: one hot feature list.
    '''

    # todo: add type-based one-hot features just as SimGNN label-based one-hot features.
    f = []
    for graph in graphs:
        h = graph.to_homogeneous(add_node_type=False, add_edge_type=False)
        i = torch.arange(h.num_nodes)
        # o = torch.nn.functional.one_hot(i, num_classes = num_classes).to(torch.float32)
        o = torch.ones(size=(h.num_nodes, num_classes), dtype=torch.float32)
        f.append(o)
    return f

def typehot_feat(node_types, num_ntype):
    """
    get a homogeneous graph type features, where this homogeneous graph is obtained by HeteroData.to_homogeneous()
    :param node_types: node type tensor like: tensor([0, 1, 2, 2, 2, 2, 3])
    :return:
    type hot feature matrix:
    tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])
    """

    # 获取节点总数和类型总数
    num_nodes = len(node_types)

    # 创建一个全零的特征矩阵，大小为（节点数 * 类型总数）
    feature_matrix = torch.zeros((num_nodes, num_ntype)).to("cuda")

    # 使用scatter_函数填充特征矩阵，实现one-hot编码
    feature_matrix.scatter_(1, node_types.view(-1, 1), 1)

    return feature_matrix

def typehot_feats(graphs, num_ntype):
    f = []
    for graph in graphs:
        h = graph.to_homogeneous()
        o = typehot_feat(h.node_type, num_ntype).to(torch.float32)
        f.append(o)
    return f

def try_gpu(i=0):
    """if gpu is available, return gpu(i), else return cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """return all available GPUs, if no GPU，return[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

def to_dgl(data, dataset):
    r"""Converts a :class:`torch_geometric.data.Data` or
    :class:`torch_geometric.data.HeteroData` instance to a :obj:`dgl` graph
    object.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The data object.

    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 3, 0], [1, 0, 2, 1, 4, 4]])
        >>> x = torch.randn(5, 3)
        >>> edge_attr = torch.randn(6, 2)
        >>> data = Data(x=x, edge_index=edge_index, edge_attr=y)
        >>> g = to_dgl(data)
        >>> g
        Graph(num_nodes=5, num_edges=6,
            ndata_schemes={'x': Scheme(shape=(3,))}
            edata_schemes={'edge_attr': Scheme(shape=(2, ))})

        >>> data = HeteroData()
        >>> data['paper'].x = torch.randn(5, 3)
        >>> data['author'].x = torch.ones(5, 3)
        >>> edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        >>> data['author', 'cites', 'paper'].edge_index = edge_index
        >>> g = to_dgl(data)
        >>> g
        Graph(num_nodes={'author': 5, 'paper': 5},
            num_edges={('author', 'cites', 'paper'): 5},
            metagraph=[('author', 'paper', 'cites')])
    """
    import dgl

    from torch_geometric.data import Data, HeteroData

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(data, Data):
        if data.edge_index is not None:
            row, col = data.edge_index
        else:
            row, col, _ = data.adj_t.t().coo()

        g = dgl.graph((row, col))

        for attr in data.node_attrs():
            g.ndata[attr] = data[attr]
        for attr in data.edge_attrs():
            if attr in ['edge_index', 'adj_t']:
                continue
            g.edata[attr] = data[attr]

        return g

    if isinstance(data, HeteroData):
        # Modify edge descriptions
        # edge_mapping = {
        #     ('author', 'to', 'paper'): ('author', 'writes', 'paper'),
        #     ('paper', 'to', 'author'): ('paper', 'by', 'author'),
        #     ('paper', 'to', 'term'): ('paper', 'has', 'term'),
        #     ('paper', 'to', 'conference'): ('paper', 'published_in', 'conference'),
        #     ('term', 'to', 'paper'): ('term', 'in', 'paper'),
        #     ('conference', 'to', 'paper'): ('conference', 'contains', 'paper'),
        #
        #     # IMDB:
        #
        #     ('movie', 'to', 'director'): ('movie', 'belongs', 'director'),
        #     ('director', 'to', 'movie'): ('director', 'directs', 'movie'),
        #     ('movie', 'to', 'actor'): ('movie', 'has', 'actor'),
        #     ('movie', '>actorh', 'actor'): ('movie', 'actorh', 'actor'),
        #     ('actor', 'to', 'movie'): ('actor', 'stars', 'movie'),
        #     ('movie', 'to', 'keyword'): ('movie', 'to', 'keyword'),
        #     ('keyword', 'to', 'movie'): ('keyword', 'on', 'movie'),
        #     # Add mappings for other edge types as needed
        #
        #     # ACM BFS
        #     ('0', 'to', '0'): ('0', 'x', '0'),
        #     ('1', 'to', '1'): ('1', 'y', '1'),
        #     ('2', 'to', '2'): ('2', 'z', '2'),
        #     ('0', 'to', '1'): ('0', 'a', '1'),
        #     ('1', 'to', '0'): ('1', 'b', '0'),
        #     ('0', 'to', '2'): ('0', 'c', '2'),
        #     ('2', 'to', '0'): ('2', 'd', '0'),
        #     ('1', 'to', '2'): ('1', 'e', '2'),
        #     ('2', 'to', '1'): ('2', 'f', '1'),
        # }
        if dataset in ["DBLP"]:
            edge_mapping = {
                ('author', 'to', 'paper'): ('author', 'writes', 'paper'),
                ('paper', 'to', 'author'): ('paper', 'by', 'author'),
                ('paper', 'to', 'term'): ('paper', 'has', 'term'),
                ('paper', 'to', 'conference'): ('paper', 'published_in', 'conference'),
                ('term', 'to', 'paper'): ('term', 'in', 'paper'),
                ('conference', 'to', 'paper'): ('conference', 'contains', 'paper'),
            }

        elif dataset in ["IMDB"]:
            edge_mapping = {
                ('movie', 'to', 'director'): ('movie', 'belongs', 'director'),
                ('director', 'to', 'movie'): ('director', 'directs', 'movie'),
                ('movie', 'to', 'actor'): ('movie', 'has', 'actor'),
                ('movie', '>actorh', 'actor'): ('movie', 'actorh', 'actor'),
                ('actor', 'to', 'movie'): ('actor', 'stars', 'movie'),
                ('movie', 'to', 'keyword'): ('movie', 'to', 'keyword'),
                ('keyword', 'to', 'movie'): ('keyword', 'on', 'movie'),
            }
        elif dataset in ["ACMNEW"]:
            edge_mapping = {
                ('paper', 'cite', 'paper'): ('paper', 'cite', 'paper'),
                ('paper', 'ref', 'paper'): ('paper', 'ref', 'paper'),
                ('paper', 'to', 'author'):('paper', 'belongto', 'author'),
                ('author', 'to', 'paper'): ('author', 'write', 'paper'),
                ('paper', 'to', 'subject'): ('paper', 'of', 'subject'),
                ('subject', 'to', 'paper'): ('subject', 'in', 'paper'),
                ('paper', 'to', 'term'): ('paper', 'at', 'term'),
                ('term', 'to', 'paper'): ('term', 'you', 'paper'),
            }
        else: assert False

        """
        The number of nodes for some node types, which is a dictionary mapping a node type T
        to the number of T-typed nodes.
        """
        num_nodes_dict = {}
        # remove sampled subgraph attributes.
        for node_type, store in data.node_items():
            if 'num_sampled_nodes' in store:
                data[node_type].num_sampled_nodes = None
            if 'input_id' in store:
                data[node_type].input_id = None
            if 'batch_size' in store:
                data[node_type].batch_size = None
            if 'num_nodes' in store:
                # data[node_type].num_nodes = torch.tensor([data[node_type].num_nodes]).to(device)
                data[node_type].num_nodes = None
            # todo: some type of nodes have no features.
            if 'x' in store:
                num_nodes_dict[node_type] = store['x'].shape[0]
            elif 'num_nodes' in store:
                num_nodes_dict[node_type] = store['num_nodes']
            elif 'n_id' in store:
                num_nodes_dict[node_type] = store['n_id'].shape[0]


        data_dict = {}
        for edge_type, store in data.edge_items():
            if store.get('edge_index') is not None:
                row, col = store.edge_index
            else:
                row, col, _ = store['adj_t'].t().coo()

            data_dict[edge_mapping[edge_type]] = (row, col)

        # todo: modify ACM BFS 1000 edge index, which is continuous from 0
        # regardless of node types, which is different from PyG official implementation
        # todo ACM dgl graph index mapping.
        '''when ACM dataset from sangs, activate the following line.'''
        if dataset in ["ACM"]:
            data_dict = map_ntype_index(data_dict, num_nodes_dict)

        g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)


        for edge_type, store in data.edge_items():
            data[edge_type].num_sampled_edges = None

        for node_type, store in data.node_items():
            # print(f"node_type, store: \n {node_type, store}")
            for attr, value in store.items():
                # print(f"attr, value: \n {attr, value}")
                g.nodes[node_type].data[attr] = value

        for edge_type, store in data.edge_items():
            for attr, value in store.items():
                if attr in ['edge_index', 'adj_t']:
                    continue
                g.edges[edge_mapping[edge_type]].data[attr] = value

        return g

    raise ValueError(f"Invalid data type (got '{type(data)}')")

def etype_emb(graph):
    '''
    get dgl heterogeneous graph edge type embedding.
    :param graph: a dgl heterogeneous graph
    :return: a Tensor like [1,1,2,1,2,3,2], each interger denoting different edge types.
    '''

    c = 0
    e_feat = []
    ntype_map = dict()
    etype_map = dict()
    g, n, e = dgl.to_homogeneous(graph, return_count=True)
    # id, type = g.ndata['_ID'].tolist(), g.ndata['_TYPE'].tolist()
    id, type = g.ndata['_ID'].tolist(), g.ndata['_TYPE'].tolist()

    for i, (id, t) in enumerate(zip(id, type)):
        ntype_map[i] = t
    # print(f'ntype_map: {ntype_map}')

    g = dgl.to_bidirected(g)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    u, v = g.edges()[0].tolist(), g.edges()[1].tolist()
    for i in range(len(graph.ntypes)):
        for j in range(i, len(graph.ntypes)):
            etype_map[(i, j)] = c
            etype_map[(j, i)] = c
            c += 1

    # print(f'etype_map: {etype_map}')

    e_feat = [etype_map[(ntype_map[i], ntype_map[j])] for i, j in zip(u, v)]
    e_feat = torch.LongTensor(e_feat)
    return g, e_feat

def map_index(edge_index):
    """
    map edge indexes in case feature matrix out of index.
    :param index: default edge index eg.
    edge_index = torch.tensor([[ 0,  1,  1,  1,  1,  1,  1],
                             [ 4,  4,  6,  7,  8,  9, 10]])
    :return: mapped edge index, continuous intergers.
                            tensor([[0, 1, 1, 1, 1, 1, 1],
                                [2, 2, 3, 4, 5, 6, 7]])
    """
    # 获取原始 tensor 中的不同元素值
    unique_values = torch.unique(edge_index)
    # 创建一个字典，将原始值映射为连续整数
    value_to_index = {}
    for i, value in enumerate(unique_values):
        value_to_index[value.item()] = i

    # 使用映射将原始 tensor 中的元素值替换为连续整数
    mapped_tensor = edge_index.clone()
    for value, index in value_to_index.items():
        mapped_tensor[mapped_tensor == value] = index

    return mapped_tensor

def add_reverse_edge(edge_index, edge_type):
    """
    for RGCN, add reverse edges on current edge index, and adjust edge type.
    :param edge_index: directed edge index
    :param edge_type: edge type
    :return:
    """
    # 创建反向边索引
    reverse_edge_index = torch.cat([edge_index[1, None, :], edge_index[0, None, :]], dim=0)

    # 创建反向边类型
    reverse_edge_type = edge_type.clone()

    # 将反向边添加到原始边索引和类型中
    extended_edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
    extended_edge_type = torch.cat([edge_type, reverse_edge_type])

    # 获取排序后的索引
    sorted_indices = torch.argsort(extended_edge_type)

    # 使用排序后的索引对边索引和类型进行重新排列
    sorted_edge_index = extended_edge_index[:, sorted_indices]
    sorted_edge_type = extended_edge_type[sorted_indices]

    # 移除重复的边
    unique_indices = torch.unique(sorted_edge_index, dim=1, return_inverse=True)[1]
    unique_edge_index = sorted_edge_index[:, unique_indices]
    unique_edge_type = sorted_edge_type[unique_indices]

    return unique_edge_index, unique_edge_type

def get_edge_types(edge_index, edge_type, edge_index_reversed, node_type):
    # Determine a unique edge type for self-loops
    max_edge_type = edge_type.max().item()
    self_loop_edge_type_offset = max_edge_type + 1  # Offset for self-loop edge types

    # Create a dictionary mapping edge indices to their corresponding types
    edge_type_dict = {tuple(edge.tolist()): edge_type[i].item()
                      for i, edge in enumerate(edge_index.t())}

    # Add self-loop edge types to the dictionary with the offset
    for i in range(node_type.size(0)):
        edge_type_dict[(i, i)] = node_type[i].item() + self_loop_edge_type_offset

    # Build the new edge type tensor based on edge_index_reversed
    new_edge_type = torch.zeros(edge_index_reversed.size(1), dtype=edge_type.dtype, device=edge_type.device)
    for i, edge in enumerate(edge_index_reversed.t()):
        new_edge_type[i] = edge_type_dict.get(tuple(edge.tolist()), 0)

    # Sort by edge type and adjust edge indices accordingly
    sorted_indices = torch.argsort(new_edge_type)
    sorted_edge_index = edge_index_reversed[:, sorted_indices]
    sorted_edge_type = new_edge_type[sorted_indices]

    return sorted_edge_index, sorted_edge_type


def map_ntype_index(edge_index, node_num_dict):
    """

    :param edge_index: 原始节点索引，从0开始的连续整数，无视节点类型。
            {('0', 'a', '1'): (tensor([0, 1], device='cuda:0'), tensor([2, 2], device='cuda:0')),
             ('0', 'c', '2'): (tensor([1, 1, 1, 1, 1], device='cuda:0'), tensor([3, 4, 5, 6, 7], device='cuda:0'))}
    :param node_num_dict: 每种类型的节点数字典 {'0': 2, '1': 1, '2': 5}
    :return: 按节点类型重新排列的字典
            {('0', 'a', '1'): (tensor([0, 1], device='cuda:0'), tensor([0, 0], device='cuda:0')),
            ('0', 'c', '2'): (tensor([1, 1, 1, 1, 1], device='cuda:0'), tensor([0, 1, 2, 3, 4], device='cuda:0'))}
    """
    # 创建一个映射，将节点类型映射为从0开始的连续整数
    type_mapping = {}
    current_index = 0
    for node_type, count in node_num_dict.items():
        type_mapping[node_type] = current_index
        current_index += count

    # 更新data_dict中的节点索引
    updated_data_dict = {}
    for edge, (src_indices, dst_indices) in edge_index.items():
        src_node_type, _, dst_node_type = edge
        src_start_index = type_mapping[src_node_type]
        dst_start_index = type_mapping[dst_node_type]
        updated_src_indices = src_indices - src_start_index
        updated_dst_indices = dst_indices - dst_start_index
        updated_data_dict[edge] = (updated_src_indices, updated_dst_indices)

    return updated_data_dict