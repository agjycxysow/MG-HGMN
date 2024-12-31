"""Models class"""
from torch_geometric.nn import GCNConv, GINConv, GATConv,RGCNConv, global_add_pool, FastRGCNConv
from layers import AttentionModule, TensorNetworkModule, TypeNorm, MultiScaleAttention
import torch
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import dgl
import math
from conv import slotGATConv, myGATConv
from torch.profiler import profile, record_function, ProfilerActivity

class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """
    def __init__(self, args, in_dim):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.device = self.args.device
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.convolution_1 = GCNConv(self.in_dim, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)

        self.attention = AttentionModule(self.args)
        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

        if not self.args.homo:
            # in_dims = [feature.shape[1] for feature in features]
            in_dims = self.args.in_dims
            self.fc_list = nn.ModuleList([nn.Linear(in_dim, self.args.pad, bias=True) for in_dim in in_dims]).to(self.device)
            for fc in self.fc_list:
                nn.init.xavier_normal_(fc.weight, gain=1.414)

        # self.layer_norms = nn.ModuleList([nn.LayerNorm(normalized_shape) for normalized_shape in shapes])


    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, edge_index, features):
        features = self.convolution_1(features, edge_index)

        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,p=self.args.dropout,training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,p=self.args.dropout,training=self.training)

        features = self.convolution_3(features, edge_index)
        return features

    def init_map(self, features):
        concatenated_features = torch.cat([fc(feature) for fc, feature in zip(self.fc_list, features)], 0)
        activated_features = F.relu(concatenated_features)
        h = F.dropout(activated_features, p=self.args.dropout, training=self.training)
        return h

    def forward(self, data):
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]

        if self.args.homo:
            features_1 = data["features_1"]
            features_2 = data["features_2"]
        else:
            # heterogeneous graph embedding needs to be
            features_1 = data["features_list_1"]
            features_2 = data["features_list_2"]

            features_1 = self.init_map(features_1)
            features_2 = self.init_map(features_2)

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        if self.args.histogram == True:

            hist = self.calculate_histogram(abstract_features_1,
                                            torch.t(abstract_features_2))

        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        if self.args.histogram == True:
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores))
        # score = score.squeeze()
        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False

        score = score.squeeze(dim=1)
        return score, pre_ged.item()

class GMN(torch.nn.Module):
    def __init__(self, args, in_dim):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super().__init__()
        self.args = args
        self.device = self.args.device
        self.in_dim = in_dim
        self.setup_layers()
    """
    GMN reimplementation from ISONET.
    https://github.com/Indradyumna/ISONET/blob/main/GraphOTSim/python/models.py
    """
    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.gated_layer = torch.nn.Linear(self.args.gcn_size[-1],
                                           self.args.gcn_size[-1])

        num_ftrs = self.in_dim;
        self.num_gcn_layers = len(self.args.gcn_size);
        self.gcn_layers = torch.nn.ModuleList([]);
        self.gcn_update_wights = torch.nn.ModuleList([]);
        for i in range(self.num_gcn_layers):
            self.gcn_layers.append(
                GCNConv(num_ftrs, self.args.gcn_size[i], bias=False))
            self.gcn_update_wights.append(torch.nn.Linear(num_ftrs * 2 + self.args.gcn_size[i], self.args.gcn_size[i]));
            num_ftrs = self.args.gcn_size[i];

        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

        if not self.args.homo:
            # in_dims = [feature.shape[1] for feature in features]
            in_dims = self.args.in_dims
            self.fc_list = nn.ModuleList([nn.Linear(in_dim, self.args.pad, bias=True) for in_dim in in_dims]).to(self.device)
            for fc in self.fc_list:
                nn.init.xavier_normal_(fc.weight, gain=1.414)

    def graph_convolutional_pass(self, edge_index_1, edge_index_2, features_1, features_2):
        """
        Making convolutional pass.
        :param graph: DGL graph.
        :param features: Feature matrix.
        :return features: List of abstract feature matrices.
        """
        for i in range(self.num_gcn_layers - 1):
            # print(features_1.shape, features_2.shape)
            conv_1_output = self.gcn_layers[i](features_1, edge_index_1)  # self.gcn_layers[i](graph_1, features_1);
            conv_2_output = self.gcn_layers[i](features_2, edge_index_2)  # self.gcn_layers[i](graph_2, features_2);

            if self.args.similarity == "cosine":
                similarity_matrix = torch.mm(F.normalize(features_1, dim=1),
                                             F.normalize(features_2, dim=1).transpose(0, 1));
            elif self.args.similarity == "euclidean":
                similarity_matrix = -torch.cdist(features_1, features_2);
            elif self.args.similarity == "dot":
                similarity_matrix = torch.mm(features_1, features_2.transpose(0, 1));
            a_1 = torch.softmax(similarity_matrix, dim=1)
            a_2 = torch.softmax(similarity_matrix, dim=0)
            attention_1 = torch.mm(a_1, features_2)
            attention_2 = torch.mm(a_2.transpose(0, 1), features_1)
            features_1 = self.gcn_update_wights[i](
                torch.cat([conv_1_output, features_1, features_1 - attention_1], dim=1));
            features_2 = self.gcn_update_wights[i](
                torch.cat([conv_2_output, features_2, features_2 - attention_2], dim=1));
            features_1 = torch.tanh(features_1)
            features_2 = torch.tanh(features_2)
            features_1 = torch.nn.functional.dropout(features_1,
                                                     p=self.args.dropout,
                                                     training=self.training)
            features_2 = torch.nn.functional.dropout(features_2,
                                                     p=self.args.dropout,
                                                     training=self.training)
        # print(features_1.shape, features_2.shape)
        conv_1_output = self.gcn_layers[-1](features_1, edge_index_1)  # self.gcn_layers[-1](graph_1, features_1);
        conv_2_output = self.gcn_layers[-1](features_2, edge_index_2)  # self.gcn_layers[-1](graph_2, features_2);

        if self.args.similarity == "cosine":
            similarity_matrix = torch.mm(F.normalize(features_1, dim=1),
                                         F.normalize(features_2, dim=1).transpose(0, 1));
        elif self.args.similarity == "euclidean":
            similarity_matrix = -torch.cdist(features_1, features_2);
        elif self.args.similarity == "dot":
            similarity_matrix = torch.mm(features_1, features_2.transpose(0, 1));
        a_1 = torch.softmax(similarity_matrix, dim=1)
        a_2 = torch.softmax(similarity_matrix, dim=0)
        attention_1 = torch.mm(a_1, features_2)
        attention_2 = torch.mm(a_2.transpose(0, 1), features_1)
        features_1 = self.gcn_update_wights[-1](
            torch.cat([conv_1_output, features_1, features_1 - attention_1], dim=1));
        features_2 = self.gcn_update_wights[-1](
            torch.cat([conv_2_output, features_2, features_2 - attention_2], dim=1));
        return features_1, features_2;

    def init_map(self, features):
        concatenated_features = torch.cat([fc(feature) for fc, feature in zip(self.fc_list, features)], 0)
        activated_features = F.relu(concatenated_features)
        h = F.dropout(activated_features, p=self.args.dropout, training=self.training)
        return h

    def forward(self, data):
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]

        if self.args.homo:
            features_1 = data["features_1"]
            features_2 = data["features_2"]
        else:
            # heterogeneous graph embedding needs to be
            features_1 = data["features_list_1"]
            features_2 = data["features_list_2"]

            features_1 = self.init_map(features_1)
            features_2 = self.init_map(features_2)

        abstract_features_1, abstract_features_2 = self.graph_convolutional_pass(edge_index_1, edge_index_2, features_1,
                                                                                 features_2)

        if self.args.readout == "max":
            pooled_features_1 = torch.max(abstract_features_1, dim=0, keepdim=True)[0].transpose(0, 1)
            pooled_features_2 = torch.max(abstract_features_2, dim=0, keepdim=True)[0].transpose(0, 1)
        elif self.args.readout == "mean":
            pooled_features_1 = torch.mean(abstract_features_1, dim=0, keepdim=True).transpose(0, 1)
            pooled_features_2 = torch.mean(abstract_features_2, dim=0, keepdim=True).transpose(0, 1)
        elif self.args.readout == "gated":
            pooled_features_1 = torch.sum(torch.sigmoid(self.gated_layer(abstract_features_1)) * abstract_features_1,
                                          dim=0, keepdim=True).transpose(0, 1)
            pooled_features_2 = torch.sum(torch.sigmoid(self.gated_layer(abstract_features_2)) * abstract_features_2,
                                          dim=0, keepdim=True).transpose(0, 1)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        scores = torch.tanh(self.fully_connected_first(scores))
        score_logit = self.scoring_layer(scores)
        score = torch.sigmoid(score_logit)

        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False
        return score.view(-1), pre_ged.item()

class RGCN(torch.nn.Module):
    '''
    https://blog.csdn.net/yzsjwd?type=blog
    '''
    def __init__(self, args):
        super(RGCN, self).__init__()
        self.args = args
        self.setup_layers()
        self.relu = F.relu
        self.dropout = self.args.dropout

    def setup_layers(self):
        in_channels = self.args.in_dim
        hidden_channels = 64
        hidden_channels_2 = 32
        out_channels = self.args.out_dim # 16
        n_layers = 3
        self.convs = torch.nn.ModuleList()
        num_relations = 6

        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations, is_sorted=True))
        # for i in range(n_layers - 2):
        #     self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
        self.convs.append(RGCNConv(hidden_channels, hidden_channels_2, num_relations, is_sorted=True))
        self.convs.append(RGCNConv(hidden_channels_2, out_channels, num_relations, is_sorted=True))

        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.args.tensor_neurons * 2,
                                                     self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def get_type_embedding(self, type_list, abstract_features):
        # get type embedding.
        type_embedding = []

        unique_types = set(type_list)
        for unique_type in unique_types:
            # 使用布尔索引来获取特定类型的行
            mask = torch.tensor(type_list) == unique_type
            sub_matrix = abstract_features[mask]
            sub_matrix = torch.mean(sub_matrix, dim=0, keepdim=True)
            type_embedding.append(sub_matrix)
        concatenated_tensor = torch.cat(type_embedding, dim=0)
        return concatenated_tensor

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        return hist
    def convolutional_pass(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i != len(self.convs)-1 :
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, data):

        edge_index_1 = data["edge_index_r_1"]
        edge_index_2 = data["edge_index_r_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        edge_type_1 = data["edge_type_1"]
        edge_type_2 = data["edge_type_2"]

        type_list_1 = data["type_list_1"]
        type_list_2 = data["type_list_2"]

        abstract_features_1 = self.convolutional_pass(features_1, edge_index_1, edge_type_1)
        abstract_features_2 = self.convolutional_pass(features_2, edge_index_2, edge_type_2)

        # type_emb_1 = self.get_type_embedding(type_list_1, abstract_features_1)
        # type_emb_2 = self.get_type_embedding(type_list_2, abstract_features_2)

        hist = self.calculate_histogram(abstract_features_1, torch.t(abstract_features_2))

        pooled_features_1 = torch.mean(abstract_features_1, dim=0, keepdim=True).transpose(0, 1)
        pooled_features_2 = torch.mean(abstract_features_2, dim=0, keepdim=True).transpose(0, 1)

        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        sim_tensor = torch.cat([scores, hist], dim=1)
        scores = torch.tanh(self.fully_connected_first(sim_tensor))
        score_logit = self.scoring_layer(scores)
        score = torch.sigmoid(score_logit)
        return score.view(-1), score_logit.view(-1)

class GREED(torch.nn.Module):
    '''
    https://github.com/idea-iitd/greed/blob/main/neuro/models.py
    '''

    def __init__(self, args):
        super(GREED, self).__init__()
        self.args = args
        self.device = self.args.device
        self.setup_layers()

    def init_map(self, features):
        concatenated_features = torch.cat([fc(feature) for fc, feature in zip(self.fc_list, features)], 0)
        activated_features = F.relu(concatenated_features)
        h = F.dropout(activated_features, p=self.args.dropout, training=self.training)
        return h

    def setup_layers(self):
        self.l2 = True
        self.n_layers = 8
        self.input_dim = self.args.in_dim
        self.hidden_dim = self.args.hidden_dim # 64
        self.output_dim = self.args.hidden_dim

        self.pre = torch.nn.Linear(self.input_dim, self.hidden_dim)

        make_conv = lambda: \
            GINConv(torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            ))

        self.convs = torch.nn.ModuleList()
        for l in range(self.n_layers):
            self.convs.append(make_conv())

        self.post = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * (self.n_layers + 1), self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim)
        )

        self.pool = global_add_pool

        self.mlp_model = torch.nn.Sequential(
            torch.nn.Linear(2 * self.output_dim, self.output_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.output_dim, 1)
        )

        if not self.args.homo:
            # in_dims = [feature.shape[1] for feature in features]
            in_dims = self.args.in_dims
            self.fc_list = nn.ModuleList([nn.Linear(in_dim, self.args.pad, bias=True) for in_dim in in_dims]).to(self.device)
            for fc in self.fc_list:
                nn.init.xavier_normal_(fc.weight, gain=1.414)

    def convolutional_pass(self, x, edge_index):
        """embedding model in GREED"""
        x = self.pre(x)
        emb = x
        xres = x
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            if i & 1:
                x += xres
                xres = x
            x = torch.nn.functional.relu(x)
            emb = torch.cat((emb, x), dim=1)
        x = emb
        x = self.pool(x, batch=None)
        x = self.post(x)
        return x


    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        if self.args.homo:
            features_1 = data["features_1"]
            features_2 = data["features_2"]
        else:
            features_1 = data["features_list_1"]
            features_2 = data["features_list_2"]

            features_1 = self.init_map(features_1)
            features_2 = self.init_map(features_2)

        abstract_features_1 = self.convolutional_pass(features_1, edge_index_1)
        abstract_features_2 = self.convolutional_pass(features_2, edge_index_2)

        if self.l2:
            score = torch.norm(abstract_features_1-abstract_features_2, dim=-1)
        else:
            score = self.mlp_model(torch.cat((abstract_features_1, abstract_features_2), dim=-1)).view(-1)
        # score = torch.nn.functional.sigmoid(score)

        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False
        return score.view(-1), pre_ged.item()


class tryout(nn.Module):
    '''
      https://blog.csdn.net/yzsjwd?type=blog
      RGCN, tryout A-10 , use cosine distance similarity matrix.
      Final MLP use ReLU.
      Add num_bases in RGCNConv().
      '''
    def __init__(self, args):
        super(tryout, self).__init__()
        self.args = args
        self.device = self.args.device
        self.setup_layers()
        self.dropout = self.args.dropout

    def setup_layers(self):
        if self.args.dataset in ["ACM"]:
            self.args.in_dim = 1902
            self.num_ntype = 3
            self.num_relations = 9  # edge: 6, self-loop: 3
        elif self.args.dataset in ["DBLP"]:
            self.args.in_dim = 128
            self.num_ntype = 4
            self.num_relations = 12  # edge: 8, self-loop: 4
        elif self.args.dataset in ["ACMNEW"]:
            self.args.in_dim = 1902
            self.num_ntype = 4
            self.num_relations = 12  # edge: 8, self-loop: 4
        elif self.args.dataset in ["IMDB"]:
            self.args.in_dim = 128
            self.num_ntype = 4
            self.num_relations = 10
        else: assert False
        self.in_channels = self.args.in_dim
        self.hidden_channels = 64
        self.out_channels = self.args.out_dim  # 16
        self.convs = torch.nn.ModuleList()
        self.n_heads = 8
        self.total_dim = self.out_channels * self.n_heads
        self.meantype = False

        self.convs.append(RGCNConv(self.in_channels, self.hidden_channels, self.num_relations, is_sorted=True, num_bases=15))
        self.convs.append(RGCNConv(self.hidden_channels, self.hidden_channels, self.num_relations, is_sorted=True, num_bases=15))
        self.convs.append(RGCNConv(self.hidden_channels, self.hidden_channels, self.num_relations, is_sorted=True, num_bases=15))
        self.convs.append(RGCNConv(self.hidden_channels, self.out_channels, self.num_relations, is_sorted=True, num_bases=15))

        # self.convs.append(FastRGCNConv(self.in_channels, self.hidden_channels, self.num_relations, is_sorted=True))
        # self.convs.append(FastRGCNConv(self.hidden_channels, self.hidden_channels, self.num_relations, is_sorted=True))
        # self.convs.append(FastRGCNConv(self.hidden_channels, self.hidden_channels, self.num_relations, is_sorted=True))
        # self.convs.append(FastRGCNConv(self.hidden_channels, self.out_channels, self.num_relations, is_sorted=True))

        self.tensor_network = TensorNetworkModule(self.args)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)
        self.fc_1 = torch.nn.Linear(self.args.bottle_neck_neurons * (self.num_ntype + 2),
                                    self.args.bottle_neck_neurons)

        # cross attention
        self.fc_query_global = nn.Linear(self.out_channels, self.total_dim, bias=False)
        self.fc_key_global = nn.Linear(self.out_channels, self.total_dim, bias=False)
        self.fc_value_global = nn.Linear(self.out_channels, self.total_dim, bias=False)
        self.fc_heads_global = nn.Linear(self.n_heads, 1, bias=False)
        self.MHA_global = torch.nn.MultiheadAttention(embed_dim=self.total_dim, num_heads=self.n_heads, dropout=0.0)

        if not self.args.homo:
            # in_dims = [feature.shape[1] for feature in features]
            in_dims = self.args.in_dims
            self.fc_list = nn.ModuleList([nn.Linear(in_dim, self.args.pad, bias=True) for in_dim in in_dims]).to(
                self.device)
            for fc in self.fc_list:
                nn.init.xavier_normal_(fc.weight, gain=1.414)

        # Initialize TypeNorm for different node types for each feature dimension
        # self.type_norm_64 = TypeNorm(self.num_ntype, self.hidden_channels)  # For features with 64 dimensions
        # self.type_norm_32 = TypeNorm(self.num_ntype, self.hidden_channels_2)  # For features with 32 dimensions

        self.ln = torch.nn.LayerNorm(self.out_channels)
        self.type_norm = TypeNorm(self.num_ntype, self.out_channels)


    def cosine_distance(self, features_1, features_2):
        similarity_matrix = torch.mm(F.normalize(features_1, dim=1),
                                     F.normalize(features_2, dim=1).transpose(0, 1));
        return similarity_matrix

    def get_type_embedding(self, type_list, abstract_features, mean=True):
        # get type embedding.
        type_embedding = []

        unique_types = set(type_list.tolist())
        for unique_type in unique_types:
            # 使用布尔索引来获取特定类型的行
            mask = type_list == unique_type
            sub_matrix = abstract_features[mask]
            if mean:
                sub_matrix = torch.mean(sub_matrix, dim=0, keepdim=True)
            else:
                sub_matrix = torch.sum(sub_matrix, dim=0, keepdim=True)
            type_embedding.append(sub_matrix)
        concatenated_tensor = torch.cat(type_embedding, dim=0)
        return concatenated_tensor

    def calculate_histogram(self, dot):
        """
        Calculate histogram from similarity matrix.

        :return hist: Histsogram of similarity scores.
        """
        # scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = dot.detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, x, edge_index, edge_type, type_list):
        if not self.args.homo:
            features = [fc(feature) for fc, feature in zip(self.fc_list, x)]
            x = torch.cat(features, 0)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, p=self.args.dropout, training=self.training)
        for i, conv in enumerate(self.convs):
            x_res = x
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                if i != 0:
                    x += x_res
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.args.ln:
            x = self.ln(x)
            # x = self.type_norm(type_list, x)
        return x

    def forward(self, data):
        edge_index_1 = data["edge_index_r_1"]
        edge_index_2 = data["edge_index_r_2"]
        if self.args.homo:
            features_1 = data["features_1"]
            features_2 = data["features_2"]
        else:
            # heterogeneous graph embedding needs to be
            features_1 = data["features_list_1"]
            features_2 = data["features_list_2"]

        edge_type_1 = data["edge_type_1"]
        edge_type_2 = data["edge_type_2"]

        type_list_1 = data["type_list_1"]
        type_list_2 = data["type_list_2"]

        abstract_features_1 = self.convolutional_pass(features_1, edge_index_1, edge_type_1, type_list_1)
        abstract_features_2 = self.convolutional_pass(features_2, edge_index_2, edge_type_2, type_list_2)

        # global cross attention to enhance graph embedding,
        gq_1 = self.fc_query_global(abstract_features_1)
        gk_1 = self.fc_key_global(abstract_features_1)
        gv_1 = self.fc_value_global(abstract_features_1)

        gq_2 = self.fc_query_global(abstract_features_2)
        gk_2 = self.fc_key_global(abstract_features_2)
        gv_2 = self.fc_value_global(abstract_features_2)

        # global cross attention embedding
        gca_emb_1, _ = self.MHA_global(gq_1, gk_2, gv_2, need_weights=False)
        gca_emb_1 = gca_emb_1.view(-1, self.n_heads, self.out_channels).transpose(1, 2)
        gca_emb_2, _ = self.MHA_global(gq_2, gk_1, gv_1, need_weights=False)
        gca_emb_2 = gca_emb_2.view(-1, self.n_heads, self.out_channels).transpose(1, 2)
        # enhanced type embedding through cross attention. ca for cross attention.
        gca_emb_1 = self.fc_heads_global(gca_emb_1).view(-1, 1, self.out_channels).squeeze()
        gca_emb_2 = self.fc_heads_global(gca_emb_2).view(-1, 1, self.out_channels).squeeze()

        # Matching
        # Graph-level: NTN
        pooled_features_1 = torch.mean(gca_emb_1, dim=0, keepdim=True).transpose(0, 1)
        pooled_features_2 = torch.mean(gca_emb_2, dim=0, keepdim=True).transpose(0, 1)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        # type-level hist.
        # type weight on global similarity matrix. dim: 3 * 3.
        type_emb_1 = self.get_type_embedding(type_list_1, gca_emb_1, self.meantype)
        type_emb_2 = self.get_type_embedding(type_list_2, gca_emb_2, self.meantype)
        # ca_dot = torch.mm(type_emb_1, torch.t(type_emb_2))
        ca_dot = self.cosine_distance(type_emb_1, type_emb_2)
        ca_type_hist = self.calculate_histogram(ca_dot)

        type_hists = []

        # node-level type hists.
        # unique_types = torch.unique(torch.cat((type_list_1, type_list_2)))
        unique_types = torch.arange(0, self.num_ntype, dtype=torch.long)
        for unique_type in unique_types:
            # 使用布尔索引来获取特定类型的行
            mask_1 = type_list_1 == unique_type
            mask_2 = type_list_2 == unique_type
            # if there is difference between types of input graphs.
            if (not mask_1.any()) or (not mask_2.any()):
                sub_dot = torch.zeros(size=(1,1), dtype=torch.float).to(self.device)
            else:
                sub_matrix_1 = gca_emb_1[mask_1]
                sub_matrix_2 = gca_emb_2[mask_2]
                # sub_dot = torch.mm(sub_matrix_1, torch.t(sub_matrix_2))
                sub_dot = self.cosine_distance(sub_matrix_1, sub_matrix_2)
            type_hist = self.calculate_histogram(sub_dot)
            # sub_matrix = torch.mean(sub_matrix, dim=0, keepdim=True)
            type_hists.append(type_hist)
        node_dot = self.cosine_distance(gca_emb_1, gca_emb_2)
        type_hists = torch.cat(type_hists, dim=1)

        sim_tensor = torch.cat([ca_type_hist, type_hists, scores], dim=1)
        scores = torch.tanh(self.fc_1(sim_tensor))
        # scores = torch.relu(self.fc_1(sim_tensor))
        score_logit = self.scoring_layer(scores)
        score = torch.sigmoid(score_logit)

        # calculate metrics, rho, tau, pk10, pk20. recover GED back.
        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False
        return score.view(-1), pre_ged.item(), ca_dot, node_dot



