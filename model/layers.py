"""Classes for SimGNN modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiScaleAttention(torch.nn.Module):
    # Multiscale feature fusing.
    def __init__(self, num_ntype, feature_dim):
        super(MultiScaleAttention, self).__init__()
        self.attention_weights = torch.nn.Parameter(torch.randn(num_ntype, feature_dim))

    def forward(self, stacked_tensors):
        # stacked_tensors = torch.stack([graph_tensor, type_tensor, node_tensor])
        # Softmax applied to the first dimension to get attention scores
        attention_scores = F.softmax(self.attention_weights, dim=0)
        # Weighted sum of the tensors
        fused_tensor = torch.sum(attention_scores * stacked_tensors, dim=0, keepdim=True)
        return fused_tensor

class TypeNorm(nn.Module):
    """
    TypeNorm module to apply Layer Normalization for each type of nodes.
    """

    def __init__(self, type_count, feature_dim):
        """
        :param type_count: Number of unique types of nodes.
        :param feature_dim: Dimension of the node features.
        """
        super(TypeNorm, self).__init__()
        self.type_count = type_count
        self.feature_dim = feature_dim
        self.layer_norms = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(type_count)])

    def forward(self, type_list, abstract_features):
        """
        Apply Layer Normalization for each type of nodes.
        :param type_list: List of types for each node.
        :param abstract_features: Node features before Layer Normalization.
        :return type_normalized_features: Node features after applying TypeNorm.
        """
        type_normalized_features = torch.zeros_like(abstract_features)
        unique_types = torch.unique(type_list, sorted=True)

        for type_idx, unique_type in enumerate(unique_types):
            mask = type_list == unique_type
            type_features = abstract_features[mask]
            type_normalized_features[mask] = self.layer_norms[type_idx](type_features)

        return type_normalized_features

class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3,
                                                             self.args.filters_3))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1, 1)))
        representation = torch.mm(torch.t(embedding), sigmoid_scores)
        return representation

class TensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(TensorNetworkModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.args.num_ntype = 3
        if self.args.model_name in ["demo4"]:
            self.dim = self.args.num_classes * self.args.num_ntype
        elif self.args.model_name in ["SimGNN", "GMN"]:
            self.dim = self.args.filters_3 # 32
        elif self.args.model_name in ["slotGAT", "myGAT"]:
            self.dim = self.args.num_classes # 8
        elif self.args.model_name in ["RGCN", "tryout"]:
            self.dim = self.args.tensor_neurons # 16
        else:
            self.dim = self.args.num_classes

        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.dim,
                                                             self.dim,
                                                             self.args.tensor_neurons))

        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons,
                                                                   2*self.dim))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        # todo xavier normal init
        # torch.nn.init.xavier_normal_(self.weight_matrix)
        # torch.nn.init.xavier_normal_(self.weight_matrix_block)
        # torch.nn.init.xavier_normal_(self.bias)

        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        scoring = torch.mm(torch.t(embedding_1), self.weight_matrix.view(self.dim, -1))
        scoring = scoring.view(self.dim, self.args.tensor_neurons)
        scoring = torch.mm(torch.t(scoring), embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores

class SlotMatchingModule(torch.nn.Module):
    """

    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(SlotMatchingModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        pass

    def init_parameters(self):
        """
        Initializing weights.
        """
        pass

    def forward(self, ft_1, ft_2, type1, type2):
        '''
        proceed slot matching between graphs.
        :param ft_1,2: 2 graph representation. n1 * d, n2 * d
        :param type1,2: g   raph node type list, like[1,2,2,3,3,3], elements denoting node types
        :return: similarity matrix.
        '''

        sim_matrix = torch.zeros(size=(ft_1.size(dim=0),ft_2.size(dim=0)))
        for i, t1 in enumerate(type1):
            for j, t2 in enumerate(type2):
                if t1 == t2:
                    dot = ft_1[i,:] @ ft_2[j,:]
                else:
                    dot = ft_1[i,t1*self.args.num_classes:(t1+1)*self.args.num_classes] @ ft_2[j,t2*self.args.num_classes:(t2+1)*self.args.num_classes]
                sim_matrix[i, j] = dot
        return sim_matrix


# GOTSim:
