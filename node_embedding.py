import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import Data
import torch
from utils import *
import torch.nn.functional as F


class MetaGraphSAGE(nn.Module):
    r"""
    Overview:
        A node embedding layer using GraphSAGE to generate embeddings on several metagraphs.

    Interface:
        __init__, forward
    """
    def __init__(self, args):
        super(MetaGraphSAGE, self).__init__()
        self.use_cuda = args.use_cuda
        self.meta_num = args.meta_num
        self.SAGE_hidden_dim = args.SAGE_hidden_dim
        self.sageconv1 = nn.ModuleList([SAGEConv(args.feature_dim, self.SAGE_hidden_dim) for i in range(self.meta_num)])
        self.activate_fn = nn.ELU(inplace=True)
        self.sageconv2 = nn.ModuleList([SAGEConv(self.SAGE_hidden_dim, args.embedding_dim) for i in range(self.meta_num)])

    def forward(self, meta_x, meta_edge_index):
        meta_embedding = {}
        for i in range(self.meta_num):
            x = self.activate_fn(self.sageconv1[i](meta_x[i], meta_edge_index[i]))
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.sageconv2[i](x, meta_edge_index[i])
            x = F.log_softmax(x, dim=1)
            meta_embedding[i] = x
        return meta_embedding

class MetaGAT(nn.Module):
    r"""
    Overview:
        A node embedding layer using GAT to generate embeddings on several metagraphs.

    Interface:
        __init__, forward
    """
    def __init__(self, args):
        super(MetaGAT, self).__init__()
        self.use_cuda = args.use_cuda
        self.meta_num = args.meta_num
        self.feature_dim = args.feature_dim
        self.embedding_dim = args.embedding_dim
        self.gatconv1 = nn.ModuleList([GATConv(self.feature_dim, 16, heads=4) for i in range(self.meta_num)])
        self.activate1 = nn.ELU(inplace=True)
        self.gatconv2 = nn.ModuleList([GATConv(16 * 4, self.embedding_dim) for i in range(self.meta_num)])

    def forward(self, feature_embedding, all_neighbours):
        embedding = {}
        for i, meta in enumerate(all_neighbours):
            train_neigh_list = get_train_neigh_list(all_neighbours[meta])
            if self.use_cuda:
                data = Data(x=torch.FloatTensor(feature_embedding).cuda(),
                            edge_index=torch.LongTensor(train_neigh_list).cuda())
            else:
                data = Data(x=torch.FloatTensor(feature_embedding),
                            edge_index=torch.LongTensor(train_neigh_list))
            # data.x = F.elu(self.gatconv1[i](data.x, data.edge_index))
            data.x = self.activate1(self.gatconv1[i](data.x, data.edge_index))
            data.x = F.dropout(data.x, p=0.5, training=self.training)
            data.x = self.gatconv2[i](data.x, data.edge_index)
            embedding[meta] = F.log_softmax(data.x, dim=1)
        return embedding


class GraphSAGE(nn.Module):
    r"""
    Overview:
        A node embedding layer using GraphSAGE to only generate embeddings on one original graph.

    Interface:
        __init__, forward
    """
    def __init__(self, args):
        super(GraphSAGE, self).__init__()
        self.feature_dim = args.feature_dim
        self.embedding_dim = args.embedding_dim
        self.SAGE_hidden_dim = args.SAGE_hidden_dim
        self.conv1 = SAGEConv(self.feature_dim, self.SAGE_hidden_dim)
        self.activate_fn = nn.ELU(inplace=True)
        self.conv2 = SAGEConv(self.SAGE_hidden_dim, self.embedding_dim)

    def forward(self, x, edge_index):
        x = self.activate_fn(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        embedding = F.log_softmax(x, dim=1)
        return embedding


class GAT(nn.Module):
    r"""
    Overview:
        A node embedding layer using GAT to only generate embeddings on one original graph.

    Interface:
        __init__, forward
    """
    def __init__(self, args):
        super(GAT, self).__init__()
        self.use_cuda = args.use_cuda
        self.feature_dim = args.feature_dim
        self.embedding_dim = args.embedding_dim
        self.gatconv1 = GATConv(self.feature_dim, 64, heads=4)
        self.gatconv2 = GATConv(64 * 4, self.embedding_dim)

    def forward(self, feature_embedding, neighbours):
        train_neigh_list = get_train_neigh_list(neighbours)
        if self.use_cuda:
            data = Data(x=torch.FloatTensor(feature_embedding).cuda(),
                        edge_index=torch.LongTensor(train_neigh_list).cuda())
        else:
            data = Data(x=torch.FloatTensor(feature_embedding),
                        edge_index=torch.LongTensor(train_neigh_list))
        data.x = F.elu(self.gatconv1(data.x, data.edge_index))
        data.x = F.dropout(data.x, p=0.5, training=self.training)
        data.x = self.gatconv2(data.x, data.edge_index)
        embedding = F.log_softmax(data.x, dim=1)
        return embedding

