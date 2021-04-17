import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


def meta_concate(args, meta_embedding, meta_node_indexmap, train_nodes):
    r"""
    Overview:
        Directly concatenate embeddings generated from different metagraphs into a fused one.

    Arguments:
        - meta_embedding (:obj:`dict`): original generated embeddings for each metagraph
        - meta_node_indexmap (:obj:`dict`): index mapping of papers for each metagraph
        - train_nodes (:obj:`list`): nodes used for training

    Returns:
        - (:obj:`torch.Tensor`): concatenated embedding
    """
    vectors = {}
    concate_embedding = torch.stack(
        [torch.cat([meta_embedding[meta][meta_node_indexmap[meta][paper]] for meta in meta_embedding], dim=0)
         for paper in list(train_nodes)], dim=0)
    for index, paper in enumerate(list(train_nodes)):
        vectors[paper] = concate_embedding[index]
    if args.use_cuda:
        vectors[0] = torch.zeros(args.embedding_dim * args.meta_num, dtype=torch.float32, requires_grad=True).cuda()
    else:
        vectors[0] = torch.zeros(args.embedding_dim * args.meta_num, dtype=torch.float32, requires_grad=True)
    return vectors


class metapath_preference_attention(nn.Module):
    r"""
    Overview:
        Use preference attention to fuse embeddings generated from different metagraphs.

    Interface:
        __init__, forward
    """

    def __init__(self, args):
        super(metapath_preference_attention, self).__init__()
        self.use_cuda = args.use_cuda
        self.meta_num = args.meta_num
        self.embedding_dim = args.embedding_dim
        self.meta_preference_dim = args.meta_preference_dim
        self.meta_preference = nn.Parameter(
            torch.FloatTensor(self.meta_num, self.meta_preference_dim))
        self.transform_layer = nn.Sequential(nn.Linear(self.embedding_dim, self.meta_preference_dim),
                                             nn.Tanh())
        init.xavier_uniform_(self.meta_preference)
        self.register_parameter("meta_preference", self.meta_preference)

    def forward(self, meta_embedding, meta_node_indexmap, train_nodes):
        vectors = {}
        original_embedding = torch.stack([torch.stack([meta_embedding[meta][meta_node_indexmap[meta][paper]]
                                                       for meta in meta_embedding], dim=0)
                                          for paper in train_nodes], dim=0)
        # print("original_embedding:", original_embedding.size())
        transformed_embedding = self.transform_layer(original_embedding)
        # print("transformed_embedding:", transformed_embedding.size())
        sim = torch.cosine_similarity(transformed_embedding, self.meta_preference.unsqueeze(
            dim=0).expand_as(transformed_embedding), dim=2)
        meta_weights = F.softmax(sim, dim=1)
        print('meta_weights: ', meta_weights)
        # print("meta_weights:", meta_weights.size())
        mul_embedding = original_embedding.mul(
            meta_weights.unsqueeze(dim=2).expand_as(original_embedding))
        # print("mul_embedding:", mul_embedding.size())
        weighted_embedding = torch.sum(mul_embedding, 1)
        # print("weighted_embedding:", weighted_embedding.size())
        for index, paper in enumerate(list(train_nodes)):
            vectors[paper] = weighted_embedding[index]
        if self.cuda:
            vectors[0] = torch.zeros(
                self.embedding_dim, requires_grad=True).cuda()
        else:
            vectors[0] = torch.zeros(self.embedding_dim, requires_grad=True)
        return vectors


class metapath_product_attention(nn.Module):
    r"""
    Overview:
        Use product attention to fuse embeddings generated from different metagraphs.

    Interface:
        __init__, forward
    """

    def __init__(self, args):
        super(metapath_product_attention, self).__init__()
        self.use_cuda = args.use_cuda
        self.meta_num = args.meta_num
        self.embedding_dim = args.embedding_dim
        self.meta_preference = nn.Parameter(torch.FloatTensor(
            self.meta_num, self.meta_num*self.embedding_dim))
        init.xavier_uniform_(self.meta_preference)
        self.register_parameter("meta_preference", self.meta_preference)

    def forward(self, meta_embedding, meta_node_indexmap, train_nodes):
        vectors = {}
        original_embedding = torch.stack([torch.stack([meta_embedding[meta][meta_node_indexmap[meta][paper]]
                                                       for meta in meta_embedding], dim=0)
                                          for paper in train_nodes], dim=0)
        concat_embedding = torch.reshape(original_embedding, (original_embedding.size()[0],
                                                              self.meta_num*self.embedding_dim))
        product_weights = torch.mm(concat_embedding, self.meta_preference.transpose(0, 1))
        meta_weights = F.softmax(product_weights, dim=1)
        batch_weight = torch.mean(meta_weights, dim=0)
        mul_embedding = original_embedding.mul(
            meta_weights.unsqueeze(dim=2).expand_as(original_embedding))
        weighted_embedding = torch.sum(mul_embedding, 1)
        for index, paper in enumerate(list(train_nodes)):
            vectors[paper] = weighted_embedding[index]
        if self.cuda:
            vectors[0] = torch.zeros(self.embedding_dim, requires_grad=True).cuda()
        else:
            vectors[0] = torch.zeros(self.embedding_dim, requires_grad=True)
        return vectors, batch_weight


class metapath_product_average_attention(nn.Module):
    r"""
    Overview:
        Use product average attention to fuse embeddings generated from different metagraphs.

    Interface:
        __init__, forward
    """

    def __init__(self, args):
        super(metapath_product_average_attention, self).__init__()
        self.use_cuda = args.use_cuda
        self.meta_num = args.meta_num
        self.embedding_dim = args.embedding_dim
        self.meta_preference_dim = args.meta_preference_dim
        self.meta_preference = nn.Parameter(
            torch.FloatTensor(self.meta_num, self.meta_preference_dim))
        init.xavier_uniform_(self.meta_preference)
        self.register_parameter("meta_preference", self.meta_preference)
        self.transform_layer = nn.Sequential(nn.Linear(self.embedding_dim, self.meta_preference_dim),
                                             nn.Tanh())

    def forward(self, meta_embedding, meta_node_indexmap, train_nodes):
        vectors = {}
        original_embedding = torch.stack([torch.stack([meta_embedding[meta][meta_node_indexmap[meta][paper]]
                                                       for meta in meta_embedding], dim=0)
                                          for paper in train_nodes], dim=0)
        transformed_embedding = self.transform_layer(original_embedding)
        product_weights = torch.mul(transformed_embedding, self.meta_preference.unsqueeze(
            dim=0).expand_as(transformed_embedding))
        product_weights = torch.sum(product_weights, dim=2)
        average_weights = torch.mean(product_weights, dim=0)
        meta_weights = F.softmax(average_weights, dim=0)
        meta_weights = meta_weights.unsqueeze(dim=1)
        mul_embedding = original_embedding.mul(meta_weights.expand(meta_weights.size()[0], self.embedding_dim)
                                               .expand_as(original_embedding))
        weighted_embedding = torch.sum(mul_embedding, 1)
        for index, paper in enumerate(list(train_nodes)):
            vectors[paper] = weighted_embedding[index]
        if self.cuda:
            vectors[0] = torch.zeros(
                self.embedding_dim, requires_grad=True).cuda()
        else:
            vectors[0] = torch.zeros(self.embedding_dim, requires_grad=True)
        return vectors


class MLP(torch.nn.Module):
    r"""
    Overview:
        A simple MLP implementation for use of MLP_fuse.

    Interface:
        __init__, forward
    """

    def __init__(self, n_feature, n_hidden, n_output, dropout=0.5):
        super(MLP, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.process = nn.Sequential(nn.Linear(n_feature, n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, n_hidden // 2),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden // 2, n_output)
                                     )

    def forward(self, x):
        x = self.process(x)
        return x


class MLP_fuse(nn.Module):
    r"""
    Overview:
        Use MLP to fuse embeddings generated from different metagraphs.

    Interface:
        __init__, forward
    """

    def __init__(self, args):
        super(MLP_fuse, self).__init__()
        self.meta_num = args.meta_num
        self.embedding_dim = args.embedding_dim
        self.use_cuda = args.use_cuda
        self.mlp = MLP(n_feature=self.embedding_dim*self.meta_num,
                       n_hidden=512, n_output=self.embedding_dim)

    def forward(self, meta_embedding, meta_node_indexmap, train_nodes):
        vectors = {}
        concate_embedding = torch.stack([torch.cat([meta_embedding[meta][meta_node_indexmap[meta][paper]] for meta in meta_embedding], dim=0)
                                         for paper in list(train_nodes)], dim=0)
        out_embedding = self.mlp(concate_embedding)
        for index, paper in enumerate(list(train_nodes)):
            vectors[paper] = out_embedding[index]
        if self.cuda:
            vectors[0] = torch.zeros(self.embedding_dim, dtype=torch.float32, requires_grad=True).cuda()
        else:
            vectors[0] = torch.zeros(self.embedding_dim, dtype=torch.float32, requires_grad=True)
        return vectors
