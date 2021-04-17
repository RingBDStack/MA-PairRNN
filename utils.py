from name import *
import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data


def readjson(path, name):
    filename = name + "." + Json
    filepath = os.path.join(path, filename)
    f = open(filepath, "r", encoding="utf-8")
    text = json.load(f)
    f.close()
    return text


def seq_cut(sequence_list, cut_percent=0.5):
    leftseq_list = []
    rightseq_list = []
    for i, seq in enumerate(sequence_list):
        position = int(len(seq) * cut_percent)
        left_seq = seq[:position]
        right_seq = seq[position:]
        right_seq.reverse()
        leftseq_list.append(left_seq)
        rightseq_list.append(right_seq)
    return leftseq_list, rightseq_list


def pad_sequences(vectorized_seqs, seq_lengths):
    # print(len(vectorized_seqs), len(vectorized_seqs[0][0]))
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max(), len(vectorized_seqs[0][0]))).long()
    # tmp_seq_tensor = torch.ones((len(vectorized_seqs), seq_lengths.max(), len(vectorized_seqs[0][0]))).long()
    # seq_tensor = seq_tensor - tmp_seq_tensor
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        # seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        seq_tensor[idx, len(seq_tensor[idx]) - seqlen:] = torch.FloatTensor(seq)
    return seq_tensor


def random_sample_creation(sequence_list, cut_percent=0.5, neg_percent=1):
    dataset = []
    neg_num = int(len(sequence_list) * neg_percent)
    leftseq_list, rightseq_list = seq_cut(sequence_list, cut_percent)
    for i in range(len(sequence_list)):
        dataset.append((leftseq_list[i], rightseq_list[i], 1))
    for i in range(neg_num):
        sample_result = random.sample(range(0, neg_num), 2)
        left_index = sample_result[0]
        right_index = sample_result[1]
        dataset.append((leftseq_list[left_index], rightseq_list[right_index], 0))
    dataset = np.array(dataset)
    return dataset


def create_dataset(seq_list, test_percent=0.2, cut_percent=0.5, neg_percent=1):
    train_num = int(len(seq_list) * (1 - test_percent))
    train_seq_list = seq_list[:train_num]
    test_seq_list = seq_list[train_num:]
    trainset = random_sample_creation(train_seq_list, cut_percent, neg_percent)
    testset = random_sample_creation(test_seq_list, cut_percent, neg_percent)
    return trainset, testset


class PaddedTensorDataset(Dataset):
    def __init__(self, left_tensor, right_tensor, label_tensor, leftlength_tensor, rightlength_tensor):
        assert left_tensor.size(0) == right_tensor.size(0) == label_tensor.size(0)
        # self.data_tensor = data_tensor
        self.left_tensor = left_tensor
        self.leftlength_tensor = leftlength_tensor
        self.right_tensor = right_tensor
        self.rightlength_tensor = rightlength_tensor
        self.label_tensor = label_tensor

    def __getitem__(self, index):
        return self.left_tensor[index], self.right_tensor[index], self.label_tensor[index], self.leftlength_tensor[
            index], self.rightlength_tensor[index]

    def __len__(self):
        return self.left_tensor.size(0)


def create_dataloader(seqs, shuffle=True, num_workers=2, batch_size=4):
    leftseqs = seqs[:, 0]
    rightseqs = seqs[:, 1]
    labels = seqs[:, 2]
    leftseq_lengths = torch.LongTensor([len(s) for s in leftseqs])
    rightseq_lengths = torch.LongTensor([len(s) for s in rightseqs])
    left_tensor = pad_sequences(leftseqs, leftseq_lengths)
    right_tensor = pad_sequences(rightseqs, rightseq_lengths)
    label_tensor = torch.LongTensor([label for label in labels])

    return DataLoader(PaddedTensorDataset(left_tensor, right_tensor, label_tensor, leftseq_lengths, rightseq_lengths),
                      shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, drop_last=True)


def batchfirst2second(fea):
    f = torch.transpose(fea, 0, 1)
    return f


def changelabel(label):
    assert len(label.shape) == 1
    newlabel = label.clone().detach()
    for i in range(0, newlabel.shape[0]):
        if newlabel[i] == 0:
            newlabel[i] = -1
    return newlabel


def sep_lis(lis, c):
    return [lis[i:i+c] for i in range(len(lis)) if i % c == 0]


def get_train_paper(authorpaper_dict, author_list):
    train_paper = []
    for author in author_list:
        papers = authorpaper_dict[author]
        tmp = []
        for paper in papers:
            tmp.append([paper])
        train_paper.append(tmp)
    return train_paper


def get_train_nodes(leftseqs, rightseqs):
    train_nodes = set()
    leftseqs = leftseqs.numpy().tolist()
    rightseqs = rightseqs.numpy().tolist()
    for author in leftseqs:
        for paper in author:
            train_nodes.add(paper[0])
            # if paper[0] != 0:
            #     train_nodes.add(paper[0])
    for author in rightseqs:
        for paper in author:
            train_nodes.add(paper[0])
            # if paper[0] != 0:
            #     train_nodes.add(paper[0])
    return list(train_nodes)


def get_tensor(vectors, seqs):
    out_tensor = torch.stack([torch.stack([vectors[paper[0].item()]
                                           for paper in author], dim=0)
                              for author in seqs], dim=0)
    return out_tensor


def get_train_neigh_list(neighbours):
    train_neigh_list = []
    source = []
    target = []
    for node in neighbours:
        for neigh in neighbours[node]:
            source.append(int(node))
            target.append(neigh)
    # print(len(source), len(target))
    train_neigh_list.append(source)
    train_neigh_list.append(target)
    return train_neigh_list


def get_1hopsubgraph(feature_embedding, neighbours, train_nodes):
    train_nodeset = set(train_nodes)
    node_set = set(train_nodes)
    source_list = []
    target_list = []
    for source in train_nodeset:
        if source != 0 and str(source) in neighbours.keys():
            neighbor_list = neighbours[str(source)]
            for target in neighbor_list:
                node_set.add(target)
                source_list.append(source)
                target_list.append(target)
                source_list.append(target)
                target_list.append(source)
    assert len(source_list) == len(target_list)
    # print(len(train_nodeset), len(node_set))
    node_list = list(node_set)
    node_indexmap = {}
    for index, node in enumerate(node_list):
        node_indexmap[node] = index
    for i in range(len(source_list)):
        source_list[i] = node_indexmap[source_list[i]]
        target_list[i] = node_indexmap[target_list[i]]
    x = feature_embedding[node_list]
    edge_index = [source_list, target_list]
    subgraph_data = Data(x=torch.FloatTensor(x),
                         edge_index=torch.LongTensor(edge_index))
    return subgraph_data, node_indexmap


def get_2hopsubgraph(feature_embedding, neighbours, train_nodes):
    train_nodeset = set(train_nodes)
    hop1_nodeset = set()
    source_list = []
    target_list = []
    for node in train_nodeset:
        if node != 0 and str(node) in neighbours.keys():
            hop1_neighborlist = neighbours[str(node)]
            # random.shuffle(hop1_neighborlist)
            # hop1_neighborlist = hop1_neighborlist[:25]
            for hop1_neighbor in hop1_neighborlist:
                hop1_nodeset.add(hop1_neighbor)
                source_list.append(node)
                target_list.append(hop1_neighbor)
                source_list.append(hop1_neighbor)
                target_list.append(node)
    hop2_nodeset = set()
    for node in hop1_nodeset:
        if node != 0 and str(node) in neighbours.keys():
            hop2_neighborlist = neighbours[str(node)]
            # random.shuffle(hop2_neighborlist)
            # hop2_neighborlist = hop2_neighborlist[:10]
            for hop2_neighbor in hop2_neighborlist:
                hop2_nodeset.add(hop2_neighbor)
                source_list.append(node)
                target_list.append(hop2_neighbor)
                source_list.append(hop2_neighbor)
                target_list.append(node)
    # print(len(train_nodeset), len(hop1_nodeset), len(hop2_nodeset))
    assert len(source_list) == len(target_list)
    node_list = list(set.union(train_nodeset, hop1_nodeset, hop2_nodeset))
    node_indexmap = {}
    for index, node in enumerate(node_list):
        node_indexmap[node] = index
    for i in range(len(source_list)):
        source_list[i] = node_indexmap[source_list[i]]
        target_list[i] = node_indexmap[target_list[i]]
    x = feature_embedding[node_list]
    edge_index = [source_list, target_list]
    subgraph_data = Data(x=torch.FloatTensor(x),
                         edge_index=torch.LongTensor(edge_index))
    return subgraph_data, node_indexmap


def load_data(args):
    neighbours = readjson(Jsondir, "neighbours")
    feature_embedding = readjson(Jsondir, "topic_embedding")
    all_neighbours = readjson(Jsondir, "meta_neighbours")
    dataset_name = 'dataset' + args.train_test_ratio
    dataset = readjson(Jsondir, dataset_name)
    trainset = dataset["trainset"]
    trainset = np.array(trainset)
    testset = dataset["testset"]
    testset = np.array(testset)
    print("Data Loaded")

    return feature_embedding, neighbours, all_neighbours, trainset, testset


def write_data(results, args):
    import csv
    results.sort(key=lambda x: (-x[1], -x[2], -x[3], -x[4], -x[5]))
    if not os.path.exists(Csvdir):
        os.makedirs(Csvdir)
    # dataset_name = 'dataset' + args.train_test_ratio
    f = open(os.path.join(Csvdir, args.model_name + ".csv"), "w")
    writer = csv.writer(f)
    writer.writerow(["epoch", "acc", "f1", "auc", "p", "r"])
    for i in results:
        tmp = (str(i[0]), str(i[1]), str(i[2]), str(i[3]), str(i[4]), str(i[5]))
        writer.writerow(tmp)
    f.close()
