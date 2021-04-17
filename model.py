import torch
import torch.nn as nn
from utils import *
from node_embedding import *
from meta_fuse import *
from PairRNN import *
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from mytest import *
import re
from tensorboardX import SummaryWriter


def weights_init(m):
    r"""
    Overview:
        A weight initialization method for nn.Linear and nn.BatchNorm1d.

    Arguments:
        - m (:obj:`nn.Module`): a nn.Module to be initialized, can only be either nn.Linear or nn.BatchNorm1d
    """
    if isinstance(m, (nn.Linear)):
        # nn.init.xavier_normal_(m.weight.data)
        # nn.init.kaiming_normal_(m.weight, mode='fan_in')
        nn.init.orthogonal_(m.weight)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Net(nn.Module):  # with GNN, with meta_fuse
    r"""
    Overview:
        A complete Net which generates node embeddings on each metagraph, fuse those embeddings 
        and measure the similarity between two paper sequences.

    Interface:
        __init__, forward
    """

    def __init__(self, args, node_embedding_layer, meta_fuse_layer, pairRNN):
        r"""
        Overview:
            An initialization method for Net.

        Arguments:
            - args (:obj:`dict`): args setting
            - node_embedding_layer (:obj:`nn.Module`): a node embedding layer to generate node embeddings
            - meta_fuse_layer (:obj:`nn.Module`): a meta fuse layer to fuse several embeddings from node embedding layer
            - pairRNN (:obj:`nn.Module`): a Siamese Network consisting of two RNNs to measure the similarity between a paper sequence pair
        """
        super(Net, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.use_cuda = args.use_cuda
        self.node_embedding_layer = node_embedding_layer
        self.meta_fuse_layer = meta_fuse_layer
        self.pairRNN = pairRNN

    def forward(self, leftseqs, rightseqs, meta_graph_data, meta_node_indexmap, train_nodes):
        r"""
        Overview:
            Net forward method.

        Arguments:
            - leftseqs (:obj:`list`): a list containing all left paper indices of the pair paper sequence
            - rightseqs (:obj:`list`): a list containing all right paper indices of the pair paper sequence
            - meta_graph_data (:obj:`dict`): graph node and edge info for different metagraphs
            - meta_node_indexmap (:obj:`nn.Module`): a meta fuse layer to fuse several embeddings from node embedding layer
            - train_nodes (:obj:`list`): nodes used for training
        """
        if self.use_cuda:
            meta_x = [meta.x.cuda() for meta in meta_graph_data]
            meta_edge_index = [meta.edge_index.cuda() for meta in meta_graph_data]
        else:
            meta_x = [meta.x for meta in meta_graph_data]
            meta_edge_index = [meta.edge_index for meta in meta_graph_data]
        meta_embedding = self.node_embedding_layer(meta_x, meta_edge_index)
        vectors, att_weight = self.meta_fuse_layer(meta_embedding, meta_node_indexmap, train_nodes)
        left_tensor = get_tensor(vectors, leftseqs)
        right_tensor = get_tensor(vectors, rightseqs)
        left_out, right_out, outputs = self.pairRNN(left_tensor, right_tensor)
        return left_out, right_out, outputs, att_weight


class Net_withoutGNN(nn.Module):  # without GNN, without meta_fuse
    r"""
    Overview:
        A variant Net which directly measure the similarity between two paper sequences.

    Interface:
        __init__, forward
    """

    def __init__(self, args, pairRNN):
        super(Net_withoutGNN, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.use_cuda = args.use_cuda
        self.pairRNN = pairRNN

    def forward(self, leftseqs, rightseqs, feature_vetor):
        for node in feature_vetor:
            feature_vetor[node] = feature_vetor[node].cuda()
        left_tensor = get_tensor(feature_vetor, leftseqs)
        right_tensor = get_tensor(feature_vetor, rightseqs)
        left_out, right_out, outputs = self.pairRNN(left_tensor, right_tensor)
        return left_out, right_out, outputs


class Net_withGNN(nn.Module):  # with GNN, without meta_fuse
    r"""
    Overview:
        A variant Net which generates node embedding on original graph 
        and measure the similarity between two paper sequences.

    Interface:
        __init__, forward
    """

    def __init__(self, args, node_embedding_layer, pairRNN):
        super(Net_withGNN, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.use_cuda = args.use_cuda
        self.node_embedding_layer = node_embedding_layer
        self.pairRNN = pairRNN

    def forward(self, leftseqs, rightseqs, graph_data, node_indexmap, train_nodes):
        x, edge_index = graph_data.x, graph_data.edge_index
        if self.use_cuda:
            x = x.cuda()
            edge_index = edge_index.cuda()
        node_embedding = self.node_embedding_layer(x, edge_index)
        vectors = {}
        for node in train_nodes:
            vectors[node] = node_embedding[node_indexmap[node]]
        left_tensor = get_tensor(vectors, leftseqs)
        right_tensor = get_tensor(vectors, rightseqs)
        left_out, right_out, outputs = self.pairRNN(left_tensor, right_tensor)
        return left_out, right_out, outputs


def train_total(args):
    r"""
    Overview:
        a complete train method for complete Net, including testing.

    Arguments:
        - args (:obj:`dict`): args setting
    """
    print(args)
    feature_embedding, neighbours, all_neighbours, trainset, testset = load_data(args)
    dataloader = create_dataloader(trainset, shuffle=True, num_workers=2, batch_size=args.batch_size)
    feature_embedding = torch.FloatTensor(feature_embedding)
    device = torch.device("cuda:" + re.split(r",", args.gpu_id)[0] if args.use_cuda else "cpu")
    gpu_ids = list(map(int, re.split(r",", args.gpu_id)))

    node_embedding_layer = MetaGraphSAGE(args)
    if args.meta_fuse == 'preference_att':
        meta_fuse_layer = metapath_preference_attention(args)
    elif args.meta_fuse == 'product_att':
        meta_fuse_layer = metapath_product_attention(args)
    elif args.meta_fuse == 'product_ave_att':
        meta_fuse_layer = metapath_product_average_attention(args)
    elif args.meta_fuse == 'mlp':
        meta_fuse_layer = MLP_fuse(args)
    pairRNN = PairLSTM(args)
    meta_fuse_layer.apply(weights_init)
    pairRNN.apply(weights_init)
    net = Net(args, node_embedding_layer, meta_fuse_layer, pairRNN)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=1e-8)

    if args.use_cuda:
        net = net.to(device)
    if len(gpu_ids) > 1:
        net = nn.DataParallel(net, device_ids=gpu_ids)
    print(net)
    for name, para in net.named_parameters():
        print(name)

    writer = SummaryWriter('log/{}/{}'.
                           format(args.model_name, time.strftime("%m-%d_%H-%M")))
    results = []
    weight_curve = True
    if weight_curve:
        weight_results = []
        import csv
        if not os.path.exists(Csvdir):
            os.makedirs(Csvdir)
        weight_name = args.model_name + "_attention_weight.csv"
        weight_file = open(os.path.join(Csvdir, weight_name), "w")
        weight_writer = csv.writer(weight_file)
        weight_writer.writerow(["epoch", "PP", "PVP", "PAP", "PTP"])

    for epoch in range(args.epoch_num):
        net.train()
        print("Epoch {}".format(epoch))
        start_time = time.time()
        total_loss = 0
        total_loss1 = 0
        total_loss2 = 0
        true_labels = list()
        pred_labels = list()
        pred_scores = list()
        meta_weight_list = list()
        att_weights = []
        for leftseqs, rightseqs, labels, leftlengths, rightlengths in dataloader:
            train_nodes = get_train_nodes(leftseqs, rightseqs)
            meta_graph_data = []
            meta_node_indexmap = []
            for meta in all_neighbours:
                subgraph_data, node_indexmap = get_1hopsubgraph(feature_embedding, all_neighbours[meta], train_nodes)
                meta_graph_data.append(subgraph_data)
                meta_node_indexmap.append(node_indexmap)
            leftlstm_out, rightlstm_out, outputs, att_weight = net(
                leftseqs, rightseqs, meta_graph_data, meta_node_indexmap, train_nodes)
            att_weights.append(att_weight)
            optimizer.zero_grad()
            cosinelabels = changelabel(labels)
            if args.use_cuda:
                loss1 = criterion1(outputs, labels.cuda())
                loss2 = criterion2(leftlstm_out, rightlstm_out,
                                   cosinelabels.cuda().float())
            else:
                loss1 = criterion1(outputs, labels)
                loss2 = criterion2(
                    leftlstm_out, rightlstm_out, cosinelabels.float())
            loss = loss1 + args.cosineloss_coefficient * loss2
            loss = 100 * loss
            loss.backward()
            optimizer.step()
            total_loss += loss
            total_loss1 += loss1
            total_loss2 += loss2
            if args.use_cuda:
                preds = torch.max(outputs, 1)[1].cpu()
                labels = labels.cpu()
            else:
                preds = torch.max(outputs, 1)[1]
            true_labels += list(labels.int())
            pred_labels += list(preds.data.int())
            pred_scores += list(outputs[:, 1])
            meta_weight_list += list(meta_weights)

        if weight_curve:
            epoch_att_weight = torch.mean(torch.stack(att_weights, dim=0), dim=0)
            weight_tmp = (str(epoch), str(float(epoch_att_weight[0])), str(float(epoch_att_weight[1])),
                          str(float(epoch_att_weight[2])), str(float(epoch_att_weight[3])))
            weight_results.append(weight_tmp)
            weight_writer.writerow(weight_tmp)

        torch.save(net.state_dict(), 'model.pth')
        acc = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)
        auc = roc_auc_score(true_labels, pred_scores)
        print("Train loss: {} - acc: {} P:{} R:{} f1:{} auc:{}"
              .format(total_loss.data.float() / len(trainset), acc, precision, recall, f1, auc))
        print(classification_report(true_labels, pred_labels))
        writer.add_scalar(
            'Train/Loss', total_loss.data.float() / len(trainset), epoch)
        writer.add_scalar('Train/CrossEntropyLoss',
                          total_loss1.data.float() / len(trainset), epoch)
        writer.add_scalar('Train/CosineEmbeddingLoss',
                          total_loss2.data.float() / len(trainset), epoch)
        writer.add_scalar('Train/Accuracy', acc, epoch)
        writer.add_scalar('Train/F1-Score', f1, epoch)
        writer.add_scalar('Train/AUC', auc, epoch)

        with torch.no_grad():
            test_loss, test_loss1, test_loss2, test_acc, test_f1, test_auc, precision, recall = test_total(
                args, feature_embedding, all_neighbours, testset, net)
            writer.add_scalar(
                'Test/Loss', test_loss.data.float() / len(testset), epoch)
            writer.add_scalar('Test/CrossEntropyLoss',
                              test_loss1.data.float() / len(testset), epoch)
            writer.add_scalar('Test/CosineEmbeddingLoss',
                              test_loss2.data.float() / len(testset), epoch)
            writer.add_scalar('Test/Accuracy', test_acc, epoch)
            writer.add_scalar('Test/F1-Score', test_f1, epoch)
            writer.add_scalar('Test/AUC', test_auc, epoch)
            results.append((epoch, test_acc, test_f1,
                            test_auc, precision, recall))
        end_time = time.time()
        print("time: " + str(end_time - start_time))
    writer.close()
    write_data(results, args)

    if weight_curve:
        for i in weight_results:
            tmp = (i[0], i[1], i[2], i[3], i[4])
            weight_writer.writerow(tmp)
        weight_file.close()


def test_total(args, feature_embedding, all_neighbours, testset, net):
    net.eval()
    testloader = create_dataloader(
        testset, shuffle=True, num_workers=2, batch_size=args.batch_size)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CosineEmbeddingLoss()
    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    true_labels = list()
    pred_labels = list()
    pred_scores = list()
    meta_weight_list = list()
    for leftseqs, rightseqs, labels, leftlengths, rightlengths in testloader:
        train_nodes = get_train_nodes(leftseqs, rightseqs)
        meta_graph_data = []
        meta_node_indexmap = []
        for meta in all_neighbours:
            subgraph_data, node_indexmap = get_1hopsubgraph(feature_embedding, all_neighbours[meta], train_nodes)
            meta_graph_data.append(subgraph_data)
            meta_node_indexmap.append(node_indexmap)
        leftlstm_out, rightlstm_out, outputs, _ = net(leftseqs, rightseqs, meta_graph_data, meta_node_indexmap,
                                                      train_nodes)
        cosinelabels = changelabel(labels)
        if args.use_cuda:
            loss1 = criterion1(outputs, labels.cuda())
            loss2 = criterion2(leftlstm_out, rightlstm_out,
                               cosinelabels.cuda().float())
        else:
            loss1 = criterion1(outputs, labels)
            loss2 = criterion2(leftlstm_out, rightlstm_out,
                               cosinelabels.float())
        loss = loss1 + args.cosineloss_coefficient * loss2
        loss = 100 * loss
        total_loss += loss
        total_loss1 += loss1
        total_loss2 += loss2
        if args.use_cuda:
            preds = torch.max(outputs, 1)[1].cpu()
            labels = labels.cpu()
        else:
            preds = torch.max(outputs, 1)[1]
        true_labels += list(labels.int())
        pred_labels += list(preds.data.int())
        pred_scores += list(outputs[:, 1])
    acc = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    auc = roc_auc_score(true_labels, pred_scores)
    print("Test loss: {} - acc: {} P:{} R:{} f1:{} auc:{}"
          .format(total_loss.data.float() / len(testset), acc, precision, recall, f1, auc))
    print(classification_report(true_labels, pred_labels))
    return total_loss, total_loss1, total_loss2, acc, f1, auc, precision, recall