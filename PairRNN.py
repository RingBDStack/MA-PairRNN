import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(torch.nn.Module):
    r"""
    Overview:
        A simple MLP implementation for use of PairLSTM.

    Interface:
        __init__, forward
    """

    def __init__(self, n_feature, n_hidden, n_output, dropout=0.5):
        super(SimpleMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.activate = nn.ReLU(inplace=True)
        self.hidden_1 = nn.Linear(n_feature, n_hidden)
        self.hidden_2 = torch.nn.Linear(n_hidden, n_hidden//8)
        self.out = torch.nn.Linear(n_hidden//8, n_output)

    def forward(self, x):
        x = self.activate(self.hidden_1(x))
        x = self.activate(self.hidden_2(x))
        x = self.out(x)
        return x


class PairLSTM(nn.Module):
    r"""
    Overview:
        A similarity measure layer using Siamese LSTM to predict whether the pair of papers belong to one author.

    Interface:
        __init__, forward
    """

    def __init__(self, args):
        super(PairLSTM, self).__init__()
        # self.input_dim = args.RNN_input_dim
        self.input_dim = args.embedding_dim
        self.hidden_dim = args.RNN_hidden_dim
        self.layer_num = args.RNN_layer_num
        self.label_size = args.label_size
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.left_lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=self.layer_num, batch_first=True)
        self.right_lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=self.layer_num, batch_first=True)
        self.classifier = SimpleMLP(n_feature=self.hidden_dim*2, n_hidden=self.hidden_dim, n_output=self.label_size)

    def forward(self, left_input, right_input):
        if not hasattr(self, '_flattened'):
            self.left_lstm.flatten_parameters()
            self.right_lstm.flatten_parameters()
            setattr(self, '_flattened', True)
        leftlstm_out, _ = self.left_lstm(left_input)
        leftlstm_out = torch.mean(leftlstm_out, 1)
        rightlstm_out, _ = self.right_lstm(right_input)
        rightlstm_out = torch.mean(rightlstm_out, 1)
        concat_embedding = torch.cat((leftlstm_out, rightlstm_out), 1)
        outputs = self.classifier(concat_embedding)
        return leftlstm_out, rightlstm_out, outputs
