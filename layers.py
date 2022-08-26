import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_layer = nn.Dropout(p=dropout)
        self.act = act
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)
        init_range = np.sqrt(6.0 / (self.input_dim + self.output_dim))
        torch.nn.init.uniform_(self.weight, a=-init_range, b=init_range)

    def forward(self, x, adj):
        x = self.dropout_layer(x)
        x = torch.mm(x, self.weight)
        x = torch.spmm(adj, x)
        outputs = self.act(x)
        return outputs

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


class GCNNModel(nn.Module):
    """Stack of graph convolutional layers."""

    def __init__(self, input_dim, output_dims, dropout=0., act=F.relu):
        super(GCNNModel, self).__init__()
        self.gcns = []
        for i in range(len(output_dims)):
            if i == 0:
                self.gcns.append(GraphConvolution(input_dim=input_dim,
                                                  output_dim=output_dims[0],
                                                  dropout=dropout,
                                                  act=act))
            else:
                self.gcns.append(GraphConvolution(input_dim=output_dims[i - 1],
                                                  output_dim=output_dims[i],
                                                  dropout=dropout,
                                                  act=act))
        self.gcns = nn.ModuleList(self.gcns)

    def forward(self, x, adj):
        for i in range(len(self.gcns)):
            x = self.gcns[i](x=x, adj=adj)
        return x


class GraphConvolutionK(nn.Module):
    """Generate K ..."""
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, dropout=0., act=F.relu):
        super(GraphConvolutionK, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_layer = nn.Dropout(p=dropout)
        self.act = act
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)
        init_range = np.sqrt(6.0 / (self.input_dim + self.output_dim))
        torch.nn.init.uniform_(self.weight, a=-init_range, b=init_range)

    def forward(self, inputs, adj):
        # K = inputs.shape[1]
        # for i in range(K):
        #     x = inputs[:, i, :].squeeze()
        #     x = self.dropout_layer(x)
        #     x = torch.mm(x, self.weight)
        #     x = torch.mm(adj, x)
        #
        #     if i == 0:
        #         outputs = self.act(x).unsqueeze(1)
        #     else:
        #         outputs = torch.cat((outputs, self.act(x).unsqueeze(1)), 1)
        support = torch.stack(
            [torch.mm(inp, self.weight) for inp in torch.unbind(inputs, dim=1)],
            dim=1)
        outputs = torch.stack(
            [torch.spmm(adj, sup) for sup in torch.unbind(support, dim=1)],
            dim=1)
        outputs = self.act(outputs)
        return outputs

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


class GCNNModelK(nn.Module):
    """Stack of graph convolutional layers."""

    def __init__(self, input_dim, output_dims, dropout=0., act=F.relu):
        super(GCNNModelK, self).__init__()
        self.gcnks = []
        for i in range(len(output_dims)):
            if i == 0:
                self.gcnks.append(GraphConvolutionK(input_dim=input_dim,
                                                    output_dim=output_dims[0],
                                                    dropout=dropout,
                                                    act=act))
            else:
                self.gcnks.append(GraphConvolutionK(input_dim=output_dims[i - 1],
                                                    output_dim=output_dims[i],
                                                    dropout=dropout,
                                                    act=act))
        self.gcnks = nn.ModuleList(self.gcnks)

    def forward(self, x, adj):
        for i in range(len(self.gcnks)):
            x = self.gcnks[i](inputs=x, adj=adj)
        return x


class DenseModel(nn.Module):
    """Stack of fully connected layers."""

    def __init__(self, input_dim, output_dims, dropout=0., act=F.tanh):
        super(DenseModel, self).__init__()
        self.act = act
        self.fcs = []
        for i in range(len(output_dims)):
            if i == 0:
                self.fcs.append(nn.Linear(in_features=input_dim, out_features=output_dims[0]))
            else:
                self.fcs.append(nn.Linear(in_features=output_dims[i - 1], out_features=output_dims[i]))
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, x):
        for i in range(len(self.fcs)):
            x = self.fcs[i](x)
            x = self.act(x)
        return x


class InnerProductDecoderNodeAttribute(nn.Module):
    """Decoder model layer for link prediction and attribute inference."""

    def __init__(self, dropout=0., act=torch.sigmoid):
        super(InnerProductDecoderNodeAttribute, self).__init__()
        self.dropout = dropout
        self.act = act  # self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, z_u, z_a):
        z_u = F.dropout(z_u, self.dropout, self.training)
        z_u_t = z_u.transpose(0, 1)
        links = torch.matmul(z_u, z_u_t)
        outputs_u = self.act(links)
        z_a = F.dropout(z_a, self.dropout, self.training)
        z_a_t = z_a.transpose(0, 1)
        attributes = torch.matmul(z_u, z_a_t)
        outputs_a = self.act(attributes)
        return outputs_u, outputs_a


if __name__ == '__main__':
    model = GCNNModel(1433, [32, 16], dropout=0.)
    for name, param in model.named_parameters():
        print(name, param.data.shape)
