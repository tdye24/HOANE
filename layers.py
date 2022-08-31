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
        support = torch.stack(
            [torch.mm(self.dropout_layer(inp), self.weight) for inp in torch.unbind(inputs, dim=1)],
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
        self.dropout_layer = nn.Dropout(p=dropout)
        for i in range(len(output_dims)):
            if i == 0:
                self.fcs.append(nn.Linear(in_features=input_dim, out_features=output_dims[0]))
            else:
                self.fcs.append(nn.Linear(in_features=output_dims[i - 1], out_features=output_dims[i]))
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, x):
        for i in range(len(self.fcs)):
            # todo(tdye): DenseModel dropout
            x = self.dropout_layer(x)
            x = self.fcs[i](x)
            x = self.act(x)
        return x


class InnerProduct_Decoder(nn.Module):
    """Decoder model layer for link prediction and attribute inference."""

    def __init__(self, dropout=0., act=torch.sigmoid):
        super(InnerProduct_Decoder, self).__init__()
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


class SparseDecoder(nn.Module):
    """Decoder model layer for link prediction."""

    def __init__(self, z_dim, dropout=0.):
        super(SparseDecoder, self).__init__()
        self.dropout = dropout
        self.rk_u = Parameter(torch.FloatTensor(z_dim))
        self.rk_a = Parameter(torch.FloatTensor(z_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.rk_u, a=-6., b=0.)
        torch.nn.init.uniform_(self.rk_a, a=-6., b=0.)

    def forward(self, z_u, z_a):
        rk_u = F.sigmoid(self.rk_u)
        z_u = F.dropout(z_u, self.dropout, self.training)
        z_u_t = z_u.transpose(0, 1)
        z_u_t = torch.matmul(torch.diag(rk_u), z_u_t)
        x = torch.matmul(z_u, z_u_t)
        # outputs_u = 1 - torch.exp(- torch.exp(x))
        outputs_u = x

        rk_a = F.sigmoid(self.rk_a)
        z_a = F.dropout(z_a, self.dropout, self.training)
        z_a_t = z_a.transpose(0, 1)
        z_a_t = torch.matmul(torch.diag(rk_a), z_a_t)
        y = torch.matmul(z_u, z_a_t)
        # outputs_a = 1 - torch.exp(- torch.exp(y))
        outputs_a = y

        return outputs_u, outputs_a


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, alpha, heads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.heads = heads
        self.gat_layers = []
        if num_layers == 1:
            # 一层GAT
            self.out_att = GraphAttentionLayer(in_features=input_dim,
                                               out_features=output_dim,
                                               dropout=dropout,
                                               alpha=alpha,
                                               concat=False)
        else:
            # input projection (no residual)
            self.gat_layers.extend(
                [GraphAttentionLayer(in_features=input_dim,
                                     out_features=hidden_dim,
                                     dropout=dropout,
                                     alpha=alpha,
                                     concat=True) for _ in range(heads)]
            )
            for i in range(1, num_layers - 1):
                self.gat_layers.extend(
                    [GraphAttentionLayer(in_features=hidden_dim * heads,
                                         out_features=hidden_dim,
                                         dropout=dropout,
                                         alpha=alpha,
                                         concat=True) for _ in range(heads)]
                )
            self.gat_layers = nn.ModuleList(self.gat_layers)
            self.out_att = GraphAttentionLayer(in_features=hidden_dim * heads,
                                               out_features=output_dim,
                                               dropout=dropout,
                                               alpha=alpha,
                                               concat=False)

    def forward(self, x, adj):
        for i in range(0, self.num_layers - 1):
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.gat_layers[i * self.heads: (i + 1) * self.heads]], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.out_att(x, adj)
        # x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)


class GraphAttentionLayerK(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayerK, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inputs, adj):
        outputs = []
        for h in torch.unbind(inputs, dim=1):
            Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
            e = self._prepare_attentional_mechanism_input(Wh)

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, Wh)

            if self.concat:
                outputs.append(F.elu(h_prime))
            else:
                outputs.append(h_prime)
        outputs = torch.stack(outputs, dim=1)
        return outputs

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GATK(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, alpha, heads):
        """Dense version of GAT."""
        super(GATK, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.heads = heads
        self.gatk_layers = []
        if num_layers == 1:
            # 一层GAT
            self.out_att = GraphAttentionLayerK(in_features=input_dim,
                                                out_features=output_dim,
                                                dropout=dropout,
                                                alpha=alpha,
                                                concat=False)
        else:
            # input projection (no residual)
            self.gatk_layers.extend(
                [GraphAttentionLayerK(in_features=input_dim,
                                      out_features=hidden_dim,
                                      dropout=dropout,
                                      alpha=alpha,
                                      concat=True) for _ in range(heads)]
            )
            for i in range(1, num_layers - 1):
                self.gatk_layers.extend(
                    [GraphAttentionLayerK(in_features=hidden_dim * heads,
                                          out_features=hidden_dim,
                                          dropout=dropout,
                                          alpha=alpha,
                                          concat=True) for _ in range(heads)]
                )
            self.gat_layers = nn.ModuleList(self.gatk_layers)
            self.out_att = GraphAttentionLayerK(in_features=hidden_dim * heads,
                                                out_features=output_dim,
                                                dropout=dropout,
                                                alpha=alpha,
                                                concat=False)

    def forward(self, x, adj):
        for i in range(0, self.num_layers - 1):
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.gatk_layers[i * self.heads: (i + 1) * self.heads]], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.out_att(x, adj)
        # x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)


class GAT_Decoder(nn.Module):
    """GAT decoder layer for link prediction and attribute inference."""

    def __init__(self, dropout=0., act=torch.sigmoid, num_layers=1):
        super(GAT_Decoder, self).__init__()
        self.dropout = dropout
        self.act = act  # self.dropout_layer = nn.Dropout(p=dropout)
        self.decoder = GAT(input_dim=1024,
                           hidden_dim=512,
                           output_dim=1433,
                           num_layers=num_layers,
                           dropout=0.,
                           alpha=0.1,
                           heads=4)

    def forward(self, x, adj, z_u, z_a):
        """

        :param x: 2708 x 1433
        :param adj: 2708 x 2708
        :param z_u: 2708 x 128
        :param z_a: 1433 x 128
        :return:
        """
        z_u = F.dropout(z_u, self.dropout, self.training)
        z_u_t = z_u.transpose(0, 1)
        links = torch.matmul(z_u, z_u_t)
        outputs_u = self.act(links)

        z_a = F.dropout(z_a, self.dropout, self.training)
        # z_a_t = z_a.transpose(0, 1)
        # weights = F.softmax(torch.matmul(z_u, z_a_t), dim=1)
        # weights = F.normalize(x + 1e-8, p=1, dim=1)
        weights = F.normalize(x, p=1, dim=1)  # torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12, out=None)
        fine_grained_features = torch.matmul(weights, z_a)
        concat_features = torch.cat((z_u, fine_grained_features), dim=1)
        # concat_features = z_u + fine_grained_features
        outputs_a = self.decoder(concat_features, adj)
        outputs_a = self.act(outputs_a)
        return outputs_u, outputs_a


class MLP_Decoder(nn.Module):
    """MLP decoder layer for link prediction and attribute inference."""

    def __init__(self, dropout=0., act=torch.sigmoid):
        super(MLP_Decoder, self).__init__()
        self.dropout = dropout
        self.act = act  # self.dropout_layer = nn.Dropout(p=dropout)
        self.decoder = nn.Linear(in_features=1024, out_features=1433)

    def forward(self, x, z_u, z_a):
        """

        :param x: 2708 x 1433
        :param z_u: 2708 x 128
        :param z_a: 1433 x 128
        :return:
        """
        z_u = F.dropout(z_u, self.dropout, self.training)
        z_u_t = z_u.transpose(0, 1)
        links = torch.matmul(z_u, z_u_t)
        outputs_u = self.act(links)

        z_a = F.dropout(z_a, self.dropout, self.training)
        # z_a_t = z_a.transpose(0, 1)
        # weights = F.softmax(torch.matmul(z_u, z_a_t), dim=1)
        # weights = F.normalize(x + 1e-8, p=1, dim=1)
        weights = F.normalize(x, p=1, dim=1)  # torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12, out=None)
        fine_grained_features = torch.matmul(weights, z_a)
        concat_features = torch.cat((z_u, fine_grained_features), dim=1)
        # concat_features = z_u + fine_grained_features
        outputs_a = self.decoder(concat_features)
        outputs_a = self.act(outputs_a)
        return outputs_u, outputs_a


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, x):
        logits = self.linear(x)
        return logits


# GAT 层数， dropout， concat


if __name__ == '__main__':
    # model = GraphAttentionLayerK(in_features=1433, out_features=512, dropout=0., alpha=0.1, concat=True)
    # x = torch.ones((2708, 3, 1433))
    # adj = torch.ones((2708, 2708))
    # for name, param in model.named_parameters():
    #     print(name, param.data.shape)
    # out = model(x, adj)
    # print(out.shape)

    model = GATK(input_dim=1433, hidden_dim=512, output_dim=512, num_layers=2, dropout=0., alpha=0.3, heads=4)
    x = torch.ones((2708, 3, 1433))
    adj = torch.ones((2708, 2708))
    for name, param in model.named_parameters():
        print(name, param.data.shape)
    out = model(x, adj)
    print(out.shape)
