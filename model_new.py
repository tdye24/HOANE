import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from layers_new import GAT, GCN, DenseModel


def sample_n(mu, sigma):
    eps = torch.randn_like(sigma)
    z = mu + eps * sigma
    return z

class Decoder(nn.Module):
    """GAT decoder layer for link prediction and attribute inference."""

    def __init__(self, dropout=0., act=lambda x: x):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.act = act
        self.decoder = GAT(in_dim=256, num_hidden=512, out_dim=1433, num_layers=1, nhead=4, nhead_out=1, activation='prelu',
                feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False, norm=None, concat_out=True, encoding=False)

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
        weights = F.normalize(x, p=1, dim=1)  # torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12, out=None)
        fine_grained_features = torch.matmul(weights, z_a)
        concat_features = torch.cat((z_u, fine_grained_features), dim=1)
        outputs_a = self.decoder(adj, concat_features)
        outputs_a = self.act(outputs_a)
        return outputs_u, outputs_a


class HOANE_New(nn.Module):
    """
    Node Encoder: GAT
    Attr Encoder: MLP
    Node Decoder: Inner Product
    Attr Decoder:
    """

    def __init__(self, device, num_nodes=2708, input_dim=1433, num_hidden=128, out_dim=128, noise_dim=5,
                 K=1, J=1, dropout=0.):
        super(HOANE_New, self).__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout

        # self.node_mu_nn = GAT(in_dim=input_dim+noise_dim, num_hidden=num_hidden, out_dim=out_dim, num_layers=2, nhead=4, nhead_out=4,
        #                       activation='prelu', feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False, norm=None, concat_out=True, encoding=True)
        # self.node_var_nn = GAT(in_dim=input_dim, num_hidden=num_hidden, out_dim=out_dim, num_layers=2, nhead=4, nhead_out=4, activation='prelu',
        #         feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False, norm=None, concat_out=True, encoding=True)

        self.node_mu_nn = GCN(in_dim=input_dim+noise_dim, hid_dim=num_hidden, out_dim=out_dim, n_layers=2)
        self.node_var_nn = GCN(in_dim=input_dim, hid_dim=num_hidden, out_dim=out_dim, n_layers=2)

        self.attr_mu_nn = DenseModel(in_dim=num_nodes+noise_dim, num_hidden=128, out_dim=128, num_layers=2, dropout=0.0)
        self.attr_var_nn = DenseModel(in_dim=num_nodes, num_hidden=128, out_dim=128, num_layers=2, dropout=0.0)

        self.decoder = Decoder(dropout=0.0)

        self.noise_dim = noise_dim
        self.noise_dist = dist.Bernoulli(torch.tensor([.5], device=device))

        self.K = K
        self.J = J


    def encode(self, graph, x):
        attr_dim = x.shape[1]
        num_nodes = x.shape[0]
        node_mu_input = x.unsqueeze(1).repeat(1, self.K + self.J, 1)
        node_noise_e = self.noise_dist.sample(torch.Size([num_nodes, self.K + self.J, self.noise_dim]))
        node_noise_e = torch.squeeze(node_noise_e)
        noised_node_mu_input = torch.concat((node_noise_e, node_mu_input), 2)
        node_mu = self.node_mu_nn(graph, noised_node_mu_input)
        node_mu_iw = node_mu[:, :self.K, :]
        node_mu_star = node_mu[:, self.K:, :]
        node_mu_iw_vec = torch.mean(node_mu_iw, 1)

        node_logv = self.node_var_nn(graph, x)
        node_logv_iw = node_logv.unsqueeze(1).repeat(1, self.K, 1)
        node_sigma_iw1 = torch.exp(0.5 * node_logv_iw)
        merged_node_sigma = node_sigma_iw1.unsqueeze(2).repeat(1, 1, self.J + 1, 1)

        node_z_samples_iw = sample_n(mu=node_mu_iw, sigma=node_sigma_iw1)
        merged_node_z_samples = node_z_samples_iw.unsqueeze(2).repeat(1, 1, self.J + 1, 1)
        node_mu_star1 = node_mu_star.unsqueeze(1).repeat(1, self.K, 1, 1)
        merged_node_mu = torch.concat((node_mu_star1, node_mu_iw.unsqueeze(2)), 2)

        attr_mu_input = x.transpose(0, 1).unsqueeze(1).repeat(1, self.K + self.J, 1)
        attr_noise_e = self.noise_dist.sample(torch.Size([attr_dim, self.K + self.J, self.noise_dim]))
        attr_noise_e = torch.squeeze(attr_noise_e)
        noised_attr_mu_input = torch.concat((attr_noise_e, attr_mu_input), 2)
        attr_mu = self.attr_mu_nn(noised_attr_mu_input)
        attr_mu_iw = attr_mu[:, :self.K, :]
        attr_mu_star = attr_mu[:, self.K:, :]
        attr_mu_iw_vec = torch.mean(attr_mu_iw, 1)

        attr_logv = self.attr_var_nn(x=x.transpose(0, 1))
        attr_logv_iw = attr_logv.unsqueeze(1).repeat(1, self.K, 1)
        attr_sigma_iw1 = torch.exp(0.5 * attr_logv_iw)
        merged_attr_sigma = attr_sigma_iw1.unsqueeze(2).repeat(1, 1, self.J + 1, 1)


        attr_z_samples_iw = sample_n(mu=attr_mu_iw, sigma=attr_sigma_iw1)
        merged_attr_z_samples = attr_z_samples_iw.unsqueeze(2).repeat(1, 1, self.J + 1, 1)
        attr_mu_star1 = attr_mu_star.unsqueeze(1).repeat(1, self.K, 1, 1)
        merged_attr_mu = torch.concat((attr_mu_star1, attr_mu_iw.unsqueeze(2)), 2)

        return merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
               merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
               node_mu_iw_vec, attr_mu_iw_vec

    def decode(self, node_z, attr_z, graph, x):
        outputs_u, outputs_a = None, None
        for i in range(self.K):
            input_u = node_z[:, i, :].squeeze()
            input_a = attr_z[:, i, :].squeeze()

            logits_node, logits_attr = self.decoder(z_u=input_u, z_a=input_a, adj=graph, x=x)

            if i == 0:
                outputs_u = logits_node.unsqueeze(2)
                outputs_a = logits_attr.unsqueeze(2)
            else:
                outputs_u = torch.cat((outputs_u, logits_node.unsqueeze(2)), 2)
                outputs_a = torch.cat((outputs_a, logits_attr.unsqueeze(2)), 2)
        return outputs_u, outputs_a

    def forward(self, graph, x):
        merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
        merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
        node_mu_iw_vec, attr_mu_iw_vec = self.encode(graph, x)

        reconstruct_node_logits, reconstruct_attr_logits = self.decode(graph=graph,
                                                                       x=x,
                                                                       node_z=node_z_samples_iw,
                                                                       attr_z=attr_z_samples_iw)

        return merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
               merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
               reconstruct_node_logits, reconstruct_attr_logits, node_mu_iw_vec, attr_mu_iw_vec

if __name__ == '__main__':
    from dataset import load
    device = torch.device("cpu")
    graph, feat, labels, num_class, train_idx, val_idx, test_idx = load('cora')
    graph = graph.to(device)
    feat = feat.to(device)
    model = HOANE_New(device=device).to(device)
    print(model)
    # res = model(graph, feat)
    res = model(graph, feat)
    print(res)
