import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from layers import GraphConvolution, GCNNModel, GraphConvolutionK, GCNNModelK, DenseModel, \
    InnerProductDecoderNodeAttribute


def sample_n(mu, sigma):
    eps = torch.randn_like(sigma)
    z = mu + eps * sigma
    return z


class HOANE(nn.Module):
    """Encode and decode."""

    def __init__(self,
                 num_nodes=2708,
                 input_dim=1433,
                 output_dim=128,
                 node_base_hidden=None,
                 node_mu_hidden=None,
                 node_var_hidden=None,
                 attr_base_hidden=None,
                 attr_mu_hidden=None,
                 attr_var_hidden=None,
                 dropout=0.,
                 K=1,
                 J=1,
                 noise_distribution="bernoulli",
                 node_noise_dim=5,
                 attr_noise_dim=5,
                 device="cpu"):
        super(HOANE, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.K = K
        self.J = J
        if noise_distribution == "bernoulli":
            self.noise_dist = dist.Bernoulli(torch.tensor([.5], device=device))
        self.node_noise_dim = node_noise_dim
        self.attr_noise_dim = attr_noise_dim
        if node_base_hidden is None:
            node_base_hidden = [1433]
        if node_mu_hidden is None:
            node_mu_hidden = [128]
        if node_var_hidden is None:
            node_var_hidden = [128]
        if attr_base_hidden is None:
            attr_base_hidden = [2708]
        if attr_mu_hidden is None:
            attr_mu_hidden = [128]
        if attr_var_hidden is None:
            attr_var_hidden = [128]

        # 节点
        # todo(tdye): 中间加了一个hidden，会不会导致层数太深，over-smoothing
        # self.node_base_gcn = GCNNModel(input_dim=input_dim, output_dims=node_base_hidden, dropout=dropout, act=F.relu)

        # 输入维度包含噪声维度
        self.node_mu_gcnk = GCNNModelK(input_dim=node_base_hidden[-1] + node_noise_dim, output_dims=node_mu_hidden,
                                       dropout=dropout, act=F.relu)
        self.node_mu_gck = GraphConvolutionK(input_dim=node_mu_hidden[-1], output_dim=output_dim, dropout=dropout,
                                             act=lambda x: x)

        self.node_var_gcn = GCNNModel(input_dim=node_base_hidden[-1], output_dims=node_var_hidden, dropout=dropout,
                                      act=F.relu)
        self.node_var_gc = GraphConvolution(input_dim=node_var_hidden[-1], output_dim=output_dim, dropout=dropout,
                                            act=lambda x: x)

        # 属性
        # self.attr_base_mlp = DenseModel(input_dim=num_nodes, output_dims=attr_base_hidden, act=torch.tanh)

        self.attr_mu_mlpk = DenseModel(input_dim=attr_base_hidden[-1] + attr_noise_dim, output_dims=attr_mu_hidden,
                                       act=torch.tanh)
        self.attr_mu_fck = nn.Linear(in_features=attr_mu_hidden[-1], out_features=output_dim)

        self.attr_var_mlp = DenseModel(input_dim=attr_base_hidden[-1], output_dims=attr_var_hidden,
                                       act=torch.tanh)
        self.attr_var_fc = nn.Linear(in_features=attr_var_hidden[-1], out_features=output_dim)

        self.decoder = InnerProductDecoderNodeAttribute(act=lambda x: x)

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'attr' in name:
                # print(name)
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    assert 'bias' in name
                    torch.nn.init.zeros_(param)

    def encode(self, x, adj):
        # todo(tdye): K个样本方差都一样
        # node_hidden = self.node_base_gcn(x=x, adj=adj)
        # 节点均值
        # 采样K+J个均值
        # node_mu_input = node_hidden.unsqueeze(1).repeat(1, self.K + self.J, 1)
        node_mu_input = x.unsqueeze(1).repeat(1, self.K + self.J, 1)
        node_noise_e = self.noise_dist.sample(torch.Size([self.num_nodes, self.K + self.J, self.node_noise_dim]))
        node_noise_e = torch.squeeze(node_noise_e)
        noised_node_mu_input = torch.concat((node_noise_e, node_mu_input), 2)
        h = self.node_mu_gcnk(x=noised_node_mu_input, adj=adj)
        node_mu = self.node_mu_gck(inputs=h, adj=adj)
        # node_mu = self.node_mu_gcnk(x=noised_node_mu_input, adj=adj)
        node_mu_iw = node_mu[:, :self.K, :]  # K个均值，第一个期望，importance weights
        node_mu_star = node_mu[:, self.K:, :]  # J个均值（\psi^star），第二个期望，防止degeneracy
        node_mu_iw_vec = torch.mean(node_mu_iw, 1)  # todo(tdye): 为什么不用node_mu_iw？用于classification，调整K的个数

        # 节点方差
        node_logv_input = self.node_var_gcn(x=x, adj=adj)
        node_logv = self.node_var_gc(x=node_logv_input, adj=adj)
        # node_logv = self.node_var_gcn(x=node_hidden, adj=adj)
        # node_logv = self.node_var_gc(x=node_logv_input, adj=adj)
        node_logv_iw = node_logv.unsqueeze(1).repeat(1, self.K, 1)
        node_sigma_iw1 = torch.exp(0.5 * node_logv_iw)  # K个方差，第一个期望，importance weights
        merged_node_sigma = node_sigma_iw1.unsqueeze(2).repeat(1, 1, self.J + 1, 1)

        # 采样节点隐变量z
        # print(f"node mu shape {node_mu_iw.shape}, node sigma shape {node_sigma_iw1.shape}")
        node_z_samples_iw = sample_n(mu=node_mu_iw, sigma=node_sigma_iw1)  # 第一个期望，得到K个均值和方差，采样z

        # 针对每一个z，copy J+1 份，用于计算分母部分\tilde{h}_J(z)
        merged_node_z_samples = node_z_samples_iw.unsqueeze(2).repeat(1, 1, self.J + 1, 1)
        # print(f"merged_node_z_samples.shape = {merged_node_z_samples.shape}")

        # 第二个期望，采样J个\psi_star（对应上面的node_mu_star），因为K个z共享J个\spi_star，所以这里要复制K份
        node_mu_star1 = node_mu_star.unsqueeze(1).repeat(1, self.K, 1, 1)
        # node_mu_iw中的每一个\psi要和J个\psi_star进行merge（concat），方便计算分母下的\tilde{h}_J(z)
        # 先对node_mu_iw（shape N*K*embedding_dim）扩充第二个维度，变成N*K*1*embedding_dim，
        # 再与node_mu_star（shape N*K*J*embedding_dim）concat在一起，output出的结果维度为N*K*J+1*embedding_dim
        merged_node_mu = torch.concat((node_mu_star1, node_mu_iw.unsqueeze(2)), 2)
        # print(f"merged_node_mu.shape = {merged_node_mu.shape}")
        # 解释：N个样本，没个样本采样K个\psi，J+1个用于计算分母部分

        # attr_hidden = self.attr_base_mlp(x.transpose(0, 1))
        # 属性均值
        # 采样K+J个均值
        attr_mu_input = x.transpose(0, 1).unsqueeze(1).repeat(1, self.K + self.J, 1)
        attr_noise_e = self.noise_dist.sample(torch.Size([self.input_dim, self.K + self.J, self.attr_noise_dim]))
        attr_noise_e = torch.squeeze(attr_noise_e)
        noised_attr_mu_input = torch.concat((attr_noise_e, attr_mu_input), 2)
        h = self.attr_mu_mlpk(noised_attr_mu_input)
        attr_mu = self.attr_mu_fck(h)
        # attr_mu = self.attr_mu_mlpk(noised_attr_mu_input)
        attr_mu_iw = attr_mu[:, :self.K, :]  # K个均值，第一个期望，importance weights
        attr_mu_star = attr_mu[:, self.K:, :]  # J个均值（\psi^star），第二个期望，防止degeneracy
        attr_mu_iw_vec = torch.mean(attr_mu_iw, 1)  # todo(tdye): 为什么不用attr_mu_iw？用于classification，调整K的个数

        # 属性方差
        attr_logv_input = self.attr_var_mlp(x=x.transpose(0, 1))
        attr_logv = self.attr_var_fc(attr_logv_input)
        # attr_logv = self.attr_var_mlp(x=attr_hidden)
        # attr_logv = self.attr_var_fc(attr_logv_input)
        attr_logv_iw = attr_logv.unsqueeze(1).repeat(1, self.K, 1)
        attr_sigma_iw1 = torch.exp(0.5 * attr_logv_iw)  # K个方差，第一个期望，importance weights
        merged_attr_sigma = attr_sigma_iw1.unsqueeze(2).repeat(1, 1, self.J + 1, 1)
        # 采样节点隐变量z
        # print(f"attr mu shape {attr_mu_iw.shape}, attr sigma shape {attr_sigma_iw1.shape}")
        attr_z_samples_iw = sample_n(mu=attr_mu_iw, sigma=attr_sigma_iw1)  # 第一个期望，得到K个均值和方差，采样z

        # 针对每一个z，copy J+1 份，用于计算分母部分\tilde{h}_J(z)
        merged_attr_z_samples = attr_z_samples_iw.unsqueeze(2).repeat(1, 1, self.J + 1, 1)
        # print(f"merged_attr_z_samples.shape = {merged_attr_z_samples.shape}")

        # 第二个期望，采样J个\psi_star（对应上面的node_mu_star），因为K个z共享J个\spi_star，所以这里要复制K份
        attr_mu_star1 = attr_mu_star.unsqueeze(1).repeat(1, self.K, 1, 1)
        # node_mu_iw中的每一个\psi要和J个\psi_star进行merge（concat），方便计算分母下的\tilde{h}_J(z)
        # 先对node_mu_iw（shape N*K*embedding_dim）扩充第二个维度，变成N*K*1*embedding_dim，
        # 再与node_mu_star（shape N*K*J*embedding_dim）concat在一起，output出的结果维度为N*K*J+1*embedding_dim
        merged_attr_mu = torch.concat((attr_mu_star1, attr_mu_iw.unsqueeze(2)), 2)
        # print(f"merged_attr_mu.shape = {merged_attr_mu.shape}")
        # 解释：N个样本，没个样本采样K个\psi，J+1个用于计算分母部分

        return merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
               merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
               node_mu_iw_vec, attr_mu_iw_vec  # node_mu_iw_vec, attr_mu_iw_vec 用于link prediction和attribute inference

    def decode(self, node_z, attr_z):
        for i in range(self.K):
            input_u = node_z[:, i, :].squeeze()
            input_a = attr_z[:, i, :].squeeze()
            logits_node, logits_attr = self.decoder(z_u=input_u, z_a=input_a)
            if i == 0:
                outputs_u = logits_node.unsqueeze(2)
                outputs_a = logits_attr.unsqueeze(2)
            else:
                outputs_u = torch.cat((outputs_u, logits_node.unsqueeze(2)), 2)
                outputs_a = torch.cat((outputs_a, logits_attr.unsqueeze(2)), 2)
        return outputs_u, outputs_a

    def forward(self, x, adj):
        merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
        merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
        node_mu_iw_vec, attr_mu_iw_vec = self.encode(x=x, adj=adj)

        # 重构
        reconstruct_node_logits, reconstruct_attr_logits = self.decode(node_z=node_z_samples_iw,
                                                                       attr_z=attr_z_samples_iw)

        return merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
               merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
               reconstruct_node_logits, reconstruct_attr_logits, node_mu_iw_vec, attr_mu_iw_vec