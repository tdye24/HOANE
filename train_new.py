import copy
import time
import torch
import dgl
import numpy as np
import scipy.sparse as sp
from torch import optim
import torch.nn.functional as F
import scipy
from utils import get_args, set_random_seed, load_data_with_labels, \
    get_rec_loss, prepare_inputs
from utils import node_classification_evaluation, load_dataset_dgl, mask_test_edges, preprocess_graph
from dataset import load
from model_new import HOANE_New


def main(args):
    print(f"Using {args.dataset} dataset")
    seeds = args.seeds
    print(f"seeds = {args.seeds}")
    test_acc_over_runs = []
    for i, seed in enumerate(seeds):
        print(f"########## Run {i} for seed {seed} ##########")
        set_random_seed(seed=seed)
        graph, feat, labels, num_class, train_mask, val_mask, test_mask = load('cora')
        # adj, features, labels, train_mask, val_mask, test_mask = load_data_with_labels(dataset_str=args.dataset)
        # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(graph.adj().to_dense().numpy())
        num_nodes, num_features = feat.shape
        adj_train = sp.csr_matrix(graph.adj().to_dense().numpy())
        adj_norm = preprocess_graph(adj_train)
        adj_train = scipy.sparse.coo_matrix((adj_norm[1], (adj_norm[0][:, 0], adj_norm[0][:, 1])),
                                                 shape=adj_norm[2])
        graph = dgl.from_scipy(adj_train)
        graph = graph.to(args.device)
        # 行归一化
        feat_row = feat
        # 不做归一化
        feat = torch.where(feat > 0, torch.ones_like(feat), torch.zeros_like(feat))
        # 按照列做归一化
        # feat = F.normalize(feat, p=1, dim=0)
        feat = feat.to(args.device)
        pos_weight_node = torch.tensor(float(num_nodes * num_nodes - graph.num_edges()) / graph.num_edges()).to(args.device)
        norm_node = num_nodes * num_nodes / float((num_nodes * num_nodes - graph.num_edges()) * 2)

        feat_nonzero = feat.nonzero().shape[0]
        pos_weight_attr = torch.tensor(
            float(num_nodes * num_features - feat_nonzero) / feat_nonzero).to(args.device)
        norm_attr = num_nodes * num_features / float(
            (num_nodes * num_features - feat_nonzero) * 2)

        adj_label = graph.add_self_loop().adj().to_dense().to(args.device)
        labels = labels.to(args.device)

        tolerance = 0  # early stopping
        outer_best_val_acc = -1
        outer_best_epoch = -1
        outer_best_test_acc = -1

        set_random_seed(seed=seed)

        model = HOANE_New(device=args.device)

        model.to(args.device)
        optimizer = optim.Adam(params=model.parameters(), lr=args.pretrain_lr, weight_decay=args.pretrain_wd)
        for epoch in range(1, args.pretrain_epochs + 1):
            if tolerance > 5:
                break
            start_time = time.time()
            warmup = np.min([epoch / 300., 1.])
            model.train()
            optimizer.zero_grad()
            merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
            merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
            reconstruct_node_logits, reconstruct_attr_logits, node_mu_iw_vec, attr_mu_iw_vec = model(graph, feat)

            node_attr_mu = torch.cat((merged_node_mu, merged_attr_mu), 0)
            node_attr_sigma = torch.cat((merged_node_sigma, merged_attr_sigma), 0)
            node_attr_z_samples = torch.cat((merged_node_z_samples, merged_attr_z_samples), 0)
            node_attr_logv_iw = torch.cat((node_logv_iw, attr_logv_iw), 0)

            ker = torch.exp(
                -0.5 * (torch.sum(
                    torch.square(node_attr_z_samples - node_attr_mu) / torch.square(node_attr_sigma + args.eps), 3)))

            log_H_iw_vec = torch.log(torch.mean(ker, 2) + args.eps) - 0.5 * torch.sum(node_attr_logv_iw, 2)
            log_H_iw = torch.mean(log_H_iw_vec, 0)

            adj_orig_tile = adj_label.unsqueeze(-1).repeat(1, 1, args.K)  # adj matrix
            log_lik_iw_node = -1 * get_rec_loss(norm=norm_node,
                                                pos_weight=pos_weight_node,
                                                pred=reconstruct_node_logits,
                                                labels=adj_orig_tile,
                                                loss_type='bce_loss')

            # node_z prior
            node_log_prior_iw_vec = -0.5 * torch.sum(torch.square(node_z_samples_iw), 2)
            node_log_prior_iw = torch.mean(node_log_prior_iw_vec, 0)

            # attr重构loss
            features_tile = feat.unsqueeze(-1).repeat(1, 1, args.K)  # feature matrix
            log_lik_iw_attr = -1 * get_rec_loss(norm=norm_attr,
                                                pos_weight=pos_weight_attr,
                                                pred=reconstruct_attr_logits,
                                                labels=features_tile,
                                                loss_type='bce_loss')

            # attr_z prior
            attr_log_prior_iw_vec = -0.5 * torch.sum(torch.square(attr_z_samples_iw), 2)
            attr_log_prior_iw = torch.mean(attr_log_prior_iw_vec, 0)

            loss = - torch.logsumexp(
                log_lik_iw_node +
                log_lik_iw_attr +
                node_log_prior_iw * warmup / num_nodes +
                attr_log_prior_iw * warmup / num_features -
                log_H_iw * warmup / (num_nodes + num_features), dim=0) + np.log(args.K)
            loss.backward()
            if epoch % args.display_step == 0:
                print(time.time() - start_time)
                print("Epoch:", '%04d' % epoch, "cost_train=", "{:.9f}".format(loss.item()))
            optimizer.step()

            if args.dataset == 'cora':
                threshold = 0
            else:
                threshold = 0

            if epoch > threshold:
                with torch.no_grad():
                    model.eval()
                    merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
                    merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
                    node_mu_iw_vec, attr_mu_iw_vec = model.encode(graph, feat)  # full edges and full attrs

                # test classification performance
                if epoch % args.finetune_interval == 0:
                    # graph_mae_embedding = np.load('./embedding.npy')
                    # node_mu_iw_vec = torch.tensor(graph_mae_embedding).to(args.device)

                    final_test_acc, inner_best_val_acc, inner_best_test_acc = node_classification_evaluation(
                        data=node_mu_iw_vec,
                        labels=labels,
                        train_mask=train_mask,
                        val_mask=val_mask,
                        test_mask=test_mask,
                        args=args)
                    print(f"Pretrain epoch {epoch}")
                    # print(
                    #     f"[Inner] --- Best ValAcc: {inner_best_val_acc:.4f}, ",
                    #     f"Final TestAcc: {final_test_acc:.4f}, ",
                    #     f"Best TestAcc: {inner_best_test_acc:.4f} --- ")
                    if inner_best_val_acc >= outer_best_val_acc:
                        tolerance = 0
                        outer_best_val_acc = inner_best_val_acc
                        outer_best_epoch = epoch
                        outer_best_test_acc = inner_best_test_acc
                    else:
                        tolerance += 1
                    print(
                        f"[Outer] --- Best ValAcc: {outer_best_val_acc:.4f} in epoch {outer_best_epoch}, ",
                        f"Best TestAcc: {outer_best_test_acc:.4f} --- ")

        print("Node classification, val_acc", '{:.5f}'.format(outer_best_val_acc), "test_acc",
              '{:.5f}'.format(outer_best_test_acc))

        test_acc_over_runs.append(outer_best_test_acc)
    print("Node classification, test accuracy", test_acc_over_runs)
    print("Node classification, test accuracy: {:.5f}".format(np.mean(test_acc_over_runs)), "+-",
          "{:.5f}".format(np.std(test_acc_over_runs)))
    with open('./normalized-result.txt', 'a') as f:
        f.write(f"pretrain_lr {args.pretrain_lr}, finetune_lr {args.finetune_lr}, pretrain_wd {args.pretrain_wd}, dropout {args.dropout}\n")
        f.write(f"Node classification, test accuracy: {np.mean(test_acc_over_runs):.4f}/{np.std(test_acc_over_runs):.4f}\n")


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
