import time
import torch
import numpy as np
import scipy.sparse as sp
from torch import optim
import torch.nn.functional as F
from utils import get_args, set_random_seed, load_data_with_labels, \
    get_rec_loss, prepare_inputs
from utils import adj_augment, attr_augment, node_classification_evaluation, load_dataset_dgl
from model import HOANE, HOANE_V2

# Model hyper-parameters
noise_dim_u = 5
noise_dim_a = 5
z_dim = 512
hidden_u = [128]
hidden_a = [128]
hidden_u_v = [128]
hidden_a_v = [128]


def main(args):
    print(f"Using {args.dataset} dataset")
    seeds = args.seeds
    print(f"seeds = {args.seeds}")
    test_acc_over_runs = []
    for i, seed in enumerate(seeds):
        print(f"########## Run {i} for seed {seed} ##########")
        set_random_seed(seed=seed)
        adj, features, labels, train_mask, val_mask, test_mask = load_data_with_labels(dataset_str=args.dataset)
        num_nodes, num_features = features.shape

        # Store original adjacency matrix (without diagonal entries) for later
        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        fea_train = features
        adj_train = adj

        # following for test, no edges or attr augmentation
        adj_norm, pos_weight, norm, features, pos_weight_a, norm_a = prepare_inputs(
            adj=adj_train, features=fea_train, args=args)
        # 特征归一化
        # features = F.normalize(features, p=1, dim=1)

        adj_label = adj_train + sp.eye(adj_train.shape[0])  # 自环-边，计算重构loss时，需要用到节点与自己的边
        adj_label = torch.FloatTensor(adj_label.toarray()).to(args.device)
        labels = torch.argmax(torch.tensor(labels), dim=1).to(args.device)

        tolerance = 0  # early stopping
        outer_best_val_acc = -1
        outer_best_epoch = -1
        outer_best_test_acc = -1

        set_random_seed(seed=seed)

        model = HOANE(input_dim=num_features,
                      output_dim=z_dim,
                      dropout=args.dropout,
                      device=args.device,
                      node_noise_dim=noise_dim_u,
                      attr_noise_dim=noise_dim_a,
                      node_mu_hidden=hidden_u,
                      node_var_hidden=hidden_u_v,
                      attr_mu_hidden=hidden_a,
                      attr_var_hidden=hidden_a_v,
                      K=args.K,
                      J=args.J,
                      encoder_type=args.encoder_type,
                      decoder_type=args.decoder_type)

        model.to(args.device)
        optimizer = optim.Adam(params=model.parameters(), lr=args.pretrain_lr, weight_decay=args.pretrain_wd)
        for epoch in range(1, args.pretrain_epochs + 1):
            if tolerance > 20:
                break
            start_time = time.time()
            warmup = np.min([epoch / 300., 1.])
            model.train()
            optimizer.zero_grad()
            merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
            merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
            reconstruct_node_logits, reconstruct_attr_logits, node_mu_iw_vec, attr_mu_iw_vec = model(x=features,
                                                                                                     adj=adj_norm)

            node_attr_mu = torch.cat((merged_node_mu, merged_attr_mu), 0)
            node_attr_sigma = torch.cat((merged_node_sigma, merged_attr_sigma), 0)
            node_attr_z_samples = torch.cat((merged_node_z_samples, merged_attr_z_samples), 0)
            node_attr_logv_iw = torch.cat((node_logv_iw, attr_logv_iw), 0)

            ker = torch.exp(
                -0.5 * (torch.sum(
                    torch.square(node_attr_z_samples - node_attr_mu) / torch.square(node_attr_sigma + args.eps), 3)))

            log_H_iw_vec = torch.log(torch.mean(ker, 2) + args.eps) - 0.5 * torch.sum(node_attr_logv_iw, 2)
            log_H_iw = torch.mean(log_H_iw_vec, 0)

            # node重构loss
            adj_orig_tile = adj_label.unsqueeze(-1).repeat(1, 1, args.K)  # adj matrix
            log_lik_iw_node = -1 * get_rec_loss(norm=norm,
                                                pos_weight=pos_weight,
                                                pred=reconstruct_node_logits,
                                                labels=adj_orig_tile,
                                                loss_type='bce_loss')

            # node_z prior
            node_log_prior_iw_vec = -0.5 * torch.sum(torch.square(node_z_samples_iw), 2)
            node_log_prior_iw = torch.mean(node_log_prior_iw_vec, 0)

            # attr重构loss
            features_tile = features.unsqueeze(-1).repeat(1, 1, args.K)  # feature matrix
            log_lik_iw_attr = -1 * get_rec_loss(norm=norm_a,
                                                pos_weight=pos_weight_a,
                                                pred=reconstruct_attr_logits,
                                                labels=features_tile,
                                                loss_type='mse_loss')

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
                    node_mu_iw_vec, attr_mu_iw_vec = model.encode(
                        x=features,
                        adj=adj_norm)  # full edges and full attrs

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
