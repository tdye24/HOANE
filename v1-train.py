import time
import torch
import scipy
import numpy as np
import scipy.sparse as sp
from torch import optim
from utils import get_args, set_random_seed, load_data_with_labels, mask_test_edges, preprocess_graph, get_loss, \
    get_roc_score, accuracy
from model import HOANE


def main(args):
    print(f"Using {args.dataset} dataset")
    seeds = args.seeds
    print(f"seeds = {args.seeds}")
    test_roc_over_runs = []
    test_ap_over_runs = []
    for i, seed in enumerate(seeds):
        print(f"########## Run {i} for seed {seed} ##########")
        set_random_seed(seed=seed)
        # Loading dataset, independent of random seed
        adj, features, labels, train_mask, val_mask, test_mask = load_data_with_labels(dataset_str=args.dataset)
        # print(f"adj.shape = {adj.shape}")
        # print(f"features.shape = {features.shape}")
        # print(f"labels.shape = {labels.shape}")

        # print("Train mask: ")
        # print(np.where(train_mask == 1)[0].tolist())
        # print("Val mask: ")
        # print(np.where(val_mask == 1)[0].tolist())
        # print("Test mask: ")
        # print(np.where(test_mask == 1)[0].tolist())

        num_nodes, num_features = features.shape

        # Store original adjacency matrix (without diagonal entries) for later
        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        features_orig = features

        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
        adj = adj_train
        adj_label = adj_train + sp.eye(adj_train.shape[0])

        adj_norm = preprocess_graph(adj)
        adj_norm = scipy.sparse.coo_matrix((adj_norm[1], (adj_norm[0][:, 0], adj_norm[0][:, 1])),
                                           shape=adj_norm[2]).toarray()
        adj_norm = torch.FloatTensor(adj_norm)

        features = torch.FloatTensor(np.array(features.todense()))
        features_nonzero = torch.where(features == 1)[0].shape[0]

        # Distribution functions
        # bernoulli_dist = dist.Bernoulli(torch.tensor([.5], device=args.device))

        # Model hyper-parameters
        noise_dim_u = [5]
        noise_dim_a = [5]
        z_dim = 128
        hidden_u = [128]
        hidden_a = [128]
        hidden_u_v = [128]
        hidden_a_v = [128]

        pos_weight = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()).to(args.device)
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        pos_weight_a = torch.tensor(float(features.shape[0] * features.shape[1] - features_nonzero) / features_nonzero).to(args.device)
        norm_a = features.shape[0] * features.shape[1] / float(
            (features.shape[0] * features.shape[1] - features_nonzero) * 2)
        # pos_weight_a = float(features[2][0] * features[2][1] - len(features[1])) / len(features[1])
        # norm_a = features[2][0] * features[2][1] / float((features[2][0] * features[2][1] - len(features[1])) * 2)

        features = features.to(args.device)
        adj_norm = adj_norm.to(args.device)
        adj_label = torch.FloatTensor(adj_label.toarray()).to(args.device)
        labels = torch.argmax(torch.tensor(labels), dim=1).to(args.device)

        tolerance = 0  # early stopping
        best_roc_val = 0
        best_ap_val = 0
        best_roc_test = 0
        best_ap_test = 0

        set_random_seed(seed=seed)

        model = HOANE(input_dim=num_features,
                      dropout=0.0,
                      device=args.device,
                      node_noise_dim=5,
                      attr_noise_dim=5,
                      K=args.K,
                      J=args.J)

        # 为不同的module设置不同weight decay
        # params_decay = []
        # for name, param in model.named_parameters():
        #     if 'mlp' in name and 'weight' in name:
        #         print(name)
        #         params_decay.append(param)
        # optimizer = optim.Adam([
        #     {'params': params_decay, 'weight_decay': 1e-5}
        # ], lr=args.pretrain_lr)
        model.to(args.device)
        optimizer = optim.Adam(params=model.parameters(), lr=args.pretrain_lr)
        for epoch in range(1, args.pretrain_epochs + 1):
            if tolerance > 100:
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
            log_lik_iw_node = get_loss(norm=norm, pos_weight=pos_weight, pred=reconstruct_node_logits,
                                       labels=adj_orig_tile, epoch=epoch)
            # print("Node Rec Loss", torch.mean(log_lik_iw_node).item())

            # node_z prior
            node_log_prior_iw_vec = -0.5 * torch.sum(torch.square(node_z_samples_iw), 2)
            node_log_prior_iw = torch.mean(node_log_prior_iw_vec, 0)

            # attr重构loss
            features_tile = features.unsqueeze(-1).repeat(1, 1, args.K)  # feature matrix
            log_lik_iw_attr = get_loss(norm=norm_a, pos_weight=pos_weight_a, pred=reconstruct_attr_logits,
                                       labels=features_tile, epoch=epoch)
            # print("Attr Rec Loss", torch.mean(log_lik_iw_attr).item())

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
            # print("Node", torch.mean(log_lik_iw_node).item())
            # print("Attr", torch.mean(log_lik_iw_attr).item())
            # print(loss.item())
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
                    reconstruct_node_logits, reconstruct_attr_logits, node_mu_iw_vec, attr_mu_iw_vec = model(x=features,
                                                                                                             adj=adj_norm)
                    roc_curr_val, ap_curr_val = get_roc_score(edges_pos=val_edges, edges_neg=val_edges_false,
                                                              emb=node_mu_iw_vec.detach().cpu().numpy(), adj=adj_orig)
                    # val_roc_score.append(roc_curr)
                    # print("Val, ROC = {:.5f}".format(roc_curr_val), "AP = {:.5f}".format(ap_curr_val))

                    roc_curr_test, ap_curr_test = get_roc_score(edges_pos=test_edges, edges_neg=test_edges_false,
                                                                emb=node_mu_iw_vec.detach().cpu().numpy(), adj=adj_orig)
                    # tst_roc_score.append(roc_currt)
                    # print("Test, ROC = {:.5f}".format(roc_curr_test), "AP = {:.5f}".format(ap_curr_test))

                    if roc_curr_val > best_roc_val and ap_curr_val > best_ap_val:
                        tolerance = 0
                        best_roc_val = roc_curr_val
                        best_ap_val = ap_curr_val
                        best_roc_test = roc_curr_test
                        best_ap_test = ap_curr_test
                    else:
                        tolerance += 1

                    print("Epoch:", '%04d' % epoch, "val_ap=", "{:.5f}".format(ap_curr_val))
                    print("Epoch:", '%04d' % epoch, "val_roc=", "{:.5f}".format(roc_curr_val))
                    print("Epoch:", '%04d' % epoch, "test_ap=", "{:.5f}".format(ap_curr_test))
                    print("Epoch:", '%04d' % epoch, "test_roc=", "{:.5f}".format(roc_curr_test))
                    print('--------------------------------')

                # test classification performance
                if args.node_classification and epoch % args.finetune_interval == 0:
                    lr_classifier = torch.nn.Linear(in_features=node_mu_iw_vec.shape[1], out_features=7)
                    finetune_optimizer = optim.Adam(params=lr_classifier.parameters(), lr=args.finetune_lr, weight_decay=5e-4)
                    criterion = torch.nn.CrossEntropyLoss()

                    lr_classifier.to(args.device)
                    lr_classifier.train()
                    for f_epoch in range(args.finetune_epochs):
                        out = lr_classifier(node_mu_iw_vec)
                        # print(out.shape)
                        loss = criterion(out[train_mask], labels[train_mask])
                        finetune_optimizer.zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
                        finetune_optimizer.step()

                        with torch.no_grad():
                            lr_classifier.eval()
                            pred = lr_classifier(node_mu_iw_vec)
                            train_acc = accuracy(pred[train_mask], labels[train_mask])
                            val_acc = accuracy(pred[val_mask], labels[val_mask])
                            test_acc = accuracy(pred[test_mask], labels[test_mask])

                            print("f_epoch", f_epoch, "train acc", train_acc, "val acc", val_acc, "test acc", test_acc)

        print("val_roc:", '{:.5f}'.format(best_roc_val), "val_ap=", "{:.5f}".format(best_ap_val))
        print("test_roc:", '{:.5f}'.format(best_roc_test), "test_ap=", "{:.5f}".format(best_ap_test))

        test_roc_over_runs.append(best_roc_test)
        test_ap_over_runs.append(best_ap_test)

    print("Test ROC", test_roc_over_runs)
    print("Test AP", test_ap_over_runs)
    print("ROC: {:.5f}".format(np.mean(test_roc_over_runs)), "+-", "{:.5f}".format(np.std(test_roc_over_runs)))
    print("AP: {:.5f}".format(np.mean(test_ap_over_runs)), "+-", "{:.5f}".format(np.std(test_ap_over_runs)))


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)