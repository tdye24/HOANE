import pickle as pkl
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import torch
import torch.nn as nn
import random
import argparse
import scipy
from torch import optim
from layers import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from dgl.data import (
    CoraGraphDataset,
    CiteseerGraphDataset,
    PubmedGraphDataset
)

GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='v1', help='Method version.')
    parser.add_argument('--node-classification', action='store_true', default=False, help='Do node classification.')
    parser.add_argument('--attr-inference', action='store_true', default=False, help='Do attribute inference.')
    parser.add_argument('--link-prediction', action='store_true', default=False, help='Do link prediction.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seeds', type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help='Random seed.')
    parser.add_argument('--pretrain-epochs', type=int, default=500, help='Number of epochs to pretrain.')
    parser.add_argument('--finetune-epochs', type=int, default=500, help='Number of epochs to finetune.')
    parser.add_argument('--pretrain-lr', type=float, default=0.001, help='Initial pretrain learning rate.')
    parser.add_argument('--finetune-lr', type=float, default=0.01, help='Initial finetune learning rate.')
    parser.add_argument('--pretrain-wd', type=float, default=0, help='Weight decay for pretraining.')
    parser.add_argument('--finetune-wd', type=float, default=0, help='Weight decay for finetune.')
    parser.add_argument('--finetune-interval', type=int, default=10, help='Interval between two finetune.')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--noise-dist', type=str, default='Bernoulli',
                        help='Distriubtion of random noise in generating psi.')
    parser.add_argument('--K', type=int, default=1,
                        help='Number of samples for importance re-weighting.')
    parser.add_argument('--J', type=int, default=1,
                        help='Number of samples for variational distribution q.')
    parser.add_argument('--eps', type=float, default=1e-10,
                        help='Eps')
    parser.add_argument('--warmup', type=float, default=0,
                        help='Warmup')
    parser.add_argument('--display-step', type=int, default=50, help='Training loss display step.')
    parser.add_argument('--encoder-type', type=str, default='gat', help='Encoder type (attributes).')
    parser.add_argument('--decoder-type', type=str, default='gat', help='Decoder type.')
    parser.add_argument('--node-loss-type', type=str, default='bce_loss', help='Node loss type.')
    parser.add_argument('--attr-loss-type', type=str, default='mse_loss', help='Attr loss type.')
    parser.add_argument('--aug-e', type=float, default=0.0, help='Mask ratio of edges.')
    parser.add_argument('--aug-a', type=float, default=0.0, help='Mask ratio of attributes.')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'

    return args


def load_dataset_dgl(dataset_name):
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    dataset = GRAPH_DICT[dataset_name]()
    graph = dataset[0]
    graph = graph.remove_self_loop()
    graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        # with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
        #     u = pkl._Unpickler(rf)
        #     u.encoding = 'latin1'
        #     cur_data = u.load()
        #     objects.append(cur_data)

        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    # features = features / features.sum(-1, keepdim=True)
    # adding a dimension to features for future expansion
    if len(features.shape) == 2:
        features = features.view([1, features.shape[0], features.shape[1]])
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def mask_test_feas(features):
    fea_row = features.nonzero()[0]
    fea_col = features.nonzero()[1]
    feas = []
    feas_dic = {}
    for i in range(len(fea_row)):
        feas.append([fea_row[i], fea_col[i]])
        feas_dic[(fea_row[i], fea_col[i])] = 1
    false_feas_dic = {}
    num_test = int(np.floor(len(feas) / 10.))
    num_val = int(np.floor(len(feas) / 20.))
    all_fea_idx = np.arange(len(feas))
    np.random.shuffle(all_fea_idx)
    val_fea_idx = all_fea_idx[:num_val]
    test_fea_idx = all_fea_idx[num_val:(num_val + num_test)]
    feas = np.array(feas)
    test_feas = feas[test_fea_idx]
    val_feas = feas[val_fea_idx]
    train_feas = np.delete(feas, np.hstack([test_fea_idx, val_fea_idx]), axis=0)
    test_feas_false = []
    val_feas_false = []
    while len(test_feas_false) < num_test or len(val_feas_false) < num_val:
        i = np.random.randint(0, features.shape[0])
        j = np.random.randint(0, features.shape[1])
        if (i, j) in feas_dic:
            continue
        if (i, j) in false_feas_dic:
            continue
        else:
            false_feas_dic[(i, j)] = 1
        if np.random.random_sample() > 0.333:
            if len(test_feas_false) < num_test:
                test_feas_false.append([i, j])
            else:
                if len(val_feas_false) < num_val:
                    val_feas_false.append([i, j])
        else:
            if len(val_feas_false) < num_val:
                val_feas_false.append([i, j])
            else:
                if len(test_feas_false) < num_test:
                    test_feas_false.append([i, j])
    data = np.ones(train_feas.shape[0])
    fea_train = sp.csr_matrix((data, (train_feas[:, 0], train_feas[:, 1])), shape=features.shape)
    return fea_train, train_feas, val_feas, val_feas_false, test_feas, test_feas_false


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data_with_labels(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, labels, train_mask, val_mask, test_mask


def get_rec_loss(norm, pos_weight, pred, labels, loss_type='bce_loss'):
    if loss_type == 'bce_loss':
        return norm * torch.mean(
            F.binary_cross_entropy_with_logits(input=pred, target=labels, reduction='none', pos_weight=pos_weight),
            dim=[0, 1])
    elif loss_type == 'sce_loss':
        return norm * sce_loss(pred, labels)
    elif loss_type == 'mse_loss':
        return norm * torch.mean(
            F.mse_loss(input=pred, target=labels, reduction='none'),
            dim=[0, 1]
        )
    else:
        assert loss_type == 'sig_loss'
        return norm * sig_loss(pred, labels)


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    # loss = loss.mean()
    loss = torch.mean(loss, dim=[0, 1])
    return loss


def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    # loss = loss.mean()
    loss = torch.mean(loss, dim=[0, 1])
    return loss


def get_roc_score_node(edges_pos, edges_neg, emb, adj):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def get_roc_score_attr(feas_pos, feas_neg, emb_node, emb_attr, features_orig):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    fea_rec = np.dot(emb_node, emb_attr.T)
    preds = []
    pos = []
    for e in feas_pos:
        preds.append(sigmoid(fea_rec[e[0], e[1]]))
        pos.append(features_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in feas_neg:
        preds_neg.append(sigmoid(fea_rec[e[0], e[1]]))
        neg.append(features_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def classes_num(dataset_str):
    return {'cora': 7,
            'citeseer': 6,
            'pubmed': 3}[dataset_str]


def adj_augment(adj_mat, aug_prob):
    # adj_mat: scipy sparse matrix
    # Symmetric Matrices

    # change inplace
    # copy is very important
    # aug_prob /= 2
    adj_mat = adj_mat.copy()
    xrow, yrow = adj_mat.nonzero()
    low_tri = xrow > yrow
    xrow = xrow[low_tri]
    yrow = yrow[low_tri]
    num_indices = len(xrow)
    selected_idx = random.sample(range(num_indices), int(num_indices * aug_prob))
    selected_idx.sort()
    xrow_ = xrow[selected_idx]
    yrow_ = yrow[selected_idx]
    adj_mat[xrow_, yrow_] = 0
    adj_mat[yrow_, xrow_] = 0
    adj_mat.eliminate_zeros()
    return adj_mat


def attr_augment(attr_mat, aug_prob):
    # attr_mat: scipy dense matrix

    # change inplace
    # copy is very important
    attr_mat = attr_mat.copy()
    xrow, yrow = attr_mat.nonzero()
    num_indices = len(xrow)
    selected_idx = random.sample(range(num_indices), int(num_indices * aug_prob))
    selected_idx.sort()
    xrow_ = xrow[selected_idx]
    yrow_ = yrow[selected_idx]
    attr_mat[xrow_, yrow_] = 0
    return attr_mat


def prepare_inputs(adj, features, args):
    adj_norm = preprocess_graph(adj)
    adj_norm = scipy.sparse.coo_matrix((adj_norm[1], (adj_norm[0][:, 0], adj_norm[0][:, 1])),
                                       shape=adj_norm[2]).toarray()
    adj_norm = torch.FloatTensor(adj_norm)
    adj_norm = adj_norm.to(args.device)
    pos_weight_node = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()).to(args.device)
    norm_node = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    features = torch.FloatTensor(np.array(features.todense()))
    features = features.to(args.device)
    features_nonzero = torch.where(features == 1)[0].shape[0]

    pos_weight_attr = torch.tensor(
        float(features.shape[0] * features.shape[1] - features_nonzero) / features_nonzero).to(args.device)
    norm_attr = features.shape[0] * features.shape[1] / float(
        (features.shape[0] * features.shape[1] - features_nonzero) * 2)

    return adj_norm, pos_weight_node, norm_node, features, pos_weight_attr, norm_attr


def node_classification_evaluation(data, labels, train_mask, val_mask, test_mask, args):
    lr_classifier = LogisticRegression(num_dim=data.shape[1],
                                       num_class=classes_num(args.dataset)).to(args.device)
    finetune_optimizer = optim.Adam(params=lr_classifier.parameters(), lr=args.finetune_lr,
                                    weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_acc = -1
    best_epoch = -1
    best_val_test_acc = -1
    for f_epoch in range(args.finetune_epochs):
        lr_classifier.train()
        out = lr_classifier(data)
        # print(out.shape)
        loss = criterion(out[train_mask], labels[train_mask])
        finetune_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lr_classifier.parameters(), max_norm=3)
        finetune_optimizer.step()
        with torch.no_grad():
            lr_classifier.eval()
            pred = lr_classifier(data)
            train_acc = accuracy(pred[train_mask], labels[train_mask])
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_epoch = f_epoch
                best_val_test_acc = test_acc
            # print("f_epoch", f_epoch, "train acc", train_acc, "val acc", val_acc, "test acc", test_acc)
    return test_acc, best_val_acc, best_val_test_acc


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")
