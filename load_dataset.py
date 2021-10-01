import json
import scipy
import pickle as pkl
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from heteo_data import load_data, read_relation_subsets, gen_rel_subset_feature, preprocess_features
import torch.nn.functional as F
import gc
import scipy.sparse as sp
import networkx as nx

def prepare_label_emb(args, g, labels, n_classes, train_idx, valid_idx, test_idx, label_teacher_emb=None):
    if args.dataset == 'ogbn-mag':
        target_type_id = g.get_ntype_id("paper")
        homo_g = dgl.to_homogeneous(g, ndata=["feat"])
        homo_g = dgl.add_reverse_edges(homo_g, copy_ndata=True)
        homo_g.ndata["target_mask"] = homo_g.ndata[dgl.NTYPE] == target_type_id
        feat = g.ndata['feat']['paper']
    print(n_classes)
    print(labels.shape[0])
    import os
    if (not os.path.exists(f'./data/{args.dataset}_label_0.pt')):
        y = np.zeros(shape=(labels.shape[0], int(n_classes)))
        y[train_idx] = F.one_hot(labels[train_idx].to(
            torch.long), num_classes=n_classes).float().squeeze(1)
        y = torch.Tensor(y)

    if args.dataset == 'ogbn-mag':
        target_mask = homo_g.ndata["target_mask"]
        target_ids = homo_g.ndata[dgl.NID][target_mask]
        num_target = target_mask.sum().item()
        new_label_emb = torch.zeros((len(homo_g.ndata["feat"]),) + y.shape[1:],
                                    dtype=y.dtype, device=y.device)
        new_label_emb[target_mask] = y[target_ids]
        y = new_label_emb
        g = homo_g
    del labels
    gc.collect()
    res=[]
    import os
    for hop in range(args.label_num_hops):
        if os.path.exists(f'./data/{args.dataset}_label_{hop}.pt'):
            y=torch.load(f'./data/{args.dataset}_label_{hop}.pt')
        else:
            y = neighbor_average_labels(g, y.to(torch.float), args)
            torch.save(y,f'./data/{args.dataset}_label_{hop}.pt')
        gc.collect()
        if hop>=args.label_start:
            if args.dataset == "ogbn-mag":
                    target_mask = g.ndata['target_mask']
                    target_ids = g.ndata[dgl.NID][target_mask]
                    num_target = target_mask.sum().item()
                    new_res = torch.zeros((num_target,) + y.shape[1:],
                            dtype=y.dtype, device=y.device)
                    new_res[target_ids] = y[target_mask]
            else:
                new_res=y
            res.append(torch.cat([new_res[train_idx], new_res[valid_idx], new_res[test_idx]], dim=0))
    return res


def neighbor_average_labels(g, feat, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged labels")
    g.ndata["f"] = feat
    g.update_all(fn.copy_u("f", "msg"),
                 fn.mean("msg", "f"))
    feat = g.ndata.pop('f')

    return feat


def neighbor_average_features(g, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats")
    g.ndata["feat_0"] = g.ndata["feat"]
    for hop in range(1, args.num_hops + 1):
        g.update_all(fn.copy_u(f"feat_{hop-1}", "msg"),
                     fn.mean("msg", f"feat_{hop}"))
    res = []
    for hop in range(args.num_hops + 1):
        res.append(g.ndata.pop(f"feat_{hop}"))
    return res

def batched_acc(labels,pred):
    # testing accuracy for single label multi-class prediction
    return (torch.argmax(pred, dim=1) == labels,)

def get_evaluator(dataset):
    dataset = dataset.lower()
    if dataset.startswith("oag"):
        return batched_ndcg_mrr
    else:
        return batched_acc

def get_ogb_evaluator(dataset):
    """
    Get evaluator from Open Graph Benchmark based on dataset
    """
#    if dataset=='ogbn-mag':
#        return batched_acc
#    else:
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
            "y_true": labels.view(-1, 1),
            "y_pred": preds.view(-1, 1),
        })["acc"]

#adapted from GCNII https://github.com/chennnM/GCNII/blob/master/utils.py
def accuracy(output, labels):
    #preds = output.type_as(labels)
    #output=output.cpu()
    correct = output.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_inv_sqrt = sp.diags(d_inv_sqrt)
   #adj=sparse
   #DA = d_inv_sqrt.view(-1,1) * d_inv_sqrt.view(-1,1)*adj
   #AD = adj*d_inv_sqrt.view(1,-1) * d_inv_sqrt.view(1,-1)
   DAD=d_inv_sqrt.dot(adj).dot(d_inv_sqrt)
   d_inv_sqrt=np.power(row_sum,-1).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)]=0
   d_inv_sqrt=sp.diags(d_inv_sqrt)
   DA=d_inv_sqrt.dot(adj).tocoo()
   return DA,DAD

def D_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -1).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# adapted from tkipf/gcn
def load_citation(dataset_str="cora"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
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
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    features = normalize(features)
    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    adj = sys_normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features, labels, idx_train, idx_val, idx_test

def load_dataset(name, device, args):
    """
    Load dataset and move graph and features to device
    """
    '''if name not in ["ogbn-products", "ogbn-arxiv","ogbn-mag"]:
        raise RuntimeError("Dataset {} is not supported".format(name))'''
    if name not in ["ogbn-products", "ogbn-mag","ogbn-papers100M"]:
        raise RuntimeError("Dataset {} is not supported".format(name))
    dataset = DglNodePropPredDataset(name=name, root=args.root)
    splitted_idx = dataset.get_idx_split()
    if name == "ogbn-products":
        train_nid = splitted_idx["train"]
        val_nid = splitted_idx["valid"]
        test_nid = splitted_idx["test"]
        g, labels = dataset[0]
        g.ndata["labels"] = labels
        g.ndata['feat'] = g.ndata['feat'].float()
        n_classes = dataset.num_classes
        labels = labels.squeeze()
        evaluator = get_ogb_evaluator(name)
    elif name == "ogbn-mag":
        data = load_data(device, args)
        g, labels, n_classes, train_nid, val_nid, test_nid = data
        evaluator = get_ogb_evaluator(name)
    elif name=="ogbn-papers100M":
        train_nid = splitted_idx["train"]
        val_nid = splitted_idx["valid"]
        test_nid = splitted_idx["test"]
        g, labels = dataset[0]
        n_classes = dataset.num_classes
        labels = labels.squeeze()
        evaluator = get_ogb_evaluator(name)
    print(f"# Nodes: {g.number_of_nodes()}\n"
          f"# Edges: {g.number_of_edges()}\n"
          f"# Train: {len(train_nid)}\n"
          f"# Val: {len(val_nid)}\n"
          f"# Test: {len(test_nid)}\n"
          f"# Classes: {n_classes}\n")

    return g, labels, n_classes, train_nid, val_nid, test_nid, evaluator

def load_large_dataset(prefix):
    """
    Load the various data files residing in the `prefix` directory.
    Files to be loaded:
        adj_full.npz        sparse matrix in CSR format, stored as scipy.sparse.csr_matrix
                            The shape is N by N. Non-zeros in the matrix correspond to all
                            the edges in the full graph. It doesn't matter if the two nodes
                            connected by an edge are training, validation or test nodes.
                            For unweighted graph, the non-zeros are all 1.
        adj_train.npz       sparse matrix in CSR format, stored as a scipy.sparse.csr_matrix
                            The shape is also N by N. However, non-zeros in the matrix only
                            correspond to edges connecting two training nodes. The graph
                            sampler only picks nodes/edges from this adj_train, not adj_full.
                            Therefore, neither the attribute information nor the structural
                            information are revealed during training. Also, note that only
                            a x N rows and cols of adj_train contains non-zeros. For
                            unweighted graph, the non-zeros are all 1.
        role.json           a dict of three keys. Key 'tr' corresponds to the list of all
                              'tr':     list of all training node indices
                              'va':     list of all validation node indices
                              'te':     list of all test node indices
                            Note that in the raw data, nodes may have string-type ID. You
                            need to re-assign numerical ID (0 to N-1) to the nodes, so that
                            you can index into the matrices of adj, features and class labels.
        class_map.json      a dict of length N. Each key is a node index, and each value is
                            either a length C binary list (for multi-class classification)
                            or an integer scalar (0 to C-1, for single-class classification).
        feats.npz           a numpy array of shape N by F. Row i corresponds to the attribute
                            vector of node i.
    Inputs:
        prefix              string, directory containing the above graph related files
        normalize           bool, whether or not to normalize the node features
    Outputs:
        adj_full            scipy sparse CSR (shape N x N, |E| non-zeros), the adj matrix of
                            the full graph, with N being total num of train + val + test nodes.
        adj_train           scipy sparse CSR (shape N x N, |E'| non-zeros), the adj matrix of
                            the training graph. While the shape is the same as adj_full, the
                            rows/cols corresponding to val/test nodes in adj_train are all-zero.
        feats               np array (shape N x f), the node feature matrix, with f being the
                            length of each node feature vector.
        class_map           dict, where key is the node ID and value is the classes this node
                            belongs to.
        role                dict, where keys are: 'tr' for train, 'va' for validation and 'te'
                            for test nodes. The value is the list of IDs of nodes belonging to
                            the train/val/test sets.
    """
    adj_full = scipy.sparse.load_npz('./data/{}/adj_full.npz'.format(prefix)).astype(np.bool)
    adj_train = scipy.sparse.load_npz('./data/{}/adj_train.npz'.format(prefix)).astype(np.bool)
    role = json.load(open('./data/{}/role.json'.format(prefix)))
    feats = np.load('./data/{}/feats.npy'.format(prefix))
    class_map = json.load(open('./data/{}/class_map.json'.format(prefix)))
    class_map = {int(k):v for k,v in class_map.items()}
    labels=[]
    for i in range(feats.shape[0]):
        labels.append(class_map[i])
    labels=torch.Tensor(labels).long()
    print(labels)
    if prefix in ['reddit','flickr']:
        labels=torch.Tensor(F.one_hot(labels,num_classes=int(max(labels)+1)).float())
    print(labels)
    idx_train=np.array(role['tr'])
    idx_val=np.array(role['va'])
    idx_test=np.array(role['te'])

    assert len(class_map) == feats.shape[0]
    # ---- normalize ----
    print("normalize")
    DA,DAD=normalized_adjacency(adj_full)
    adj = sparse_mx_to_torch_sparse_tensor(DA)
    feats = normalize(feats)
    feats=torch.FloatTensor(feats).float()
    print("normalize is over")
    # -------------------------
    return adj, feats, labels,idx_train,idx_val,idx_test
from sklearn.metrics import f1_score

def mutilabel_f1(y_pred,y_true):
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    #print(y_true)
    return f1_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average="micro")

def prepare_data(device, args):
    """
    Load dataset and compute neighbor-averaged node features used by SIGN model
    """
    dataset=args.dataset
    if dataset in ['ogbn-mag','ogbn-products','ogbn-papers100M']:
        data = load_dataset(args.dataset, device, args)
        g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data
        if args.dataset == 'ogbn-products':
            feats = neighbor_average_features(g, args)
            in_feats = feats[0].shape[1]
        elif args.dataset == 'ogbn-mag':
            rel_subsets = read_relation_subsets(args.use_relation_subsets)

            with torch.no_grad():
                feats = preprocess_features(g, rel_subsets, args, device)
                print("Done preprocessing")
            _, num_feats, in_feats = feats[0].shape
        elif args.dataset == 'ogbn-papers100M':
            g = dgl.add_reverse_edges(g, copy_ndata=True)
            feat=g.ndata.pop('feat')
        gc.collect()
        label_emb = None
        if args.use_label:
            label_emb = prepare_label_emb(args, g, labels, n_classes, train_nid, val_nid, test_nid)
        # move to device
        if args.dataset=='ogbn-papers100M':
            feats=[]
            for i in range(args.num_hops+1):
                feats.append(torch.load(f"./data/papers100m_feat_{i}.pt"))
            in_feats=feats[0].shape[1]
        else:
            for i, x in enumerate(feats):
                feats[i] = torch.cat((x[train_nid], x[val_nid], x[test_nid]), dim=0)
        train_nid = train_nid.to(device)
        val_nid = val_nid.to(device)
        test_nid = test_nid.to(device)
        labels = labels.to(device).to(torch.long)
    elif dataset in ['ppi','flickr','reddit','yelp','amazon']:
        adj, features, labels, train_nid, val_nid, test_nid=load_large_dataset(dataset)
        in_feats=features.shape[1]
        n_classes=labels.shape[1]
        if dataset in ['ppi','yelp']:
            evaluator=lambda preds, labels: mutilabel_f1(preds,labels)
        else:
            evaluator=lambda preds, labels: accuracy(preds,labels)
        feats=[]
        feats.append(features)
        print("feature preprocess begin")
        for i in range(args.num_hops):
            feats.append(adj@feats[-1])
        print("feature preprocess end")
        for i, x in enumerate(feats):
            feats[i] = torch.cat((x[train_nid], x[val_nid], x[test_nid]), dim=0)
        gc.collect()
        y = np.zeros(shape=(labels.shape[0], int(n_classes)))
        y[train_nid] = labels[train_nid].float()
        y = torch.Tensor(y)
        label_emb=[]
        for hop in range(args.label_num_hops+args.label_start):
             y=adj@y
             gc.collect()
             if hop>=args.label_start:
                label_emb.append(torch.cat([y[train_nid], y[val_nid], y[test_nid]], dim=0))
    return feats, torch.cat([labels[train_nid], labels[val_nid], labels[test_nid]]),int(in_feats), int(n_classes), \
            train_nid, val_nid, test_nid, evaluator, label_emb
