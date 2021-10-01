import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import uuid
import random
from model import R_GAMLP,JK_GAMLP
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

def gen_model(args,in_size,num_classes):
    if args.method=="R_GAMLP":
        return R_GAMLP(in_size, args.hidden, num_classes,args.num_hops+1,args.label_num_hops+1,
                 args.dropout, args.input_drop,args.att_drop,args.label_drop,args.alpha,args.n_layers_1,args.n_layers_2,args.n_layers_3,args.n_layers_4,args.act,args.pre_process,args.residual,args.use_label)
    elif args.method=="JK_GAMLP":
        return JK_GAMLP(in_size, args.hidden, num_classes,args.num_hops+1,args.label_num_hops+1,
                 args.dropout, args.input_drop,args.att_drop,args.label_drop,args.alpha,args.n_layers_1,args.n_layers_2,args.n_layers_3,args.n_layers_4,args.act,args.pre_process,args.residual,args.use_label)

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def train(model, feats, labels, loss_fcn, optimizer, train_loader,label_emb,evaluator,dataset,use_label):
    model.train()
    device = labels.device
    total_loss = 0
    iter_num=0
    y_true=[]
    y_pred=[]
    for batch in train_loader:
        batch_feats = [x[batch].to(device) for x in feats]
        if use_label:
            batch_label = [x[batch].to(device) for x in label_emb]
        else:
            batch_label=[]
        output_att=model(batch_feats,batch_label)
        y_true.append(labels[batch].to(torch.long))
        y_pred.append(output_att.argmax(dim=-1))
        L1 = loss_fcn(output_att, labels[batch].long())
        loss_train = L1
        total_loss = loss_train
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        iter_num+=1
    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_pred, dim=0),torch.cat(y_true))
    return loss,acc

@torch.no_grad()
def test(model, feats, labels, test_loader, evaluator, label_emb,dataset,use_label):
    model.eval()
    device = labels.device
    preds = []
    true=[]
    for batch in test_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        if use_label:
            batch_label = [x[batch].to(device) for x in label_emb]
        else:
            batch_label = []
        #preds.append(torch.argmax(model(batch_feats,batch_label), dim=-1))
        if dataset in ['ppi','yelp']:
            true.append(labels[batch].to(torch.float))
            preds.append(torch.sigmoid(model(batch_feats,batch_label)))
        else:
            true.append(labels[batch].to(torch.long))
            preds.append(torch.argmax(model(batch_feats,batch_label), dim=-1))
    true=torch.cat(true)
    preds = torch.cat(preds, dim=0)
    res = evaluator(preds, true)

    return res
def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
