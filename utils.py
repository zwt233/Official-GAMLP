import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from load_dataset import load_dataset
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import uuid
import random
from model import R_GAMLP,JK_GAMLP,NARS_JK_GAMLP,NARS_R_GAMLP

def gen_model_mag(args,num_feats,in_feats,num_classes):
    if args.method=="R_GAMLP":
        return NARS_R_GAMLP(in_feats, args.hidden, num_classes, args.num_hops+1,num_feats,args.alpha,args.n_layers_1,args.n_layers_2,args.n_layers_3,args.n_layers_4,args.act,args.dropout, args.input_drop, args.att_drop,args.label_drop,args.pre_process,args.residual,args.use_label)
    elif args.method=="JK_GAMLP":
        return NARS_JK_GAMLP(in_feats, args.hidden, num_classes, args.num_hops+1,num_feats,args.alpha,args.n_layers_1,args.n_layers_2,args.n_layers_3,args.n_layers_4,args.act,args.dropout, args.input_drop, args.att_drop,args.label_drop,args.pre_process,args.residual,args.use_label)
def gen_model(args,in_size,num_classes):
    if args.method=="R_GAMLP":
        return R_GAMLP(in_size, args.hidden, num_classes,args.num_hops+1,args.label_num_hops,
                 args.dropout, args.input_drop,args.att_drop,args.label_drop,args.alpha,args.n_layers_1,args.n_layers_2,args.n_layers_3,args.n_layers_4,args.act,args.pre_process,args.residual,args.use_label)
    elif args.method=="JK_GAMLP":
        return JK_GAMLP(in_size, args.hidden, num_classes,args.num_hops+1,args.label_num_hops,
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

@torch.no_grad()
def gen_output_torch(model, feats, test_loader, device, label_emb):
    model.eval()
    preds = []
    for batch in test_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        preds.append(model(batch_feats,label_emb[batch].to(device)).cpu())
    preds = torch.cat(preds, dim=0)
    return preds
