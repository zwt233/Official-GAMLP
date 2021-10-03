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
import uuid
import gc

from load_dataset import prepare_data
from utils import gen_output_torch, set_seed, train,  test,  gen_model, gen_model_mag



def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def run(args, device):
    checkpt_file = f"./output/{args.dataset}/"+uuid.uuid4().hex
    with torch.no_grad():
        data = prepare_data(device, args)
    feats, labels, in_size, num_classes, \
            train_nid, val_nid, test_nid, evaluator,label_emb = data
    train_loader = torch.utils.data.DataLoader(
                 torch.arange(len(train_nid)), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
            torch.arange(len(train_nid),len(train_nid)+len(val_nid)), batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
            torch.arange(len(train_nid)+len(val_nid),len(train_nid)+len(val_nid)+len(test_nid)), batch_size=args.eval_batch_size,
            shuffle=False, drop_last=False)
    all_loader = torch.utils.data.DataLoader(
            torch.arange(len(train_nid)+len(val_nid)+len(test_nid)), batch_size=args.eval_batch_size,
            shuffle=False, drop_last=False)

    train_node_nums = len(train_nid)
    valid_node_nums = len(val_nid)
    test_node_nums = len(test_nid)
    total_num_nodes = len(train_nid) + len(val_nid) + len(test_nid)
    if args.dataset == "ogbn-mag":
        _, num_feats, in_feats = feats[0].shape
        model = gen_model_mag(args, num_feats, in_feats, num_classes)
    else:
        model = gen_model(args, in_size, num_classes)
    print(model)
    model = model.to(device)
    labels=labels.to(device)

    print("# Params:", get_n_params(model))
    loss_fcn = nn.CrossEntropyLoss()
    if args.method == 'JK_GAMLP':
        optimizer_sett = [
        {'params': model.lr_att.parameters(), 'weight_decay': 0, 'lr': 0.0001},
        {'params': model.process.parameters(), 'weight_decay': 0, 'lr': 0.0001},
        {'params': model.lr_jk_ref.parameters(), 'weight_decay': 0, 'lr': 0.0001}, 
        {'params': model.lr_output.parameters(), 'weight_decay': 0, 'lr': 0.0001},
        {'params': model.label_att.parameters(), 'weight_decay': 0, 'lr': 1e-4},
        {'params': model.label_output.parameters(), 'weight_decay': 0, 'lr': 1e-4},
        ]
    else:
        optimizer_sett = [
        {'params': model.lr_att.parameters(), 'weight_decay': 0, 'lr': 0.0001},
        {'params': model.process.parameters(), 'weight_decay': 0, 'lr': 0.0001},
        {'params': model.lr_output.parameters(), 'weight_decay': 0, 'lr': 0.0001},
        {'params': model.label_att.parameters(), 'weight_decay': 0, 'lr': 1e-4},
        {'params': model.label_output.parameters(), 'weight_decay': 0, 'lr': 1e-4},
        ]
    optimizer = torch.optim.Adam(optimizer_sett)
    # Start training
    best_epoch = 0
    best_val = 0
    best_test = 0
    count = 0

    for epoch in range(args.epochs+1):
        gc.collect()
        start = time.time()
        loss,acc=train(model, feats, labels, loss_fcn, optimizer, train_loader, label_emb,evaluator,args.dataset,args.use_label)
        end = time.time()
        log = "Epoch {}, Time(s): {:.4f},Train loss: {:.4f}, Train acc: {:.4f} ".format(epoch, end - start,loss,acc*100)
        if epoch % args.eval_every == 0 and epoch >args.train_epochs:
            with torch.no_grad():
                acc = test(model, feats, labels, val_loader, evaluator,
                            label_emb,args.dataset,args.use_label)
            end = time.time()
            log += "Epoch {}, Time(s): {:.4f}, ".format(epoch, end - start)
            log += "Val {:.4f}, ".format(acc)
            if acc > best_val:
                best_epoch = epoch
                best_val = acc
                best_test = test(model, feats, labels, test_loader, evaluator,
                                label_emb,args.dataset,args.use_label)
                torch.save(model.state_dict(),checkpt_file+f'.pkl')
                count = 0
            else:
                count = count+args.eval_every
                if count >= args.patience:
                    break
            log += "Best Epoch {},Val {:.4f}, Test {:.4f}".format(
                            best_epoch, best_val, best_test)
        print(log)

    print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(
            best_epoch, best_val, best_test))
    return best_val, best_test

def main(args):
    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.gpu)

    val_accs = []
    test_accs = []
    for i in range(args.num_runs):
        print(f"Run {i} start training")
        set_seed(args.seed+i)
        best_val, best_test = run(args, device)
        val_accs.append(best_val)
        test_accs.append(best_test)

    print(f"Average val accuracy: {np.mean(val_accs):.4f}, "
          f"std: {np.std(val_accs):.4f}")
    print(f"Average test accuracy: {np.mean(test_accs):.4f}, "
          f"std: {np.std(test_accs):.4f}")

    return np.mean(test_accs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMLP")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--num-hops", type=int, default=5,
                        help="number of hops")
    parser.add_argument("--label-num-hops",type=int,default=3,
                        help="number of hops for label")
    parser.add_argument("--seed", type=int, default=0,
                        help="the seed used in the training")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout on activation")
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--eval-batch-size", type=int, default=500000)
    parser.add_argument("--n-layers-1", type=int, default=4,
                        help="number of feed-forward layers")
    parser.add_argument("--n-layers-2", type=int, default=4,
                        help="number of feed-forward layers")
    parser.add_argument("--n-layers-3", type=int, default=4,
                        help="number of feed-forward layers")
    parser.add_argument("--n-layers-4", type=int, default=4,
                        help="number of feed-forward layers")
    parser.add_argument("--num-runs", type=int, default=10,
                        help="number of times to repeat the experiment")
    parser.add_argument("--patience", type=int, default=100,
                        help="early stop of times of the experiment")
    parser.add_argument("--alpha", type=float, default=1,
                        help="temperature of the output prediction")
    parser.add_argument("--beta", type=float, default=0,
                        help="temperature of the output prediction")
    parser.add_argument("--input-drop", type=float, default=0,
                        help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0.5,
                        help="attention dropout of model")
    parser.add_argument("--label-drop", type=float, default=0.5,
                        help="label feature dropout of model")
    parser.add_argument("--label-start", type=int, default=0,
                        help="label feat dropout of model")
    parser.add_argument("--pre-process", action='store_true', default=False,
                        help="whether to process the input features")
    parser.add_argument("--use-label", action='store_true', default=False,
                        help="whether to use the label information")
    parser.add_argument("--residual", action='store_true', default=False,
                        help="whether to connect the input features")
    parser.add_argument("--act", type=str, default="relu",
                        help="the activation function of the model")
    parser.add_argument("--method", type=str, default="JK_GAMLP",
                        help="the model to use")
    parser.add_argument("--use-emb", type=str)
    parser.add_argument("--root", type=str, default='/data4/zwt/')
    parser.add_argument("--emb_path", type=str, default='/data4/zwt/NARS-main')
    parser.add_argument("--use-relation-subsets", type=str, default='/data4/zwt/NARS-main/sample_relation_subsets/examples/mag')
    parser.add_argument("--train-epochs", type=int, default=0,
                        help="The Train epoch setting for each stage.")
    parser.add_argument("--epochs",type=int, default=1000,
                        help="The epoch setting for each stage.")

    args = parser.parse_args()
    print(args)
    main(args)
