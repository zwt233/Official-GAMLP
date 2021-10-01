from utils import *
from model import *
import json
from networkx.readwrite import json_graph
import torch.utils.data as Data
import sys
import networkx as nx
import scipy.sparse as sp
sys.setrecursionlimit(99999)
def find_split(adj, mapping, ds_label):
    nb_nodes = adj.shape[0]
    dict_splits={}
    for i in range(nb_nodes):
        #for j in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i]==0 or mapping[j]==0:
                dict_splits[0]=None
            elif mapping[i] == mapping[j]:
                if ds_label[i]['val'] == ds_label[j]['val'] and ds_label[i]['test'] == ds_label[j]['test']:

                    if mapping[i] not in dict_splits.keys():
                        if ds_label[i]['val']:
                            dict_splits[mapping[i]] = 'val'

                        elif ds_label[i]['test']:
                            dict_splits[mapping[i]]='test'

                        else:
                            dict_splits[mapping[i]] = 'train'

                    else:
                        if ds_label[i]['test']:
                            ind_label='test'
                        elif ds_label[i]['val']:
                            ind_label='val'
                        else:
                            ind_label='train'
                        if dict_splits[mapping[i]]!= ind_label:
                            print ('inconsistent labels within a graph exiting!!!')
                            return None
                else:
                    print ('label of both nodes different, exiting!!')
                    return None
    return dict_splits
def dfs_split(adj):
    # Assume adj is of shape [nb_nodes, nb_nodes]
    nb_nodes = adj.shape[0]
    ret = np.full(nb_nodes, -1, dtype=np.int32)

    graph_id = 0

    for i in range(nb_nodes):
        if ret[i] == -1:
            run_dfs(adj, ret, i, graph_id, nb_nodes)
            graph_id += 1

    return ret
def run_dfs(adj, msk, u, ind, nb_nodes):
    if msk[u] == -1:
        msk[u] = ind
        #for v in range(nb_nodes):
        for v in adj[u,:].nonzero()[1]:
            #if adj[u,v]== 1:
            run_dfs(adj, msk, v, ind, nb_nodes)  
def load_ppi():
    print ('Loading G...')
    with open('../data/ppi/ppi-G.json') as jsonfile:
        g_data = json.load(jsonfile)
    # print (len(g_data))
    G = json_graph.node_link_graph(g_data)

    #Extracting adjacency matrix
    adj=nx.adjacency_matrix(G)

    prev_key=''
    for key, value in g_data.items():
        if prev_key!=key:
            # print (key)
            prev_key=key

    # print ('Loading id_map...')
    with open('../data/ppi/ppi-id_map.json') as jsonfile:
        id_map = json.load(jsonfile)
    # print (len(id_map))

    id_map = {int(k):int(v) for k,v in id_map.items()}
    for key, value in id_map.items():
        id_map[key]=[value]
    # print (len(id_map))

    print ('Loading features...')
    features_=np.load('../data/ppi/ppi-feats.npy')
    # print (features_.shape)

    #standarizing features
    from sklearn.preprocessing import StandardScaler

    train_ids = np.array([id_map[n] for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']])
    train_feats = features_[train_ids[:,0]]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    features_ = scaler.transform(features_)

    features = sp.csr_matrix(features_).tolil()


    print ('Loading class_map...')
    class_map = {}
    with open('../data/ppi/ppi-class_map.json') as jsonfile:
        class_map = json.load(jsonfile)
    # print (len(class_map))
    
    #pdb.set_trace()
    #Split graph into sub-graphs
    # print ('Splitting graph...')
    splits=dfs_split(adj)

    #Rearrange sub-graph index and append sub-graphs with 1 or 2 nodes to bigger sub-graphs
    # print ('Re-arranging sub-graph IDs...')
    list_splits=splits.tolist()
    group_inc=1

    for i in range(np.max(list_splits)+1):
        if list_splits.count(i)>=3:
            splits[np.array(list_splits) == i] =group_inc
            group_inc+=1
        else:
            #splits[np.array(list_splits) == i] = 0
            ind_nodes=np.argwhere(np.array(list_splits) == i)
            ind_nodes=ind_nodes[:,0].tolist()
            split=None
            
            for ind_node in ind_nodes:
                if g_data['nodes'][ind_node]['val']:
                    if split is None or split=='val':
                        splits[np.array(list_splits) == i] = 21
                        split='val'
                    else:
                        raise ValueError('new node is VAL but previously was {}'.format(split))
                elif g_data['nodes'][ind_node]['test']:
                    if split is None or split=='test':
                        splits[np.array(list_splits) == i] = 23
                        split='test'
                    else:
                        raise ValueError('new node is TEST but previously was {}'.format(split))
                else:
                    if split is None or split == 'train':
                        splits[np.array(list_splits) == i] = 1
                        split='train'
                    else:
                        pdb.set_trace()
                        raise ValueError('new node is TRAIN but previously was {}'.format(split))

    #counting number of nodes per sub-graph
    list_splits=splits.tolist()
    nodes_per_graph=[]
    for i in range(1,np.max(list_splits) + 1):
        nodes_per_graph.append(list_splits.count(i))

    #Splitting adj matrix into sub-graphs
    subgraph_nodes=np.max(nodes_per_graph)
    adj_sub=np.empty((len(nodes_per_graph), subgraph_nodes, subgraph_nodes))
    feat_sub = np.empty((len(nodes_per_graph), subgraph_nodes, features.shape[1]))
    labels_sub = np.empty((len(nodes_per_graph), subgraph_nodes, 121))

    for i in range(1, np.max(list_splits) + 1):
        #Creating same size sub-graphs
        indexes = np.where(splits == i)[0]
        subgraph_=adj[indexes,:][:,indexes]

        if subgraph_.shape[0]<subgraph_nodes or subgraph_.shape[1]<subgraph_nodes:
            subgraph=np.identity(subgraph_nodes)
            feats=np.zeros([subgraph_nodes, features.shape[1]])
            labels=np.zeros([subgraph_nodes,121])
            #adj
            subgraph = sp.csr_matrix(subgraph).tolil()
            subgraph[0:subgraph_.shape[0],0:subgraph_.shape[1]]=subgraph_
            adj_sub[i-1,:,:]=subgraph.todense()
            #feats
            feats[0:len(indexes)]=features[indexes,:].todense()
            feat_sub[i-1,:,:]=feats
            #labels
            for j,node in enumerate(indexes):
                labels[j,:]=np.array(class_map[str(node)])
            labels[indexes.shape[0]:subgraph_nodes,:]=np.zeros([121])
            labels_sub[i - 1, :, :] = labels

        else:
            adj_sub[i - 1, :, :] = subgraph_.todense()
            feat_sub[i - 1, :, :]=features[indexes,:].todense()
            for j,node in enumerate(indexes):
                labels[j,:]=np.array(class_map[str(node)])
            labels_sub[i-1, :, :] = labels

    # Get relation between id sub-graph and tran,val or test set
    dict_splits = find_split(adj, splits, g_data['nodes'])

    # Testing if sub graphs are isolated
    # print ('Are sub-graphs isolated?')
    # print (test(adj, splits))

    #Splitting tensors into train,val and test
    train_split=[]
    val_split=[]
    test_split=[]
    for key, value in dict_splits.items():
        if dict_splits[key]=='train':
            train_split.append(int(key)-1)
        elif dict_splits[key] == 'val':
            val_split.append(int(key)-1)
        elif dict_splits[key] == 'test':
            test_split.append(int(key)-1)

    train_adj=adj_sub[train_split,:,:]
    val_adj=adj_sub[val_split,:,:]
    test_adj=adj_sub[test_split,:,:]

    train_feat=feat_sub[train_split,:,:]
    val_feat = feat_sub[val_split, :, :]
    test_feat = feat_sub[test_split, :, :]

    train_labels = labels_sub[train_split, :, :]
    val_labels = labels_sub[val_split, :, :]
    test_labels = labels_sub[test_split, :, :]

    train_nodes=np.array(nodes_per_graph[train_split[0]:train_split[-1]+1])
    val_nodes = np.array(nodes_per_graph[val_split[0]:val_split[-1]+1])
    test_nodes = np.array(nodes_per_graph[test_split[0]:test_split[-1]+1])


    #Masks with ones

    tr_msk = np.zeros((len(nodes_per_graph[train_split[0]:train_split[-1]+1]), subgraph_nodes))
    vl_msk = np.zeros((len(nodes_per_graph[val_split[0]:val_split[-1] + 1]), subgraph_nodes))
    ts_msk = np.zeros((len(nodes_per_graph[test_split[0]:test_split[-1]+1]), subgraph_nodes))

    for i in range(len(train_nodes)):
        for j in range(train_nodes[i]):
            tr_msk[i][j] = 1

    for i in range(len(val_nodes)):
        for j in range(val_nodes[i]):
            vl_msk[i][j] = 1

    for i in range(len(test_nodes)):
        for j in range(test_nodes[i]):
            ts_msk[i][j] = 1

    train_adj_list = []
    val_adj_list = []
    test_adj_list = []
    for i in range(train_adj.shape[0]):
        adj = sp.coo_matrix(train_adj[i])
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        tmp = sys_normalized_adjacency(adj)
        train_adj_list.append(sparse_mx_to_torch_sparse_tensor(tmp))
    for i in range(val_adj.shape[0]):
        adj = sp.coo_matrix(val_adj[i])
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        tmp = sys_normalized_adjacency(adj)
        val_adj_list.append(sparse_mx_to_torch_sparse_tensor(tmp))
        adj = sp.coo_matrix(test_adj[i])
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        tmp = sys_normalized_adjacency(adj)
        test_adj_list.append(sparse_mx_to_torch_sparse_tensor(tmp))

    train_feat = torch.FloatTensor(train_feat)
    val_feat = torch.FloatTensor(val_feat)
    test_feat = torch.FloatTensor(test_feat)

    train_labels = torch.FloatTensor(train_labels)
    val_labels = torch.FloatTensor(val_labels)
    test_labels = torch.FloatTensor(test_labels)

    tr_msk = torch.LongTensor(tr_msk)
    vl_msk = torch.LongTensor(vl_msk)
    ts_msk = torch.LongTensor(ts_msk)

    return train_adj_list,val_adj_list,test_adj_list,train_feat,val_feat,test_feat,train_labels,val_labels, test_labels, train_nodes, val_nodes, test_nodes

def evaluate(epochs,feats, model ,idx, labels, loss_fcn,label_list):
    model.eval()
    with torch.no_grad():
        output_att= model(feats,label_list)
        L1 = loss_fcn(output_att[:idx], labels[:idx].float())
        #L2 = loss_fcn(output_concat[:idx], labels[:idx].float())
        loss_data = L1    
        predict = np.where(output_att[:idx].data.cpu().numpy() > 0.5, 1, 0)
        score = f1_score(labels.data[:idx].cpu().numpy(),predict, average='micro')
        return loss_data,score

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    #parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    #parser.add_argument('--epochs', type=int, default=800,
    #                help='Number of epochs to train.')
    #parser.add_argument('--lr', type=float, default=0.001,
    #                help='Initial learning rate.')
    #parser.add_argument('--wd', type=float, default=0,
    #                help='Weight decay (L2 loss on parameters).')
    #parser.add_argument('--hidden', type=int, default=2048,
    #                help='Number of hidden units.')
    #parser.add_argument('--dropout', type=float, default=0.1,
    #                help='Dropout rate (1 - keep probability). ')
    #parser.add_argument('--pl',type=int,default=10,help="the number of previous layers in the model")
    #
    #parser.add_argument('--dataset',type=str,default="ppi",help="dataset")
    ##parser.add_argument('--patience',type=int,default=100,help="patience of early stop")
    parser = argparse.ArgumentParser(description="GMLP")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--num-hops", type=int, default=5,
                        help="number of hops")
    parser.add_argument("--label-num-hops",type=int,default=9,
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
    #parser.add_argument("--batch-size", type=int, default=10000)
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
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="initial residual parameter for the model")
    parser.add_argument("--input-drop", type=float, default=0,
                        help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0.5,
                        help="attention dropout of model")
    parser.add_argument("--label-drop", type=float, default=0.5,
                        help="attention dropout of model")
    parser.add_argument("--pre-process", action='store_true', default=False,
                        help="whether to process the input features")
    parser.add_argument("--residual", action='store_true', default=False,
                        help="whether to process the input features")
    parser.add_argument("--act", type=str, default="relu",
                        help="the activation function of the model")
    parser.add_argument("--method", type=str, default="JK_GAMLP",
                        help="the model to use")
    parser.add_argument("--use-label", action='store_true', default=False,
                        help="whether to use the reliable data distillation")
    parser.add_argument("--train-num-epochs",type=int, default=100)
    parser.add_argument("--epochs", type=int, default=800,
                        help="The epoch setting for each stage.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    args=parser.parse_args()
    print(args)
    print("device:",device)
    
    #random.seed(args.seed)
    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)
    set_seed(args.seed)
    train_adj,val_adj,test_adj,train_feat,val_feat,test_feat,train_labels,val_labels, test_labels,train_nodes, val_nodes, test_nodes = load_ppi()
    
    train_feature_list=[]
    train_label_list=[]
    for i in range(len(train_adj)):
        train_feature_list.append([])
        train_feature_list[i].append(train_feat[i])
        train_label_list.append([])
        train_label_list[i].append(train_labels[i].float())
        for j in range(args.num_hops):
            train_propagated_fea = torch.spmm(train_adj[i],train_feature_list[i][-1])            
            train_feature_list[i].append(train_propagated_fea)
        for j in range(args.label_num_hops):
            train_label=torch.spmm(train_adj[i],train_label_list[i][-1])
            train_label_list[i].append(train_label)
    for i in range(len(train_label_list)):
        for j in range(len(train_label_list[i])):
             alpha=np.cos(j*np.pi/(args.label_num_hops*2))                                                                 
             train_label_list[i][j]=(1-alpha)*train_label_list[i][j]+alpha*train_label_list[i][-1]  
#train_label_list[i]=
    print("train_feature_list has been done")
    val_feature_list=[]
    val_label_list=[]
    for i in range(len(val_adj)):
        val_feature_list.append([])
        val_feature_list[i].append(val_feat[i])
        val_label_list.append([])
        val_label_list[i].append(val_labels[i].float())        
        for j in range(args.num_hops):
            val_propagated_fea = torch.spmm(val_adj[i],val_feature_list[i][-1])            
            val_feature_list[i].append(val_propagated_fea)
        for j in range(args.label_num_hops):
            val_label=torch.spmm(val_adj[i],val_label_list[i][-1])
            val_label_list[i].append(val_label)
    for i in range(len(val_label_list)):
        for j in range(len(val_label_list[i])):
             alpha=np.cos(j*np.pi/(args.label_num_hops*2))  
             val_label_list[i][j]=(1-alpha)*val_label_list[i][j]+alpha*val_label_list[i][-1]
    print("val_feature_list has been done")    
    test_feature_list=[]
    test_label_list=[]
    for i in range(len(test_adj)):
        test_feature_list.append([])
        test_feature_list[i].append(test_feat[i])
        test_label_list.append([])
        test_label_list[i].append(test_labels[i].float())
        for j in range(args.num_hops):
            test_propagated_fea = torch.spmm(test_adj[i],test_feature_list[i][-1])            
            test_feature_list[i].append(test_propagated_fea)
        for j in range(args.label_num_hops):
            test_label=torch.spmm(test_adj[i],test_label_list[i][-1])
            test_label_list[i].append(test_label)
    for i in range(len(test_label_list)):
        for j in range(len(test_label_list[i])):
             alpha=np.cos(j*np.pi/(args.label_num_hops*2))  
             test_label_list[i][j]=(1-alpha)*test_label_list[i][j]+alpha*test_label_list[i][-1]    
    print("test_feature_list has been done")    
    
    train_idx = torch.LongTensor(range(20))
    train_loader = Data.DataLoader(dataset=train_idx,batch_size=1,shuffle=True,num_workers=0)
    val_idx = torch.LongTensor(range(len(val_adj)))
    val_loader = Data.DataLoader(dataset=val_idx,batch_size=1,shuffle=True,num_workers=0)
    test_idx = torch.LongTensor(range(len(test_adj)))
    test_loader = Data.DataLoader(dataset=test_idx,batch_size=1,shuffle=True,num_workers=0)
    print(train_feat[0].shape[1])
    model=gen_model(args,train_feat[0].shape[1],train_labels[0].shape[1]).to(device)
    #model = GMLPppi(nfeat=train_feat[0].shape[1], nhid=args.hidden,
    #             nclass=train_labels[0].shape[1],
    #             dropout=args.dropout,pl=args.pl).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)                      
    record = {}
    loss_fcn = torch.nn.BCELoss()
    best=1e9
    count=0
    print("training begin")
    best
    from sklearn.metrics import f1_score
    for epoch in range(args.epochs):
        model.train()
        for step,batch in enumerate(train_loader):
            optimizer.zero_grad()
            #train_feature_list[batch[0]]= train_feature_list[batch[0]].to(device)
            train_features=train_feature_list[batch[0]]
            batch_labels=train_label_list[batch[0]]
            for i in range(len(train_features)):
                train_features[i]=train_features[i].to(device)
            for i in range(len(batch_labels)):
                batch_labels[i]=batch_labels[i].to(device)
            batch_label=train_labels[batch[0]].to(device)
            batch_nodes=train_nodes[batch]
            output_att= model(train_features,batch_labels)
            L1 = loss_fcn(output_att[:batch_nodes], batch_label[:batch_nodes].float())
            loss_train = L1 

            predict = np.where(output_att[:batch_nodes].data.cpu().numpy() > 0.5, 1, 0)
            acc_train = f1_score(batch_label[:batch_nodes].data.cpu().numpy(),predict, average='micro')           

            loss_train.backward()
            optimizer.step()
        model.eval()
        for step,batch in enumerate(val_loader):
            val_features=val_feature_list[batch[0]]
            batch_labels=val_label_list[batch[0]]
            for i in range(len(val_features)):
                val_features[i]=val_features[i].to(device)
            for i in range(len(batch_labels)):
                batch_labels[i]=batch_labels[i].to(device)
            loss_val,acc_val=evaluate(args.epochs,val_features,model,val_nodes[batch],val_labels[batch[0]].to(device),loss_fcn,batch_labels)        
        for step,batch in  enumerate(test_loader):    
            test_features=test_feature_list[batch[0]]
            batch_labels=test_label_list[batch[0]]
            for i in range(len(test_features)):
                test_features[i]=test_features[i].to(device)
            for i in range(len(batch_labels)):
                batch_labels[i]=batch_labels[i].to(device)
            loss_test,acc_test=evaluate(args.epochs,test_features,model,test_nodes[batch],test_labels[batch[0]].to(device),loss_fcn,batch_labels)     
        if loss_val<best:
            best=loss_val
            count=0
        else:
            count=count+1
        if count>args.patience:
            break
        
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),              
              'acc_val: {:.4f}'.format(acc_val.item()),
              'loss_test: {:.4f}'.format(loss_test.item()),                
              'acc_test: {:.4f}'.format(acc_test.item())
              )
        record[loss_val.item()] = acc_test.item()
    print("Optimization Finished!")
    bit_list = sorted(record.keys())
    #bit_list.reverse()
    for key in bit_list[:10]:
        value = record[key]
        print(key, value)    
        

