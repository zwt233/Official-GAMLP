from layer import *

class GMLP(nn.Module):
    def __init__(self, nfeat, hidden, nclass,num_hops,
                 dropout, input_drop,att_dropout,alpha,n_layers_1,n_layers_2,pre_process=False):
        super(GMLP, self).__init__()
        self.num_hops=num_hops
        self.prelu=nn.PReLU()
        if pre_process:
          self.lr_left1 = FeedForwardNetII(num_hops*hidden, hidden, hidden, n_layers_1-1, dropout,alpha)
          self.lr_att = nn.Linear(hidden + hidden, 1)
          self.lr_right1 = FeedForwardNetII(hidden, hidden, nclass, n_layers_2, dropout,alpha)
          self.fcs = nn.ModuleList([FeedForwardNet(nfeat, hidden, hidden , 2, dropout) for i in range(num_hops)])
        else:
          self.lr_left1 = FeedForwardNetII(num_hops*nfeat, hidden, hidden, n_layers_1-1, dropout,alpha)
          self.lr_att = nn.Linear(nfeat + hidden, 1)
          self.lr_right1 = FeedForwardNetII(nfeat, hidden, nclass, n_layers_2, dropout,alpha)
        self.lr_left2 = nn.Linear(hidden, nclass)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop=nn.Dropout(att_dropout)
        self.pre_process=pre_process
        self.res_fc=nn.Linear(nfeat,hidden,bias=False)
        self.norm=nn.BatchNorm1d(hidden)
    def forward(self, feature_list):
        num_node=feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        hidden_list=[]
        if self.pre_process:
            for i in range(len(feature_list)):
                hidden_list.append(self.fcs[i](feature_list[i]))
        concat_features = torch.cat(hidden_list, dim=1)
        left_1 =self.dropout(self.prelu(self.lr_left1(concat_features)))
        left_2 = self.lr_left2(left_1)
        attention_scores = [torch.sigmoid(self.lr_att(torch.cat((left_1, x), dim=1))).view(num_node, 1) for x in
                            hidden_list]
        W = torch.cat(attention_scores, dim=1)
        W = F.softmax(W, 1)
        right_1 = torch.mul(hidden_list[0], self.att_drop(W[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            right_1 = right_1 + torch.mul(hidden_list[i],self.att_drop(W[:, i].view(num_node, 1)))
        #right_1 += self.res_fc(feature_list[0])
        #right_1= self.norm(right_1)
        right_1=self.dropout(self.prelu(right_1))
        right_1 = self.lr_right1(right_1)
        return right_1,left_2


class WeightedAggregator(nn.Module):
        def __init__(self, subset_list, in_feats, num_hops):
            super(WeightedAggregator, self).__init__()
            self.num_hops = num_hops
            self.subset_list =subset_list
            self.agg_feats = nn.ParameterList()
            for _ in range(num_hops):
                self.agg_feats.append(nn.Parameter(torch.Tensor(len(subset_list), in_feats)))
                nn.init.xavier_uniform_(self.agg_feats[-1])
        def forward(self, feats_dict):
            new_feats = []
            for k in range(self.num_hops):
                feats = torch.cat([feats_dict[rel_subset][k].unsqueeze(1) for rel_subset in self.subset_list], dim=1)
                new_feats.append((feats * self.agg_feats[k].unsqueeze(0)).sum(dim=1))
            return new_feats

class NARS_gmlp(nn.Module):
       def __init__(self, in_feats, hidden, out_feats, label_in_feats, num_hops, multihop_layers, n_layers, num_heads, subset_list, clf="sagn", relu="relu", batch_norm=True,dropout=0.5, input_drop=0.0, attn_drop=0.0, negative_slope=0.2, last_bias=False, use_labels=False, use_features=True):
           super(NARS_gmlp, self).__init__()
           self.aggregator = WeightedAggregator(subset_list, in_feats, num_hops)
           if clf == "sagn":
                self.clf = SAGN(in_feats, hidden, out_feats, label_in_feats,num_hops, multihop_layers, n_layers, num_heads, relu=relu, batch_norm=batch_norm, dropout=dropout, input_drop=input_drop, attn_drop=attn_drop,negative_slope=negative_slope,use_labels=use_labels, use_features=use_features)
           if clf == "sign":
                self.clf = SIGN(in_feats, hidden, out_feats, label_in_feats, num_hops, n_layers, dropout, input_drop,use_labels=use_labels)
           if clf == "gmlp":
                self.clf=GMLP(in_feats,label_in_feats, hidden, out_feats,num_hops,dropout, input_drop,attn_drop,0.5,4,6,4,pre_process=True)
       def forward(self, feats_dict, label_emb):
           feats = self.aggregator(feats_dict)
           out1,out2 = self.clf(feats, label_emb)
           return out1,out2
