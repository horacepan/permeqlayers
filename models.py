import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, nin, nhid, nout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(nin, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.fc3 = nn.Linear(nhid, nout)
        self.init_params()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def init_params(self):
        for p in self.parameters():
            if len(p.shape) > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)


class DeepSets(nn.Module):
    def __init__(self, nin, nhid, nout, pool='sum'):
        super(DeepSets, self).__init__()
        self.in_embed = MLP(nin, nhid, nhid)
        self.out_embed = nn.Linear(nhid, nout)
        if pool == 'sum':
            self.pool = torch.sum
        elif pool == 'max':
            self.pool = torch.amax
        elif pool == 'mean':
            self.pool = torch.mean

    def forward(self, x):
        x = self.in_embed(x)
        x = self.pool(x, dim=-1)
        return self.out_embed(x)

class PairSetEmbed(nn.Module):
    def __init__(self, feat_in, set_in, nhid, nout):
        super(PairSetEmbed, self).__init__()
        self.feat_embed = MLP(feat_in, nhid, nhid)
        self.set_embed = DeepSets(set_in, nhid, nhid)
        self.out_embed = nn.Linear(nhid + nhid, nout)

    def forward(self, feats, set_feats):
        x_feat = self.feat_embed(feats)
        x_set = self.set_embed(set_feats)
        x_cat = torch.hstack([x_feat, x_set])
        return self.out_embed(x_cat)

if __name__ == '__main__':
    n = 12
    k = 3
    feat_in = 17
    set_in = 21
    nhid = 13
    nout = 1
    xp = torch.rand(n, feat_in)
    xs = torch.rand(n, k, set_in)

    net = PairSetEmbed(feat_in, set_in, nhid, nout)
    x = net(xp, xs)
    print(x.shape)
