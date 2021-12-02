import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import DeepSets
from main_drug import BaselineDeepSetsFeatCat, CatEmbedDeepSets
from equivariant_layers import ops_1_to_1, ops_2_to_2, set_ops_3_to_3, set_ops_4_to_4
from equivariant_layers_expand import eops_1_to_1, eops_2_to_2, eset_ops_3_to_3, eset_ops_4_to_4
from eq_models import Net1to1, Net2to2, SetNet3to3, SetNet4to4, SetEq3to3

OPS_DISPATCH = {
    'expand': {
        1: eops_1_to_1,
        2: eops_2_to_2,
        3: eset_ops_3_to_3,
        4: eset_ops_4_to_4
    },
    'default': {
        1: ops_1_to_1,
        2: ops_2_to_2,
        3: set_ops_3_to_3,
        4: set_ops_4_to_4
    }
}

def reset_params(model):
    for p in model.parameters():
        if len(p.shape) > 1:
            torch.nn.init.xavier_uniform_(p)
        else:
            torch.nn.init.zeros_(p)

class Eq1Net(nn.Module):
    def __init__(self, nembed, embed_dim, layers, out_dim, ops_func=eops_1_to_1):
        super(Eq1Net, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(nembed, embed_dim)
        self.eq_net = Net1to1(layers, out_dim, ops_func=ops_func)
        self.out_net = nn.Linear(out_dim, 1)

    def forward(self, xcat, xfeat):
        x = self.embed(xcat)
        x = torch.cat([x, xfeat.unsqueeze(-1)], axis=-1)
        x = F.relu(self.eq_net(x)) # x should have dim bxnxd
        x = self.out_net(x.sum(dim=(-2)))
        return x

class Eq2Net(nn.Module):
    def __init__(self, nembed, embed_dim, layers, out_dim, ops_func=eops_2_to_2, dropout_prob=0.5, pool='sum'):
        super(Eq2Net, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(nembed, embed_dim)
        self.eq_net = Net2to2(layers, out_dim, ops_func=ops_func)
        self.out_net = nn.Linear(out_dim, 1)
        self.dropout_prob = dropout_prob
        if pool == 'sum':
            self.pool = torch.sum
        elif pool == 'mean':
            self.pool = torch.mean
        else:
             pool == torch.amax

    def forward(self, xcat, xfeat):
        x = self.embed(xcat)
        x = torch.cat([x, xfeat.unsqueeze(-1)], axis=-1)
        x = torch.einsum('bid,bjd->bdij', x, x)
        x = self.eq_net(x)
        x = F.relu(x)
        #x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x, dim=(-3, -2))
        x = F.relu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.out_net(x)
        return x

class Eq2NetSet(nn.Module):
    def __init__(self, nembed, embed_dim, hid_dim, ops_func=eops_2_to_2, dropout_prob=0.5, pool='mean', output='mlp'):
        super(Eq2NetSet, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(nembed, embed_dim)
        self.enc = nn.Sequential(
            nn.Linear(embed_dim + 1, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
 ,       )
        if output == 'linear':
            self.dec = nn.Linear(hid_dim, 1)
        elif output == 'mlp':
            self.dec = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, 1)
            )

        self.dropout_prob = dropout_prob
        if pool == 'sum':
            self.pool = torch.sum
        elif pool == 'mean':
            self.pool = torch.mean
        else:
             pool == torch.amax

    def forward(self, xcat, xfeat):
        x = self.embed(xcat)
        x = torch.cat([x, xfeat.unsqueeze(-1)], axis=-1)
        x = torch.einsum('bid,bjd->bdij', x, x)
        x = x.permute(0, 2, 3, 1)
        x = self.enc(x)
        x = F.relu(x)
        x = self.pool(x, dim=(-3, -2))
        x = F.relu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.dec(x)
        return x

class Eq3Net(nn.Module):
    def __init__(self, nembed, embed_dim, layers, out_dim, ops_func=eset_ops_3_to_3, dropout_prob=0.5, pool='sum'):
        super(Eq3Net, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(nembed, embed_dim)
        self.eq_net = SetNet3to3(layers, out_dim, ops_func=ops_func)
        self.out_net = nn.Linear(out_dim, 1)
        self.dropout_prob = dropout_prob
        if pool == 'sum':
            self.pool = torch.sum
        elif pool == 'mean':
            self.pool = torch.mean
        else:
             pool == torch.amax

    def forward(self, xcat, xfeat):
        x = self.embed(xcat)
        x = torch.cat([x, xfeat.unsqueeze(-1)], axis=-1)
        x = torch.einsum('bid,bjd,bkd->bdijk', x, x, x)
        x = self.eq_net(x)
        x = F.relu(x)
        #x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x, dim=(-2, -3, -4))
        x = F.relu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.out_net(x)
        return x

class Eq3NetMini(nn.Module):
    def __init__(self, nembed, embed_dim, hid_dim, ops_func=eset_ops_3_to_3, dropout_prob=0.5, pool='mean'):
        super(Eq3NetMini, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(nembed, embed_dim)
        self.eq_layer = SetEq3to3(embed_dim + 1, hid_dim, ops_func=ops_func)
        self.out_net = nn.Linear(hid_dim, 1)
        self.dropout_prob = dropout_prob
        if pool == 'sum':
            self.pool = torch.sum
        elif pool == 'mean':
            self.pool = torch.mean
        else:
             pool == torch.amax

    def forward(self, xcat, xfeat):
        x = self.embed(xcat)
        x = torch.cat([x, xfeat.unsqueeze(-1)], axis=-1)
        x = torch.einsum('bid,bjd,bkd->bdijk', x, x, x)
        x = self.eq_layer(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = F.relu(x)
        #x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x, dim=(-2, -3, -4))
        x = F.relu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.out_net(x)
        return x

class Eq3Set(nn.Module):
    def __init__(self, nembed, embed_dim, hid_dim, ops_func=eset_ops_3_to_3, dropout_prob=0.5, pool='mean', output='mlp'):
        super(Eq3Set, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(nembed, embed_dim)
        self.enc = nn.Sequential(
            nn.Linear(embed_dim + 1, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )
        if output == 'linear':
            self.dec = nn.Linear(hid_dim, 1)
        elif output == 'mlp':
            self.dec = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, 1)
            )

        self.dropout_prob = dropout_prob
        if pool == 'sum':
            self.pool = torch.sum
        elif pool == 'mean':
            self.pool = torch.mean
        else:
             pool == torch.amax

    def forward(self, xcat, xfeat):
        x = self.embed(xcat)
        x = torch.cat([x, xfeat.unsqueeze(-1)], axis=-1)
        x = torch.einsum('bid,bjd,bkd->bdijk', x, x, x)
        #x = self.eq_layer(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.enc(x)
        x = F.relu(x)
        #x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x, dim=(-2, -3, -4))
        x = F.relu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.dec(x)
        return x

class Eq4Net(nn.Module):
    def __init__(self, nembed, embed_dim, layers, out_dim, ops_func=eset_ops_4_to_4):
        super(Eq4Net, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(nembed, embed_dim)
        self.eq_net = SetNet4to4(layers, out_dim, ops_func=ops_func)
        self.out_net = nn.Linear(out_dim, 1)

    def forward(self, xcat, xfeat):
        x = F.relu(self.embed(xcat))
        x = torch.cat([x, xfeat.unsqueeze(-1)], axis=-1)
        x = torch.einsum('bid,bjd,bkd,bld->bdijkl', x, x, x, x)
        x = F.relu(self.eq_net(x))
        x = self.out_net(x.sum(dim=(-2, -3, -4, -5)))
        return x

class Eq2DeepSet(nn.Module):
    def __init__(self, nembed, embed_dim, hid_dim, out_dim, dropout_prob=0):
        super(Eq2DeepSet, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(nembed, embed_dim) # B x n x d
        self.fc1 = nn.Linear(embed_dim + 1, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.out_net = nn.Linear(out_dim, 1)
        self.dropout_prob = dropout_prob

    def forward(self, xcat, xfeat):
        x = F.relu(self.embed(xcat))
        x = torch.cat([x, xfeat.unsqueeze(-1)], axis=-1)
        x = torch.einsum('bid,bjd->bijd', x, x)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = x.sum(dim=(-2, -3))
        x = self.out_net(x)
        return x
