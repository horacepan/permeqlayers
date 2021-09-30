import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import DeepSets
from equivariant_layers_expand import eops_1_to_1, eops_2_to_2, eset_ops_3_to_3, eset_ops_4_to_4
from eq_models import Net1to1, Net2to2, SetNet3to3, SetNet4to4

class BaselineEmbedDeepSets(nn.Module):
    def __init__(self, nembed, embed_dim, hid_dim, num_classes=2):
        super(BaselineEmbedDeepSets, self).__init__()
        self.embed = nn.Embedding(nembed, embed_dim)
        self.set_embed1 = DeepSets(embed_dim, hid_dim, hid_dim)
        self.set_embed2 = DeepSets(embed_dim, hid_dim, hid_dim)
        self.fc_out = nn.Linear(2 * hid_dim, num_classes)

    def forward(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        #embed1 = F.relu(self.embed(x1))
        #embed2 = F.relu(self.embed(x2))
        embed1 = self.embed(x1)
        embed2 = self.embed(x2)

        set1 = F.relu(self.set_embed1(embed1))
        set2 = F.relu(self.set_embed1(embed2))
        sets = torch.hstack([set1, set2])
        return self.fc_out(sets)

class Dota2Eq2Embed(nn.Module):
    def __init__(self, nembed, embed_dim, hid_dim, out_dim, num_layers=1, num_classes=2, dropout_prob=0.5):
        super(Dota2Eq2Embed, self).__init__()
        layers = [(embed_dim, hid_dim)] + [(hid_dim, hid_dim) for _ in range(num_layers - 1)]
        self.embed = nn.Embedding(nembed, embed_dim)
        self.eq2_team1 = Net2to2(layers, out_dim, ops_func=eops_2_to_2)
        self.eq2_team2 = Net2to2(layers, out_dim, ops_func=eops_2_to_2)
        self.eq2_pair = Net2to2(layers, out_dim, ops_func=eops_2_to_2)
        #self.fc_out = nn.Linear(2 * out_dim, num_classes)
        self.fc_out = nn.Linear(3 * out_dim, num_classes)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        #e1 = F.relu(self.embed(x1))
        #e2 = F.relu(self.embed(x2))
        e1 = self.embed(x1)
        e2 = self.embed(x2)

        ep1 = torch.einsum('bid,bjd->bdij', e1, e1)
        ep2 = torch.einsum('bid,bjd->bdij', e2, e2)
        ep3 = torch.einsum('bid,bjd->bdij', e1, e2)

        t1 = self.eq2_team1(ep1)
        t2 = self.eq2_team2(ep2)
        t3 = self.eq2_pair(ep3)

        t1 = F.relu(t1)
        t1 = F.dropout(t1, self.dropout_prob, training=self.training)
        t2 = F.relu(t2)
        t2 = F.dropout(t2, self.dropout_prob, training=self.training)
        t3 = F.relu(t3)
        t3 = F.dropout(t3, self.dropout_prob, training=self.training)

        t1_embed = t1.mean(dim=(-2, -3))
        t2_embed = t2.mean(dim=(-2, -3))
        t3_embed = t3.mean(dim=(-2, -3))
        #t12_embed = torch.hstack([t1_embed, t2_embed])
        t12_embed = torch.hstack([t1_embed, t2_embed, t3_embed])
        t12_embed = F.dropout(t12_embed)
        return self.fc_out(t12_embed)

class Dota2Eq3Embed(nn.Module):
    def __init__(self, nembed, embed_dim, hid_dim, out_dim, num_layers=1, num_classes=2, dropout_prob=0):
        super(Dota2Eq3Embed, self).__init__()
        layers = [(embed_dim, hid_dim)] + [(hid_dim, hid_dim) for _ in range(num_layers - 1)]
        self.embed = nn.Embedding(nembed, embed_dim)
        self.eq3_team1 = SetNet3to3(layers, out_dim, ops_func=eset_ops_3_to_3)
        self.eq3_team2 = SetNet3to3(layers, out_dim, ops_func=eset_ops_3_to_3)
        self.fc_out = nn.Linear(2 * out_dim, num_classes)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        e1 = F.relu(self.embed(x1))
        e2 = F.relu(self.embed(x2))
        e1 = torch.einsum('bid,bjd,bkd->bdijk', e1, e1, e1)
        e2 = torch.einsum('bid,bjd,bkd->bdijk', e2, e2, e2)
        t1 = self.eq3_team1(e1)
        t2 = self.eq3_team2(e2)
        t1_embed = F.relu(t1.mean(dim=(-2, -3, -4)))
        t2_embed = F.relu(t2.mean(dim=(-2, -3, -4)))
        t12_embed = torch.hstack([t1_embed, t2_embed])
        t12_embed = F.dropout(t12_embed, self.dropout_prob, training=self.training)
        return self.fc_out(t12_embed)

if __name__ == '__main__':
    nembed = 120
    embed_dim = 32
    hid_dim = 32
    out_dim = 64
    batch = 32
    team = 5
    net = Dota2Eq3Embed(nembed, embed_dim, hid_dim, out_dim)
    x = torch.randint(0, nembed-1, size=(batch, 2, team))
    res = net(x)
    print(res.shape)
