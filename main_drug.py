import pdb
import pickle
import time
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from eq_models import Eq1to2, Eq2to2

from drug_dataloader import PrevalenceDataset, PrevalenceCategoricalDataset
from dataloader import Dataset, DataWithMask, BowDataset
from models import DeepSets

class BaselineDeepSetsFeatCat(nn.Module):
    def __init__(self, nembed, embed_dim, hid_dim, out_dim, dropout_prob=0):
        super(BaselineDeepSetsFeatCat, self).__init__()
        self.embed = nn.Embedding(nembed, embed_dim)
        self.set_embed = DeepSets(embed_dim + 1, hid_dim, out_dim) # catted a feature
        self.fc_out = nn.Linear(out_dim, 1)
        self.dropout_prob = dropout_prob

    def forward(self, xcat, xfeat):
        embed = F.relu(self.embed(xcat))
        embed_catted = torch.cat([embed, xfeat.unsqueeze(-1)], axis=-1)
        set_embed = F.relu(self.set_embed(embed_catted))
        x = F.dropout(set_embed, self.dropout_prob, training=self.training)
        return self.fc_out(x)

    def pred_batch(self, batch, device):
        xcat, xfeat, _ = batch
        xcat = xcat.to(device)
        xfeat = xfeat.to(device)
        return self.forward(xcat, xfeat)

class CatEmbedDeepSets(nn.Module):
    def __init__(self, nembed, embed_dim, hid_dim):
        super(CatEmbedDeepSets, self).__init__()
        self.embed = nn.Embedding(nembed, embed_dim)
        self.set_embed = DeepSets(embed_dim, hid_dim, hid_dim) # catted a feature
        self.fc_out = nn.Linear(hid_dim, 1)

    def forward(self, xcat):
        embed = F.relu(self.embed(xcat))
        set_embed = F.relu(self.set_embed(embed))
        return self.fc_out(set_embed)

    def pred_batch(self, batch, device):
        xs, _ = batch
        xs = xs.to(device)
        return self.forward(xs)

def main(args):
    print(args)
    torch.random.manual_seed(args.seed)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    params = {'batch_size': args.batch_size, 'shuffle': True, 'pin_memory': True}

    if args.data== 'normal':
        train_data = PrevalenceDataset(args.train_fn)
        val_data = PrevalenceDataset(args.train_fn)
        test_data = PrevalenceDataset(args.test_fn)
        model = BaselineDeepSetsFeatCat(train_data.num_entities + 1, args.embed_dim, args.hid_dim).to(device)
    elif args.data == 'allcat':
        print('Using all categorical')
        train_data = PrevalenceCategoricalDataset(args.train_fn)
        val_data =   PrevalenceCategoricalDataset(args.train_fn)
        test_data =  PrevalenceCategoricalDataset(args.test_fn)
        model = CatEmbedDeepSets(train_data.num_entities + 1, args.embed_dim, args.hid_dim).to(device)

    train_dataloader = DataLoader(train_data, **params)
    val_dataloader = DataLoader(val_data, **params)
    test_dataloader = DataLoader(test_data, **params)

    loss_func = nn.MSELoss()
    opt= torch.optim.Adam(model.parameters(), lr=args.lr)
    nupdates = 0
    st = time.time()
    losses = []
    maes = []

    for e in range(args.epochs+ 1):
        #for xcat, xfeat, ybatch in train_dataloader:
        for batch in train_dataloader:
            opt.zero_grad()
            ybatch = batch[-1].to(device)
            ypred = model.pred_batch(batch, device)
            loss = loss_func(ybatch, ypred)
            batch_mae = (ypred - ybatch).abs().mean().item()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            maes.append(batch_mae)

        if nupdates % args.print_update == 0:
            tot_se = 0
            tot_ae = 0
            with torch.no_grad():
                #for _xcat, _xfeat, _ybatch in test_dataloader:
                for _batch in test_dataloader:
                    _ybatch = _batch[-1].to(device)
                    _ypred = model.pred_batch(_batch, device)
                    _loss  = loss_func(_ybatch, _ypred)
                    tot_se += (_loss.item() * len(_ybatch))
                    tot_ae += (_ypred - _ybatch).abs().sum().item()
                tot_mse = tot_se / len(test_data)
                tot_mae = tot_ae / len(test_data)
                print('Epoch: {:4d} | Num updates: {:5d} | Last 100 updates: {:.3f}, mse: {:.3f} | Tot test mae: {:.3f} | Tot test mse: {:.3f} | Time: {:.2f}mins'.format(
                    e, nupdates, np.mean(maes[-100:]), np.mean(losses[-100:]), tot_mae, tot_mse, (time.time() - st) / 60.
                ))
        nupdates += 1

    if args.save_fn:
        torch.save(model.state_dict(), args.save_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_pkl', type=str, default='./data/prevalence_dataset.pkl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--print_update', type=int, default=1000)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--train_fn', type=str, default='./data/prevalence/prevalence_train.csv')
    parser.add_argument('--test_fn', type=str, default='./data/prevalence/prevalence_test.csv')
    parser.add_argument('--data', type=str, default='normal')
    parser.add_argument('--save_fn', type=str, default='')
    args = parser.parse_args()
    main(args)
