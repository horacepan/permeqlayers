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
from eq_models import Eq1to2, Eq2to2, SetNet3to3, SetNet4to4
from main import BaselineEmbedDeepSets

from drug_dataloader import PrevalenceDataset, PrevalenceCategoricalDataset, gen_sparse_drug_data
from dataloader import Dataset, DataWithMask, BowDataset
from models import DeepSets
from main_drug import BaselineDeepSetsFeatCat, CatEmbedDeepSets

'''
Eq3Net and Eq4Net do the work of getting data into right format
- input data -> embed
- embedding -> cat the doseage
- construct kth order tensor from this by taking the outer product
- feed it through 3->3/4->4 layers
- linear layer
'''
class Eq3Net(nn.Module):
    def __init__(self, nembed, embed_dim, layers, out_dim=1):
        super(Eq3Net, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(nembed, embed_dim)
        self.eq_net = SetNet3to3(layers, out_dim)

    def forward(self, xcat, xfeat):
        x = F.relu(self.embed(xcat))
        x = torch.cat([x, xfeat.unsqueeze(-1)], axis=-1)
        x = torch.einsum('bid,bjd,bkd->bdijk', x, x, x)
        x = self.eq_net(x)
        return x

    def pred_batch(self, batch, device):
        xcat, xfeat, _ = batch
        xcat = xcat.to(device)
        xfeat = xfeat.to(device)
        return self.forward(xcat, xfeat)

class Eq4Net(nn.Module):
    def __init__(self, nembed, embed_dim, layers, out_dim=1):
        super(Eq4Net, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(nembed, embed_dim)
        self.eq_net = SetNet4to4(layers, out_dim)

    def forward(self, xcat, xfeat):
        x = F.relu(self.embed(xcat))
        x = torch.cat([x, xfeat.unsqueeze(-1)], axis=-1)
        x = torch.einsum('bid,bjd,bkd,bld->bdijkl', x, x, x, x)
        x = self.eq_net(x)
        return x

    def pred_batch(self, batch, device):
        xcat, xfeat, _ = batch
        xcat = xcat.to(device)
        xfeat = xfeat.to(device)
        return self.forward(xcat, xfeat)

def main(args):
    print(args)
    torch.random.manual_seed(args.seed)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    params = {'batch_size': args.batch_size, 'shuffle': True, 'pin_memory': True, 'num_workers': args.num_workers}
    layers = [(args.embed_dim+ 1, args.hid_dim)] + [(args.hid_dim, args.hid_dim) for _ in range(args.num_eq_layers - 1)]

    if args.data== 'sparse':
        print('Generating sparse data')
        train_data, test_data  = gen_sparse_drug_data(args.max_drugs, args.train_pct, seed=args.seed)
        print(f'Train size: {len(train_data)} | Test size: {len(test_data)}')
    else:
        train_data = PrevalenceDataset(args.train_fn)
        test_data = PrevalenceDataset(args.test_fn)
        print(f'Train size: {len(train_data)} | Test size: {len(test_data)}')

    if args.max_drugs == 3 and args.model == 'eq':
        print('3-3 model')
        model = Eq3Net(PrevalenceDataset.num_entities + 1, args.embed_dim, layers).to(device)
    elif args.max_drugs == 4 and args.model == 'eq':
        print('4-4 model')
        model = Eq4Net(PrevalenceDataset.num_entities + 1, args.embed_dim, layers).to(device)
    else:
        print('Baseline model')
        model = BaselineDeepSetsFeatCat(PrevalenceDataset.num_entities + 1,
                                        args.embed_dim,
                                        args.hid_dim
                                       ).to(device)

    train_dataloader = DataLoader(train_data, **params)
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
            nupdates += 1

        if e % args.print_update == 0:
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
                print('Epoch: {:4d} | Num updates: {:5d} | Last 100 updates: mae {:.3f}, mse: {:.3f} | Tot test mae: {:.3f} | Tot test mse: {:.3f} | Time: {:.2f}mins'.format(
                    e, nupdates, np.mean(maes[-100:]), np.mean(losses[-100:]), tot_mae, tot_mse, (time.time() - st) / 60.
                ))

    if args.save_fn:
        torch.save(model.state_dict(), args.save_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_pkl', type=str, default='./data/prevalence_dataset.pkl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--num_eq_layers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--print_update', type=int, default=1000)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--train_fn', type=str, default='./data/prevalence/prevalence_train.csv')
    parser.add_argument('--test_fn', type=str, default='./data/prevalence/prevalence_test.csv')
    parser.add_argument('--train_pct', type=float, default=0.8)
    parser.add_argument('--data', type=str, default='sparse')
    parser.add_argument('--save_fn', type=str, default='')
    parser.add_argument('--max_drugs', type=int, default=4)
    parser.add_argument('--model', type=str, default='baseline')
    args = parser.parse_args()
    main(args)
