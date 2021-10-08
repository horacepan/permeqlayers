import pdb
import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, Omniglot
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from equivariant_layers_expand import *
from eq_models import *
from models import MLP
from utils import setup_experiment_log, get_logger

def nparams(model):
    tot = 0
    for p in model.parameters():
        tot += p.numel()
    return tot

def load_train_test(dataset, root='./data/'):
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
    ])
    if dataset == 'mnist':
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        train = MNIST(root, train=True, download=True, transform=transform)
        test = MNIST(root, train=False, download=True, transform=transform)
    elif dataset == 'omniglot':
        normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206],
                                         std=[0.08426, 0.08426, 0.08426])
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        train = Omniglot(root, background=True, download=True, transform=transform)
        test = Omniglot(root, background=False, download=True, transform=transform)
    return train, test

class SetDataset(Dataset):
    def __init__(self, data, ys, set_size):
        self.data = data.view(len(data), -1)
        self.ys = ys
        self.set_size = set_size
        self.samples = torch.randint(0, len(self.data),
                                     size=(len(self.data), set_size))

    def __getitem__(self, idx):
        sample = self.samples[idx]
        items = torch.index_select(self.data, 0, sample)
        vals = torch.index_select(self.ys, 0, sample)
        nuniques = len(set(vals.tolist()))
        return items, nuniques

    def __len__(self):
        return len(self.data)

class MiniDeepSets(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MiniDeepSets, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        '''
        x: torch tensor of shape B x n x d
        '''
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = x.sum(dim=1)
        x = self.fc_out(x)
        return x

    def nparams(self):
        tot = 0
        for p in self.parameters():
            tot += p.numel()
        return tot

class UniqueEq2Net(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, ops_func=eops_2_to_2, dropout_prob=0):
        super(UniqueEq2Net, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU()
        )
        self.eq_net = Net2to2([(hid_dim, hid_dim)], out_dim, ops_func=ops_func)
        self.out_net = nn.Linear(out_dim, 1)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        x = self.embed(x)
        x = torch.einsum('bid,bjd->bdij', x, x)
        x = self.eq_net(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = x.sum(dim=(-3, -2))
        x = F.relu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.out_net(x)
        return x

class UniqueEq3Net(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, ops_func=eset_ops_3_to_3, dropout_prob=0):
        super(UniqueEq3Net, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU()
        )
        self.eq_net = SetNet3to3([(hid_dim, hid_dim)], out_dim, ops_func=ops_func)
        self.out_net = nn.Linear(out_dim, 1)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        x = self.embed(x)
        x = torch.einsum('bid,bjd,bkd->bdijk', x, x, x)
        x = self.eq_net(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = x.sum(dim=(-4, -3, -2))
        x = F.relu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.out_net(x)
        return x

def main(args):
    logfile, swr = setup_experiment_log(args, args.savedir, args.exp_name, args.save)
    savedir = os.path.join(args.savedir, args.exp_name)
    log = get_logger(logfile)

    train, test = load_train_test(args.dataset)
    train_data = SetDataset(train.train_data, train.train_labels, 10)
    test_data = SetDataset(test.test_data, test.test_labels, 10)
    train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True, pin_memory=args.pin, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_data, batch_size=256, shuffle=True, pin_memory=args.pin, num_workers=args.num_workers)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    in_dim = 784
    criterion = torch.nn.MSELoss()
    if args.model == 'baseline':
        model = MiniDeepSets(in_dim, args.hid_dim, out_dim=1).to(device)
    elif args.model == 'eq2':
        model = UniqueEq2Net(in_dim, args.hid_dim, args.out_dim, dropout_prob=args.dropout_prob).to(device)
    elif args.model == 'eq3':
        model = UniqueEq3Net(in_dim, args.hid_dim, args.out_dim, dropout_prob=args.dropout_prob).to(device)

    for p in model.parameters():
        if len(p.shape) == 1:
            torch.nn.init.zeros_(p)
        else:
            torch.nn.init.xavier_uniform_(p)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log.info('Starting training on: {}'.format(model.__class__))
    log.info('Model parameters: {}'.format(nparams(model)))
    model.train()

    for e in (range(args.epochs)):
        batch_losses = []
        for batch in train_dataloader:
            opt.zero_grad()
            x, y = batch
            x = x.float().to(device)
            y = y.float().to(device)
            ypred = model(x.float())
            loss = criterion(y, ypred.squeeze(-1))
            batch_losses.append(loss.item())
            loss.backward()
            opt.step()

        if e % args.print_update == 0:
            model.eval()
            tot_se = tot_ae = 0
            ncorrect = 0
            with torch.no_grad():
                for x, y in test_dataloader:
                    y = y.float().to(device)
                    ypred = model(x.float().to(device))
                    ypred = ypred.squeeze(-1)
                    loss = criterion(y.float().to(device), ypred)
                    tot_se += loss.item() * len(y)
                    tot_ae += (ypred - y).abs().sum().item()
                    ncorrect += (torch.round(ypred).int() == y.int()).sum().item()
            mse = tot_se / len(test_data)
            mae = tot_ae / len(test_data)
            nuniques = len(set(torch.round(ypred).int().view(-1).tolist()))
            tuniques = len(set(y.int().view(-1).tolist()))
            accuracy = ncorrect / len(test_data)
            log.info('Epoch {:4d} | Test MAE: {:.2f}, MSE: {:.2f} | acc: {:3f} | uniques: {}, true uniques: {}'.format(
                e, mae, mse, accuracy, nuniques, tuniques))

            model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, default='./results/unique/')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--out_dim', type=int, default=128)
    parser.add_argument('--num_eq_layers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--print_update', type=int, default=1000)
    parser.add_argument('--save_iter', type=int, default=5000)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin', action='store_true', default=False)
    parser.add_argument('--train_pct', type=float, default=0.8)
    parser.add_argument('--data', type=str, default='sparse')
    parser.add_argument('--save_fn', type=str, default='')
    parser.add_argument('--eqn', type=int, default=2)
    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--ops', type=str, default='expand')
    parser.add_argument('--dropout_prob', type=float, default=0)
    args = parser.parse_args()
    main(args)
