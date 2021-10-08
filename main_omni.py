import pdb
import os
import argparse
from tqdm import tqdm
import numpy as np

from scipy.special import factorial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, Omniglot
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, BatchSampler, SequentialSampler

from conv import BasicConvNet
from equivariant_layers_expand import *
from eq_models import *
from models import MLP
from omni_data import OmniSetData
from utils import setup_experiment_log, get_logger, check_memory

def nparams(model):
    tot = 0
    for p in model.parameters():
        tot += p.numel()
    return tot

def neg_ll(pred, y, factorial_y):
    log_lik = y*torch.log(pred) - pred - torch.log(factorial_y)
    return -log_lik

class MiniDeepSets(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, **kwargs):
        super(MiniDeepSets, self).__init__()
        self.embed = BasicConvNet(in_dim, hid_dim, **kwargs)
        self.fc_out = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        '''
        x: torch tensor of shape B x set size x channel x h x w
        '''
        B, k, h, w = x.shape
        x = x.view(B * k, 1, h, w)
        x = self.embed(x)
        x = F.relu(x)
        x = x.view(B, k, x.shape[-1])
        x = x.sum(dim=1)
        x = self.fc_out(x)
        return x

class UniqueEq2Net(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, ops_func=eops_2_to_2, dropout_prob=0):
        super(UniqueEq2Net, self).__init__()
        self.embed = BasicConvNet(in_dim, hid_dim)
        self.eq_net = Net2to2([(hid_dim, hid_dim)], out_dim, ops_func=ops_func)
        self.out_net = nn.Linear(out_dim, 1)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        x = self.embed(x)
        x = F.relu(x)
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
        self.embed = BasicConvNet(in_dim, hid_dim, out_dim)
        self.eq_net = SetNet3to3([(hid_dim, hid_dim)], out_dim, ops_func=ops_func)
        self.out_net = nn.Linear(out_dim, 1)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        x = self.embed(x)
        x = F.relu(x)
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
    log.info('Starting experiment!')
    log.info('Args')
    log.info(args)

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    train_dataset = OmniSetData.from_files(args.idx_pkl, args.tgt_pkl, args.img_fn)
    test_dataset = OmniSetData.from_files(args.test_idx_pkl, args.test_tgt_pkl, imgs=train_dataset.imgs)
    train_dataloader = DataLoader(dataset=train_dataset,
        sampler=BatchSampler(
            SequentialSampler(train_dataset), batch_size=32, drop_last=False
        ),
        num_workers=args.num_workers, pin_memory=args.pin
    )
    test_dataloader = DataLoader(dataset=test_dataset,
        sampler=BatchSampler(
            SequentialSampler(test_dataset), batch_size=32, drop_last=False
        ),
        num_workers=args.num_workers, pin_memory=args.pin
    )

    log.info('Post load data: Consumed {:.2f}mb mem'.format(check_memory(False)))
    in_dim = 105
    if args.model == 'baseline':
        kwargs = {'nchannels': args.nchannels}
        model = MiniDeepSets(in_dim, args.hid_dim, out_dim=1, **kwargs)
        log.info('Post load model on cpu: Consumed {:.2f}mb mem'.format(check_memory(False)))
        for p in model.parameters():
            print(p.shape)
        model = model.to(device)
    elif args.model == 'eq2':
        model = UniqueEq2Net(in_dim, args.hid_dim, args.out_dim, dropout_prob=args.dropout_prob).to(device)
    elif args.model == 'eq3':
        model = UniqueEq3Net(in_dim, args.hid_dim, args.out_dim, dropout_prob=args.dropout_prob).to(device)
    log.info('Post model to device: Consumed {:.2f}mb mem'.format(check_memory(False)))

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
        ncorrect = 0
        tot = 0
        for batch in train_dataloader:
            opt.zero_grad()
            x, y = batch[0][0], batch[1][0] # batch sampler returns 1 x B x ...
            x = x.float().to(device)
            factorial_y = torch.FloatTensor(factorial(y)).to(device)
            y = y.to(device)
            ypred = model(x)

            loss = torch.sum(neg_ll(ypred.squeeze(-1), y, factorial_y))
            estimated = torch.round(ypred).int()
            ncorrect += torch.sum(estimated==y).item()
            tot += len(batch)
            batch_losses.append(loss.item())
            loss.backward()
            opt.step()

        epoch_acc = ncorrect / tot
        epoch_loss = np.mean(batch_losses)

        if e % args.print_update == 0:
            model.eval()
            val_loss = 0
            ncorrect = tot = 0

            with torch.no_grad():
                for x, y in test_dataloader:
                    x = x.float().to(device)
                    factorial_y = factorial(y).to(device)
                    y = y.float().to(device)

                    ypred = model(x)
                    loss = torch.sum(neg_ll(ypred, y, factorial_y))
                    ncorrect += (torch.round(ypred).int() == y.int()).sum().item()
                    val_loss += loss.item()
                    tot += len(x)

            acc = ncorrect / tot
            log.info('Epoch {:4d} | Last ep acc: {:.2f}, loss: {:.2f} | Test acc: {:.2f}, loss: {:.2f}'.format(
                     e, epoch_acc, epoch_loss, acc, val_loss))

            model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, default='./results/unique/')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--idx_pkl', type=str, default='./data/omniglot-py/unique_chars_dataset/train_idx_1.pkl')
    parser.add_argument('--tgt_pkl', type=str, default='./data/omniglot-py/unique_chars_dataset/train_targets_1.pkl')
    parser.add_argument('--test_idx_pkl', type=str, default='./data/omniglot-py/unique_chars_dataset/test_idx_1.pkl')
    parser.add_argument('--test_tgt_pkl', type=str, default='./data/omniglot-py/unique_chars_dataset/test_targets_1.pkl')
    parser.add_argument('--img_fn', type=str, default='./data/omniglot-py/train_images.npy')

    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--nchannels', type=int, default=12)
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
