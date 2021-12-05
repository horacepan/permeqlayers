import time
import sys
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
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler, SequentialSampler, SubsetRandomSampler
from torch_sigma_m import torch_sigma

from conv import BasicConvNet
from equivariant_layers_expand import *
from eq_models import *
from models import MLP
from omni_data import OmniSetData, StochasticOmniSetData
from utils import setup_experiment_log, get_logger, check_memory
from modules import SAB, PMA

def nparams(model):
    tot = 0
    for p in model.parameters():
        tot += p.numel()
    return tot

def save_checkpoint(epoch, model, optimizer, fname):
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, fname)

def load_checkpoint(model, optimizer, log, filename='checkpoint.pth.tar'):
    '''
    model: nn.Module object
    optimizer: torch.optim object
    log: log object
    filename: str file name of the checkpoint
    '''
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        log.info("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        log.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
        success = True
    else:
        log.info("=> no checkpoint found at '{}'".format(filename))
        success = False

    return model, optimizer, start_epoch, success

def try_load_weights(savedir, model, device, saved=True):
    if not saved:
        return False, 0

    files = os.listdir(savedir)
    models = [f[f.index('model'):] for f in files if 'model_' in f and 'final' not in f]
    if len(models) == 0:
        return False, 0

    max_ep = max([int(f[6:f.index('.')]) for f in models])
    fname = os.path.join(savedir, f'model_{max_ep}.pt')
    sd = torch.load(fname, map_location=device)

    model.load_state_dict(sd)
    return True, max_ep

def neg_ll(pred, y, factorial_y):
    log_lik = y*torch.log(pred) - pred - torch.log(factorial_y)
    return -log_lik

class MiniDeepSets(nn.Module):
    def __init__(self, in_dim, enc_dim, dec_dim, **kwargs):
        super(MiniDeepSets, self).__init__()
        self.embed = BasicConvNet(in_dim, enc_dim)
        self.dec = nn.Sequential(
            nn.Linear(enc_dim, dec_dim),
            nn.ReLU(),
            #nn.Dropout(kwargs['dropout']),
            nn.Linear(dec_dim, dec_dim),
            nn.ReLU(),
            #nn.Dropout(kwargs['dropout']),
            nn.Linear(dec_dim, dec_dim)
        )
        self.fc1 = nn.Linear(dec_dim, dec_dim)
        self.fc2 = nn.Linear(dec_dim, 1)

    def forward(self, x):
        '''
        x: torch tensor of shape B x set size x channel x h x w
        '''
        B, k, h, w = x.shape
        x = x.view(B * k, 1, h, w)
        x = self.embed(x)
        x = F.relu(x)
        x = self.dec(x) # (B*k) x d
        x = x.view(B, k, x.shape[-1])
        x = x.sum(dim=1)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class UniqueEq12Net(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout_prob=0, nlayers=1, **kwargs):
        super(UniqueEq12Net, self).__init__()
        self.embed = BasicConvNet(in_dim, hid_dim, **kwargs)
        #self.eq1_net = Net1to2([(hid_dim, hid_dim)] * nlayers, out_dim, ops_func=eops_1_to_2)
        self.eq1_net = Eq1to2(hid_dim, hid_dim)
        self.eq2_net = Eq2to2(hid_dim, out_dim)
        #self.eq1_net = Net1to2([(hid_dim, hid_dim)] * nlayers, out_dim, ops_func=eops_1_to_2)
        #self.eq2_net = Net2to2([(hid_dim, hid_dim)] * nlayers, out_dim, ops_func=eops_2_to_2)
        self.out_net = nn.Linear(out_dim, 1)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        B, k, h, w = x.shape
        x = x.view(B * k, 1, h, w)
        x = self.embed(x)
        x = F.relu(x)
        x = x.view(B, k, x.shape[-1]).permute(0, 2, 1)
        x = self.eq1_net(x)
        x = F.relu(x)
        x = self.eq2_net(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = x.sum(dim=(-2, -1))
        x = F.relu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.out_net(x)
        return x


class UniqueEq2Net(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout_prob=0, nlayers=1, **kwargs):
        super(UniqueEq2Net, self).__init__()
        self.embed = BasicConvNet(in_dim, hid_dim, **kwargs)
        #self.eq_net = Net2to2([(hid_dim, hid_dim)] * nlayers, out_dim, ops_func=ops_func)
        self.eq_net = Eq2to2(hid_dim, out_dim)
        self.out_net = nn.Linear(out_dim, 1)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        B, k, h, w = x.shape
        x = x.view(B * k, 1, h, w)
        x = self.embed(x)
        #x = F.relu(x)
        x = x.view(B, k, x.shape[-1])
        x = torch.einsum('bid,bjd->bdij', x, x)
        x = self.eq_net(x)
        x = F.relu(x)
        #x = F.dropout(x, self.dropout_prob, training=self.training)
        x = x.mean(dim=(-2, -1))
        x = F.relu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.out_net(x)
        return x

class UniqueEq3NetMini(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, ops_func=eset_ops_3_to_3, dropout_prob=0):
        super(UniqueEq3NetMini, self).__init__()
        self.embed = BasicConvNet(in_dim, hid_dim)
        self.eq_net = SetEq3to3(hid_dim, out_dim, ops_func=ops_func)
        self.out_net = nn.Linear(out_dim, 1)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        B, k, h, w = x.shape
        x1 = x.view(B * k, 1, h, w)
        x2 = self.embed(x1)
        #x3 = F.relu(x2)
        #x4 = x3.view(B, k, x3.shape[-1])
        x4 = x2.view(B, k, x2.shape[-1])
        x5 = torch.einsum('bid,bjd,bkd->bdijk', x4, x4, x4)
        x6 = self.eq_net(x5)
        x6 = x6.permute(0, 2, 3, 4, 1)
        x7 = F.relu(x6)
        #x = F.dropout(x, self.dropout_prob, training=self.training)
        x8 = x7.mean(dim=(-4, -3, -2))
        x9 = F.relu(x8)
        x10 = F.dropout(x9, self.dropout_prob, training=self.training)
        x11 = self.out_net(x10)
        return x11

class UniqueEq3Net(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, ops_func=eset_ops_3_to_3, dropout_prob=0):
        super(UniqueEq3Net, self).__init__()
        self.embed = BasicConvNet(in_dim, hid_dim)
        self.eq_net = SetNet3to3([(hid_dim, hid_dim)], out_dim, ops_func=ops_func)
        self.out_net = nn.Linear(out_dim, 1)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        B, k, h, w = x.shape
        x1 = x.view(B * k, 1, h, w)
        x2 = self.embed(x1)
        #x3 = F.relu(x2)
        #x4 = x3.view(B, k, x3.shape[-1])
        x4 = x2.view(B, k, x2.shape[-1])
        x5 = torch.einsum('bid,bjd,bkd->bdijk', x4, x4, x4)
        x6 = self.eq_net(x5)
        x7 = F.relu(x6)
        #x = F.dropout(x, self.dropout_prob, training=self.training)
        x8 = x7.mean(dim=(-4, -3, -2))
        x9 = F.relu(x8)
        x10 = F.dropout(x9, self.dropout_prob, training=self.training)
        x11 = self.out_net(x10)
        return x11
        #return x

class MiniSetTransformer(nn.Module):
    def __init__(self, hid_dim=64):
        super(MiniSetTransformer, self).__init__()
        self.embed = BasicConvNet(105, hid_dim)
        self.enc = nn.Sequential(
            SAB(dim_in=hid_dim, dim_out=hid_dim, num_heads=4),
            SAB(dim_in=hid_dim, dim_out=hid_dim, num_heads=4)
        )

        self.dec = nn.Sequential(
            PMA(dim=hid_dim, num_heads=8, num_seeds=1),
            nn.Linear(hid_dim, 1)
        )
    def forward(self, x):
        B, k, h, w = x.shape
        x = x.view(B * k, 1, h, w)
        x = self.embed(x)
        x = F.relu(x)
        x = x.view(B, k, x.shape[-1])
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-1)

def set_seeds(s):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    np.random.seed(s)

def make_exp_name(args):
    name = '{}_{}h_{}o_{}seed'.format(args.model, args.hid_dim, args.out_dim, args.seed)
    if args.lr_decay:
        name += '_lrdecay'
    return name

def main(args):
    exp_name = make_exp_name(args)
    logfile, swr = setup_experiment_log(args, args.savedir, exp_name, args.save)
    savedir = os.path.join(args.savedir, exp_name)
    log = get_logger(logfile)
    set_seeds(args.seed)
    log.info('Starting experiment! Saving in: {}'.format(exp_name))
    log.info('Command line:')
    log.info('python ' + ' '.join(sys.argv))
    log.info(args)

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206],
                                         std=[0.08426, 0.08426, 0.08426])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    omni = Omniglot(root='./data/', transform=transform, background=True, download=True)
    if args.datatype == 'stochastic':
        train_dataset = StochasticOmniSetData(omni, epoch_len=12800, max_set_length=10, min_set_length=6)
        test_dataset = StochasticOmniSetData(omni, epoch_len=6400, max_set_length=args.test_max_set_len, min_set_length=args.test_min_set_len)
    elif args.datatype == 'fixed':
        log.info('Loading fixed splits')
        train_dataset = OmniSetData.from_files(args.idx_pkl, args.tgt_pkl, omni, args.train_fraction)
        test_dataset = OmniSetData.from_files(args.test_idx_pkl, args.test_tgt_pkl, omni)
        log.info('Train data size: {} | Test data size: {}'.format(len(train_dataset), len(test_dataset)))
    log.info('Test data is: {}'.format(test_dataset))
    train_dataloader = DataLoader(dataset=train_dataset,
        sampler=BatchSampler(
            #SequentialSampler(train_dataset), batch_size=args.batch_size, drop_last=False
            RandomSampler(train_dataset), batch_size=args.batch_size, drop_last=False
        ),
        num_workers=args.num_workers, pin_memory=args.pin
    )
    test_dataloader = DataLoader(dataset=test_dataset,
        sampler=BatchSampler(
            RandomSampler(test_dataset), batch_size=args.batch_size, drop_last=False
        ),
        num_workers=args.num_workers, pin_memory=args.pin
    )

    log.info('Post load data: Consumed {:.2f}mb mem'.format(check_memory(False)))
    in_dim = 105
    if args.model == 'baseline':
        kwargs = {'nchannels': args.nchannels, 'conv_layers': args.conv_layers, 'dropout': args.dropout_prob}
        model = MiniDeepSets(in_dim, args.hid_dim, args.out_dim, **kwargs).to(device)
    elif args.model == 'eq12':
        kwargs = {'nchannels': args.nchannels, 'dropout': args.dropout_prob, 'conv_layers': args.conv_layers, 'dropout': args.conv_dropout}
        model = UniqueEq12Net(in_dim, args.hid_dim, args.out_dim, dropout_prob=args.dropout_prob, nlayers=args.num_eq_layers, **kwargs)
    elif args.model == 'eq2':
        kwargs = {'nchannels': args.nchannels, 'dropout': args.dropout_prob, 'conv_layers': args.conv_layers, 'dropout': args.conv_dropout}
        model = UniqueEq2Net(in_dim, args.hid_dim, args.out_dim, dropout_prob=args.dropout_prob, nlayers=args.num_eq_layers, **kwargs)
    elif args.model == 'eq3':
        model = UniqueEq3Net(in_dim, args.hid_dim, args.out_dim, dropout_prob=args.dropout_prob)
    elif args.model == 'eq3mini':
        model = UniqueEq3NetMini(in_dim, args.hid_dim, args.out_dim, dropout_prob=args.dropout_prob)
    elif args.model == 'dmps':
        model = torch_sigma()
    elif args.model == 'set':
        model = MiniSetTransformer(args.hid_dim)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model, opt, start_epoch, load_success = load_checkpoint(model, opt, log, os.path.join(savedir, 'checkpoint.pth'))
    if not load_success:
        log.info('Nothing to load. Init weights with: {}'.format(args.init_method))
        for p in model.parameters():
            if len(p.shape) == 1:
                torch.nn.init.zeros_(p)
            else:
                if args.init_method == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(p)
                elif args.init_method == 'xavier_normal':
                    torch.nn.init.xavier_normal_(p)
        start_epoch = 0
    else:
        log.info(f'Loaded weights from checkpoint in {savedir}')

    if args.lr_decay:
        log.info('Doing lr decay: factor {}, patience: {}'.format(args.lr_factor, args.lr_patience))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=args.lr_factor, patience=args.lr_patience)

    log.info('Starting training on: {}'.format(model.__class__))
    log.info('Model parameters: {}'.format(nparams(model)))
    out_func = F.softplus if args.out_func == 'softplus' else torch.exp
    log.info('Output function: {}'.format(out_func))
    model.train()

    for e in range(start_epoch, start_epoch + args.epochs+ 1):
        batch_losses = []
        ncorrect = 0
        tot_ae = 0
        tot = 0
        bcnt = 0
        st = time.time()
        for batch in (train_dataloader):
            opt.zero_grad()
            x, y = batch[0][0], batch[1][0] # batch sampler returns 1 x B x ...
            x = x.float().to(device)
            factorial_y = torch.FloatTensor(factorial(y)).to(device)
            y = y.to(device)
            ypred = out_func(model(x)).squeeze(-1)

            loss = torch.sum(neg_ll(ypred, y, factorial_y))
            estimated = torch.round(ypred).int()
            tot_ae += torch.abs(ypred - y).sum().item()
            ncorrect += torch.sum(estimated==y).item()
            tot += len(x)
            batch_losses.append(loss.item())
            if torch.isnan(loss):
                for p in model.parameters():
                    p.data *= 0.8
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            bcnt += 1
        end = time.time()
        minibatchtime = (end - st) / bcnt
        epoch_mae = tot_ae / tot
        epoch_acc = ncorrect / tot
        epoch_loss = np.mean(batch_losses)
        if swr:
            swr.add_scalar('train/acc', epoch_acc, e)
            swr.add_scalar('train/mae', epoch_mae, e)
            swr.add_scalar('train/loss', epoch_loss, e)

        if e % args.print_update == 0:
            model.eval()
            val_losses = []
            aes = []
            ncorrect = tot = 0

            with torch.no_grad():
                for batch in (test_dataloader):
                    x, y = batch[0][0], batch[1][0]
                    x = x.float().to(device)
                    factorial_y = torch.FloatTensor(factorial(y)).to(device)
                    y = y.float().to(device)
                    ypred = out_func(model(x)).squeeze(-1)

                    loss = torch.sum(neg_ll(ypred, y, factorial_y))
                    estimated = torch.round(ypred).int()
                    ncorrect += (estimated == y.int()).sum().item()
                    val_losses.append(loss.item())
                    aes.extend(torch.abs(ypred - y).tolist())
                    tot += len(x)

            acc = ncorrect / tot
            mae = np.mean(aes)
            std_ae = np.std(aes)
            if swr:
                swr.add_scalar('test/acc', acc, e)
                swr.add_scalar('test/mae', mae, e)
                swr.add_scalar('test/std_ae', std_ae, e)
            log.info('Epoch {:4d} | Last ep acc: {:.4f}, mae: {:.4f}, loss: {:.2f} | Test acc: {:.4f}, mae: {:.4f}, std mae: {:.3f} | mb time: {:.5f}s'.format(
                     e, epoch_acc, epoch_mae, epoch_loss, acc, mae, std_ae, minibatchtime))
            model.train()
            if args.lr_decay:
                scheduler.step(np.mean(val_losses))

        if e % args.save_iter == 0 and e > 0 and args.save:
            checkpoint_fn = os.path.join(savedir, 'checkpoint.pth')
            save_checkpoint(e, model, opt, checkpoint_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, default='./results/unique/')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--idx_pkl', type=str, default='./data/omniglot-py/unique_chars_dataset/train_idx_1.pkl')
    parser.add_argument('--tgt_pkl', type=str, default='./data/omniglot-py/unique_chars_dataset/train_targets_1.pkl')
    parser.add_argument('--test_idx_pkl', type=str, default='./data/omniglot-py/unique_chars_dataset/test_idx_2.pkl')
    parser.add_argument('--test_tgt_pkl', type=str, default='./data/omniglot-py/unique_chars_dataset/test_targets_2.pkl')
    parser.add_argument('--img_fn', type=str, default='./data/omniglot-py/train_images.npy')

    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nchannels', type=int, default=12)
    parser.add_argument('--conv_layers', type=int, default=4)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--out_dim', type=int, default=128)
    parser.add_argument('--num_eq_layers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--print_update', type=int, default=1)
    parser.add_argument('--save_iter', type=int, default=10)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin', action='store_true', default=False)
    parser.add_argument('--datatype', type=str, default='stochastic')
    parser.add_argument('--save_fn', type=str, default='')
    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--ops', type=str, default='expand')
    parser.add_argument('--dropout_prob', type=float, default=0)
    parser.add_argument('--conv_dropout', type=float, default=0)
    parser.add_argument('--out_func', type=str, default='softplus')
    parser.add_argument('--lr_decay', action='store_true', default=False)
    parser.add_argument('--clip_grad', type=float, default=0.5)
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--lr_patience', type=int, default=3)
    parser.add_argument('--train_fraction', type=float, default=1)
    parser.add_argument('--init_method', type=str, default='xavier_uniform')
    parser.add_argument('--test_min_set_len', type=int, default=6)
    parser.add_argument('--test_max_set_len', type=int, default=10)
    args = parser.parse_args()
    main(args)
