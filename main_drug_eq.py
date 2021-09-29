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
from main import BaselineEmbedDeepSets
from utils import get_logger, setup_experiment
from drug_dataloader import PrevalenceDataset, PrevalenceCategoricalDataset, gen_sparse_drug_data
from dataloader import Dataset, DataWithMask, BowDataset
from main_drug import BaselineDeepSetsFeatCat, CatEmbedDeepSets
from prevalence_models import Eq1Net, Eq2Net, Eq3Net, Eq4Net, Eq2DeepSet

def reset_params(model):
    for p in model.parameters():
        if len(p.shape) > 1:
            torch.nn.init.xavier_uniform_(p)
        else:
            torch.nn.init.zeros_(p)


def main(args):
    log = get_logger(args.logfile, args.stdout)
    log.info(args)
    torch.random.manual_seed(args.seed)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    if args.data== 'sparse':
        train_data, test_data  = gen_sparse_drug_data(args.max_drugs, args.train_pct, seed=args.seed)
    else:
        train_data = PrevalenceDataset(args.train_fn)
        test_data = PrevalenceDataset(args.test_fn)

    layers = [(args.embed_dim+ 1, args.hid_dim)] + [(args.hid_dim, args.hid_dim) for _ in range(args.num_eq_layers - 1)]
    if args.eqn == 1 and args.model == 'eq':
        model = Eq1Net(PrevalenceDataset.num_entities + 1, args.embed_dim, layers, args.out_dim).to(device)
    elif args.eqn == 2 and args.model == 'eq':
        model = Eq2Net(PrevalenceDataset.num_entities + 1, args.embed_dim, layers, args.out_dim, dropout_prob=args.dropout_prob).to(device)
    elif args.eqn == 3 and args.model == 'eq':
        model = Eq3Net(PrevalenceDataset.num_entities + 1, args.embed_dim, layers, args.out_dim, dropout_prob=args.dropout_prob).to(device)
    elif args.eqn == 4 and args.model == 'eq':
        model = Eq4Net(PrevalenceDataset.num_entities + 1, args.embed_dim, layers, args.out_dim).to(device)
    elif args.eqn == 2 and args.model == 'mlp':
        log.info('Making Eq2DeepSet')
        model = Eq2DeepSet(PrevalenceDataset.num_entities + 1, args.embed_dim, args.hid_dim, args.out_dim).to(device)
    else:
        log.info('Doing baseline with dropout')
        model = BaselineDeepSetsFeatCat(PrevalenceDataset.num_entities + 1,
                                        args.embed_dim,
                                        args.hid_dim,
                                        args.dropout_prob
                                       ).to(device)
    reset_params(model)
    params = {'batch_size': args.batch_size, 'shuffle': True, 'pin_memory': args.pin, 'num_workers': args.num_workers}
    train_dataloader = DataLoader(train_data, **params)
    test_dataloader = DataLoader(test_data, **params)
    loss_func = nn.MSELoss()
    opt= torch.optim.Adam(model.parameters(), lr=args.lr)
    st = time.time()
    losses = []
    maes = []
    nupdates = 0

    for e in range(args.epochs+ 1):
        _est = time.time()
        for batch in train_dataloader:
            for param in model.parameters():
                param.grad = None

            ybatch = batch[-1].to(device)
            ypred = model.pred_batch(batch, device)
            loss = loss_func(ybatch, ypred)
            batch_mae = (ypred - ybatch).abs().mean().item()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            maes.append(batch_mae)
            nupdates += 1
        _eend = time.time()
        etime = _eend - _est

        if e % args.print_update == 0:
            tot_se = 0
            tot_ae = 0
            with torch.no_grad():
                _tst = time.time()
                model.eval()
                for _batch in test_dataloader:
                    _ybatch = _batch[-1].to(device)
                    _ypred = model.pred_batch(_batch, device)
                    _loss  = loss_func(_ybatch, _ypred)
                    tot_se += (_loss.item() * len(_ybatch))
                    tot_ae += (_ypred - _ybatch).abs().sum().item()
                tot_mse = tot_se / len(test_data)
                tot_mae = tot_ae / len(test_data)
                _tend = time.time()
                _ttime = _tend - _tst
                log.info('Ep: {:4d} | Ups: {:5d} | Last 100 ups: mae {:.2f}, mse: {:.2f} | Tot test mae: {:.2f} | Tot test mse: {:.2f} | Time: {:.2f}mins | Ep time: {:.2f}s, T time: {:.2f}s'.format(
                    e, nupdates, np.mean(maes[-min(100, len(maes)):]), np.mean(losses[-min(100, len(losses)):]), tot_mae, tot_mse, (time.time() - st) / 60., etime, _ttime
                ))
            model.train()

    if args.save_fn:
        torch.save(model.state_dict(), args.save_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str, default='')
    parser.add_argument('--stdout', action='store_true', default=False)
    parser.add_argument('--train_pkl', type=str, default='./data/prevalence_dataset.pkl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--out_dim', type=int, default=128)
    parser.add_argument('--num_eq_layers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--print_update', type=int, default=1000)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--pin', action='store_true', default=False)
    parser.add_argument('--train_fn', type=str, default='./data/prevalence/prevalence_train.csv')
    parser.add_argument('--test_fn', type=str, default='./data/prevalence/prevalence_test.csv')
    parser.add_argument('--train_pct', type=float, default=0.8)
    parser.add_argument('--data', type=str, default='sparse')
    parser.add_argument('--save_fn', type=str, default='')
    parser.add_argument('--max_drugs', type=int, default=4)
    parser.add_argument('--eqn', type=int, default=2)
    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--ops', type=str, default='expand')
    parser.add_argument('--dropout_prob', type=float, default=0)
    args = parser.parse_args()
    main(args)
