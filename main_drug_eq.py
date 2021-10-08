import pdb
import pickle
import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import get_logger, setup_experiment_log
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

def try_load_weights(savedir, model, device):
    files = os.listdir(savedir)
    models = [f[f.index('model'):] for f in files if 'model_' in f and 'final' not in f]
    if len(models) == 0:
        return False, 0

    max_ep = max([int(f[6:f.index('.')]) for f in models])
    fname = os.path.join(savedir, f'model_{max_ep}.pt')
    sd = torch.load(fname, map_location=device)

    model.load_state_dict(sd)
    return True, max_ep

def pred_batch(model, batch, device):
    xcat, xfeat, _ = batch
    xcat = xcat.to(device)
    xfeat = xfeat.to(device)
    return model.forward(xcat, xfeat)

def main(args):
    logfile, swr = setup_experiment_log(args, args.savedir, args.exp_name, args.save)
    savedir = os.path.join(args.savedir, args.exp_name)
    log = get_logger(logfile)
    log.info(args)
    log.info(f'Saving in: {savedir}')
    torch.random.manual_seed(args.seed)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    if args.data== 'sparse':
        train_data, test_data = gen_sparse_drug_data(args.max_drugs, args.train_pct, seed=args.seed)
    else:
        train_data = PrevalenceDataset(args.train_fn)
        test_data = PrevalenceDataset(args.test_fn)

    layers = [(args.embed_dim+ 1, args.hid_dim)] + [(args.hid_dim, args.hid_dim) for _ in range(args.num_eq_layers - 1)]
    if args.eqn == 1 and args.model == 'eq':
        model = Eq1Net(PrevalenceDataset.num_entities + 1, args.embed_dim, layers, args.out_dim).to(device)
    elif args.eqn == 2 and args.model == 'eq':
        log.info('Eq2net: {} embed, {} layers, {} final linear, {:.1f} dropout, pool mode {}'.format(
            args.embed_dim, layers, args.out_dim, args.dropout_prob, args.pool
        ))
        model = Eq2Net(PrevalenceDataset.num_entities + 1, args.embed_dim, layers, args.out_dim, dropout_prob=args.dropout_prob, pool=args.pool).to(device)
    elif args.eqn == 3 and args.model == 'eq':
        log.info('Eq3net: {} embed, {} layers, {} final linear, {:.1f} dropout, pool mode {}'.format(
            args.embed_dim, layers, args.out_dim, args.dropout_prob, args.pool
        ))
        model = Eq3Net(PrevalenceDataset.num_entities + 1, args.embed_dim, layers, args.out_dim, dropout_prob=args.dropout_prob, pool=args.pool).to(device)
    elif args.eqn == 4 and args.model == 'eq':
        model = Eq4Net(PrevalenceDataset.num_entities + 1, args.embed_dim, layers, args.out_dim).to(device)
    elif args.eqn == 2 and args.model == 'mlp':
        log.info('Making Eq2DeepSet')
        model = Eq2DeepSet(PrevalenceDataset.num_entities + 1, args.embed_dim, args.hid_dim, args.out_dim).to(device)
    else:
        log.info('Doing baseline with dropout {}, pool {}'.format(
            args.dropout_prob, args.pool
        ))
        model = BaselineDeepSetsFeatCat(PrevalenceDataset.num_entities + 1,
                                        args.embed_dim,
                                        args.hid_dim,
                                        args.out_dim,
                                        dropout_prob=args.dropout_prob,
                                        pool=args.pool
                                       ).to(device)

    loaded, start_epoch = try_load_weights(savedir, model, device)
    if not loaded:
        log.info('Init model parameters')
        reset_params(model)
        start_epoch = 0
    else:
        log.info('Loaded weights from {savedir}')

    params = {'batch_size': args.batch_size, 'shuffle': True, 'pin_memory': args.pin, 'num_workers': args.num_workers}
    val_len = int(0.1 * len(train_data))
    #train_data, val_data = random_split(train_data, (len(train_data) - val_len, val_len))
    train_dataloader = DataLoader(train_data, **params)
    #val_dataloader = DataLoader(val_data, **params)
    test_dataloader = DataLoader(test_data, **params)
    loss_func = nn.MSELoss()
    opt= torch.optim.Adam(model.parameters(), lr=args.lr)
    st = time.time()

    for e in range(start_epoch, start_epoch + args.epochs+ 1):
        ep_start = time.time()
        batch_maes = []
        batch_mses = []
        for batch in train_dataloader:
            for param in model.parameters():
                param.grad = None

            ybatch = batch[-1].to(device)
            ypred = pred_batch(model, batch, device)
            loss = loss_func(ybatch, ypred)
            batch_mae = (ypred - ybatch).abs().mean().item()
            loss.backward()
            opt.step()
            batch_mses.append(loss.item())
            batch_maes.append(batch_mae)
        epoch_time = time.time() - ep_start

        # save avg batch mae/mse
        if swr:
            swr.add_scalar('train/batch_avg_mae', np.mean(batch_maes), e)
            swr.add_scalar('train/batch_avg_mse', np.mean(batch_mses), e)

        if e % args.print_update == 0:
            tot_se = tot_ae = 0
            with torch.no_grad():
                test_start = time.time()
                model.eval()
                for _batch in test_dataloader:
                    _ybatch = _batch[-1].to(device)
                    _ypred = pred_batch(model, _batch, device)
                    _loss  = loss_func(_ybatch, _ypred) # loss is mean squared error
                    tot_se += (_loss.item() * len(_ybatch))
                    tot_ae += (_ypred - _ybatch).abs().sum().item()
                tot_mse = tot_se / len(test_data)
                tot_mae = tot_ae / len(test_data)
                test_time = time.time() - test_start
                log.info('Ep: {:4d} | Last ep: mae {:.2f}, mse: {:.2f} | Test mae: {:.2f} | Test mse: {:.2f} | Time: {:.2f}mins | Ep time: {:.2f}s, Test time: {:.2f}s'.format(
                    e, np.mean(batch_maes), np.mean(batch_mses), tot_mae, tot_mse, (time.time() - st) / 60., epoch_time, test_time
                ))

            if swr:
                swr.add_scalar('val/mse', tot_mse, e)
                swr.add_scalar('val/mae', tot_mse, e)
            model.train()

            torch.save(model.state_dict(), os.path.join(savedir, f'last_model_{e}.pt'))
            last_fname = os.path.join(savedir, f'last_model_{e - args.print_update}.pt')
            if os.path.exists(last_fname):
                os.remove(last_fname)

        if e % args.save_iter == 0 and e > 0:
            torch.save(model.state_dict(), os.path.join(savedir, f'model_{e}.pt'))

    torch.save(model.state_dict(), os.path.join(savedir, f'model_{e}_final.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, default='./results/prevalence/')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--out_dim', type=int, default=128)
    parser.add_argument('--num_eq_layers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--print_update', type=int, default=1000)
    parser.add_argument('--save_iter', type=int, default=5000)
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
    parser.add_argument('--pool', type=str, default='sum')
    parser.add_argument('--dropout_prob', type=float, default=0)
    args = parser.parse_args()
    main(args)
