import os
import psutil
import sys
import json
import logging
import torch
from tensorboardX import SummaryWriter
from argparse import Namespace

def get_logger(fname=None, stdout=True):
    '''
    fname: file location to store the log file
    '''
    handlers = []
    if stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers.append(stdout_handler)
    if fname:
        file_handler = logging.FileHandler(filename=fname)
        handlers.append(file_handler)

    str_fmt = '[%(asctime)s.%(msecs)03d] %(message)s'
    date_fmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=str_fmt,
        datefmt=date_fmt,
        handlers=handlers
    )

    logger = logging.getLogger(__name__)
    return logger

def setup_experiment_log(args, savedir='./results/prevalence/', exp_name='test', save=False):
    '''
    savedir: str location to save contents in
    save: boolean
    Returns: tuple of str (log file) and SummaryWriter
    '''
    if not save:
        return None, None

    if os.path.exists(savedir):
        exp_dir = os.path.join(savedir, exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        sumdir = os.path.join(exp_dir, 'summary')
        swr = SummaryWriter(sumdir)
        json.dump(args.__dict__, open(os.path.join(exp_dir, 'args.json'), 'w'))
        logfile = os.path.join(exp_dir, 'output.log')
        cnt = 1
        while os.path.exists(logfile):
            logfile = os.path.join(exp_dir, f'output{cnt}.log')
            cnt += 1

    else:
        # make the save dir, retry
        os.makedirs(savedir)
        return setup_experiment_log(savedir, exp_name, save)

    return logfile, swr

def check_memory(verbose=True):
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    if verbose:
        print("Consumed {:.2f}mb memory".format(mem))
    return mem

def save_checkpoint(epoch, model, optimizer, fname):
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, fname)

def load_checkpoint(model, optimizer, log, fname):
    start_epoch = 0
    if os.path.isfile(fname):
        log.info("=> loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        log.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(fname, checkpoint['epoch']))
        success = True
    else:
        log.info("=> no checkpoint found at '{}'".format(fname))
        success = False

    return model, optimizer, start_epoch, success

'''
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
'''

