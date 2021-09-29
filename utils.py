import os
import psutil
import sys
import logging

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
        level=logging.DEBUG,
        format=str_fmt,
        datefmt=date_fmt,
        handlers=handlers
    )

    logger = logging.getLogger(__name__)
    return logger

def setup_experiment(args, savedir, save):
    '''
    savedir: str location to save contents in
    save: boolean
    
    '''
    if os.path.exists(sumdir) and save:
        os.makedirs(sumdir)
    if args.save:
        swr = SummaryWriter(sumdir)
        json.dumps(args.__dict__, open(os.path.join(sumdir, 'args.json'), 'w'))
        logfile = os.path.join(sumdir, 'output.log')
        cnt = 1
        while os.path.exists(logfile):
            logfile = os.path.join(sumdir, f'output{cnt}.log')
            cnt += 1
    else:
        logfile = args.logfile

def check_memory(verbose=True):
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    if verbose:
        print("Consumed {:.2f}mb memory".format(mem))
    return mem
