import pdb
import time
import torch
from equivariant_layers import ops_1_to_1, ops_1_to_2, ops_2_to_1, ops_2_to_2, set_ops_3_to_3, set_ops_4_to_4
from equivariant_layers_expand import eops_1_to_1, eops_1_to_2, eops_2_to_1, eops_2_to_2, eset_ops_3_to_3, eset_ops_4_to_4
from utils import check_memory

from torch.profiler import profile, record_function, ProfilerActivity

def do_profile(ofunc, x):
    with profile(activities=[
            ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            ofunc(x)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

def main():
    B = 32
    d = 32
    m = 5
    check_memory()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x1 = torch.rand(B, d, m).to(device)
    x2 = torch.rand(B, d, m, m).to(device)
    x3 = torch.rand(B, d, m, m, m).to(device)
    #x4 = torch.rand(B, d, m, m, m, m).to(device)

    #print('Trying original')
    #do_profile(ops_2_to_2, x2)
    print('-------')
    print('Trying expand')
    check_memory()
    res = eops_2_to_2, x2
    print('post eops22')
    check_memory()
    res3 = eset_ops_3_to_3(x3)
    print('post eops33')
    check_memory()
    pdb.set_trace()

    do_profile(eops_2_to_2, x2)
    #do_profile(eset_ops_3_to_3, x3)
    return None

def main2():
    B = 32
    d = 128
    m = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x1 = torch.rand(B, d, m).to(device)
    x2 = torch.rand(B, d, m, m).to(device)
    x3 = torch.rand(B, d, m, m, m).to(device)
    x4 = torch.rand(B, d, m, m, m, m).to(device)
    check_memory()


    args = [x1, x1, x2, x2, x3, x4]
    ofs = [ops_1_to_1, ops_1_to_2, ops_2_to_1, ops_2_to_2, set_ops_3_to_3, set_ops_4_to_4]
    efs = [eops_1_to_1, eops_1_to_2, eops_2_to_1, eops_2_to_2, eset_ops_3_to_3, eset_ops_4_to_4]
    os = []

    es = []
    for x, of in zip(args, ofs):
        st = time.time()
        os.append(of(x))
        ot = time.time() - st
    check_memory()
    del os

    for x, ef in zip(args, efs):
        st = time.time()
        es.append(ef(x))
        et = time.time() - st
        #print(f'Elapsed: {ot:.6f}s | expand version: {et:.6f}s')
        #print(f'expand version: {et:.6f}s')
    check_memory()
    del es

    check_memory()

if __name__ == '__main__':
    main()
