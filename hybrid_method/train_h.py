import os
import random
import torch
import sys
import numpy as np

from argparse import ArgumentParser
from data_reader import train

def set_seed(s):
    random.seed(s)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(s)
    torch.cuda.manual_seed(s)
    np.random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(12345)
    
    parser = ArgumentParser()
    parser.add_argument('--lr', dest='lr', default=1e-04, type=float, help='learning rate')
    parser.add_argument('--maxsteps', dest='maxsteps', default=50000, type=int, help='maximum training step')
    parser.add_argument('--bsize', dest='bsize', default=32, type=int, help='batch size')
    parser.add_argument('--accum', dest='accumsteps', default=2, type=int, help='gradient accumulation step')
    parser.add_argument('--do_eval_steps', dest='do_eval_steps', default=100, type=int, help='number of training steps between evaluation')
    parser.add_argument('--print_log_steps', dest='print_log_steps', default=100, type=int)

    parser.add_argument('--data_dir', dest='data_dir', default='../metal_data')
    parser.add_argument('--doc_file', dest='doc_file', default='documents.pkl')
    parser.add_argument('--triples_train', dest='triples_train', default='triples_train.pkl')
    parser.add_argument('--triples_dev', dest='triples_dev', default='triples_dev.pkl')
    parser.add_argument('--triples_test', dest='triples_test', default='triples_test.pkl')

    parser.add_argument('--ret_result_file', dest='ret_result_file', default='../Ret/model_metal_ret_LR1e-05_BSIZE128/metal_ret_hybrid_test.json')
    parser.add_argument('--clam_result_file', dest='clam_result_file', default='../Cla_M/model_metal_clam_LR1e-05_BSIZE128/metal_clam_hybrid_test.json')
    
    parser.add_argument('--rerank_num', dest='rerank_num', type=int, default=100, help='only reranking the top candidates of Ret')
    
    parser.add_argument('--model_save_dir', dest='model_save_dir', default='./model_metal_hybrid')
    parser.add_argument('--model_to_test', dest='model_to_test', default='./model_metal_hybrid_LR0.0001_BSIZE32', help='path of the model to test')
    parser.add_argument('--result_save_file', dest='result_save_file', default='metal_hybrid_test.json')
    parser.add_argument('--device', dest='device', default="cuda:0")


    args = parser.parse_args()

    assert args.bsize % args.accumsteps == 0, ((args.bsize, args.accumsteps),
                                               "The batch size must be divisible by the number of gradient accumulation steps.")

    args.triples_train = os.path.join(args.data_dir, args.triples_train)
    args.triples_dev = os.path.join(args.data_dir, args.triples_dev)
    args.triples_test = os.path.join(args.data_dir, args.triples_test)
    args.doc_file = os.path.join(args.data_dir, args.doc_file)
    
    args.model_save_dir += '_LR' + str(args.lr) + '_BSIZE' + str(args.bsize)
    if os.path.isdir(args.model_save_dir) == False:
        os.mkdir(args.model_save_dir)
    args.result_save_file = os.path.join(args.model_save_dir, args.result_save_file)
    
    train(args)

if __name__ == "__main__":
    main()







