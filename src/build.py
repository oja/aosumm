import argparse
from math import pi
import time
import copy
import os
import shutil


from others.logging import init_logger
from prepro import data_builder

from pathlib import Path

import torch
torch.manual_seed(0)

import random
random.seed(0)

import numpy as np
np.random.seed(0)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_dataset_args(root, name, raw):
    pipeline_labels = ["raw", "tokenized", "json", "binary"]
    pipeline_dirs = [os.path.join(root, label, raw) if label == 'raw' else os.path.join(root, label, name) for label in pipeline_labels]
    return [(pipeline_dirs[i], pipeline_dirs[i + 1]) for i in range(len(pipeline_dirs) - 1)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a query-focused dataset by preprocessing, tokenizing, and binarizing a given raw dataset.")

    parser.add_argument("-pretrained_model", default='bert', type=str, help="which pretrained model to use")

    #parser.add_argument("-mode", default='', type=str)
    #parser.add_argument("-select_mode", default='greedy', type=str)
    parser.add_argument("-map_path", default='../urls/')
    #parser.add_argument("-raw_path", default='../../line_data')
    #parser.add_argument("-save_path", default='../../data/')
    parser.add_argument("-root", required=True, default='../data/', help="location of root directory for data")
    parser.add_argument("-raw", required=True, help="name of raw directory within the root directory")
    parser.add_argument("-name", required=True, help="name of the generated datset")
    parser.add_argument('-overwrite', dest='overwrite', action='store_true', default=False, help="overwrite existing datasets that have the same name")

    # mins are set to 0 to include the entire collected dataset, but be careful!
    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_src_nsents', default=3, type=int) #use 0 and 10 for eval datasetz
    parser.add_argument('-max_src_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)
    parser.add_argument('-min_tgt_ntokens', default=5, type=int)
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)

    parser.add_argument('-summary_size', default=3, type=int)

    parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-use_bert_basic_tokenizer", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument('-log_file', default='../logs/cnndm.log')

    #parser.add_argument('-dataset', default='')

    parser.add_argument('-n_cpus', default=2, type=int)

    # query-focused options 
    parser.add_argument("-qf", type=str2bool, nargs="?", const=True, default=False, help="generate a query-focused dataset")
    parser.add_argument("-keywords", type=str, default=None, required=False, help="(useful for eval) train on these supplied keywords, otherwise use TF-IDF keywords") # if keywords are not supplied, TFIDF will be used
    parser.add_argument("-contrastive", choices=['none', 'binary'], default="none", help='whether to use contrastive training')
    parser.add_argument('-intensity', default=0.2, type=float, help="intensity of oracle summary modification")

    parser.add_argument("-bertscore", type=str2bool, nargs="?", const=True, default=False, help='whether to use bertscore instead of rougescore')

    parser.add_argument('-dataset', default='')

    args = parser.parse_args()
    ["tokenize", "format_to_lines", "format_to_bert"]

    if args.keywords is not None:
        args.keywords = [str(keyword) for keyword in args.keywords.split(",")]

    args.aspect = False
    print(args)
    for mode, (arg1, arg2) in zip(['tokenize', 'format_to_lines', 'format_to_bert'], get_dataset_args(args.root, args.name, args.raw)):
        print(f"> Starting {mode}...")
        start_time = time.time()
        tempargs = copy.deepcopy(args)
        tempargs.raw_path = arg1

        if os.path.isdir(arg2):
            if not args.overwrite:
                print(f"found dir {arg2} already exists, pass -overwrite if you want")
                exit(1)
            else:
               shutil.rmtree(arg2)
        
        Path(arg2).mkdir(parents=True, exist_ok=True)

        if mode == "format_to_lines":
            arg2 += "/t"
        tempargs.save_path = arg2

        init_logger(args.log_file)
        eval('data_builder.'+mode+'(tempargs)')
        end_time = time.time()
        print(f"> Finished {mode} in {end_time - start_time} seconds")

