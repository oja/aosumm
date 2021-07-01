from prepro.data_builder import hashhex
from os.path import join as pjoin
import argparse
import glob
import shutil
from tqdm import tqdm

def get_mapping(args):
    corpus_mapping = {}
    for corpus_type in ['valid', 'test', 'train']:
        temp = []
        for line in open(pjoin(args.map, 'mapping_' + corpus_type + '.txt')):
            temp.append(hashhex(line.strip()))
        corpus_mapping[corpus_type] = set([key.strip() for key in temp])
    return corpus_mapping

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-in_dir", required=True)
    parser.add_argument("-out_dir", required=True)
    parser.add_argument("-map", required=True)
    parser.add_argument("-limit", type=int, required=True)

    args = parser.parse_args()

    mapping = get_mapping(args)

    train_count = 0
    valid_test_count = 0
    for f in tqdm(glob.glob(pjoin(args.in_dir, '*.story'))):
        real_name = f.split('/')[-1].split('.')[0]
        if (real_name in mapping['valid']):
            shutil.copy2(f, args.out_dir)
            valid_test_count += 1
        elif (real_name in mapping['test']):
            shutil.copy2(f, args.out_dir)
            valid_test_count += 1
        elif (real_name in mapping['train']):
            if train_count < args.limit:
                shutil.copy2(f, args.out_dir)
                train_count += 1
    
    print(f"Copied {train_count + valid_test_count} files!")

