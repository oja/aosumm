import argparse

from os import listdir
from os.path import isfile, join
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-filename", required=True, type=str)
    args = parser.parse_args()

    with open(args.filename) as f:
        x = json.load(f)
        for sample in x:
            print("================ Article =================")
            print(sample['text'])

            print("================ QFSumm =================")
            print([sample['text'][i] for i in sample['ids']])
