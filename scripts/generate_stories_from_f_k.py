import argparse
import os
import shutil
from pathlib import Path
from os import listdir
from os.path import isfile, join
import random
import string
import hashlib
import re

def clean_lines(lines):
    new_lines = []
    for line in lines:
        new_lines.append(line.strip().split('\t')[1].replace('-lrb-', '(').replace('-rrb-', ')'))

    return new_lines

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", required=True, type=str)
    parser.add_argument("-out", required=True, type=str)
    parser.add_argument('-overwrite', dest='overwrite', action='store_true', default=False)
    args = parser.parse_args()

    if os.path.isdir(args.out):
            if not args.overwrite:
                print(f"found dir {args.out} already exists, pass -overwrite if you want")
                exit(1)
            else:
               shutil.rmtree(args.out)

    Path(args.out).mkdir(parents=True, exist_ok=True)

    for split in ['train', 'test', 'val']:
        filenames = [join(args.input, split, f) for f in listdir(os.path.join(args.input, split)) if isfile(join(args.input, split, f))]
        base_strings = []

        for filename in filenames:
            pivots = []

            with open(filename) as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if line.strip() == '':
                        pivots.append(i)

            aspect = clean_lines(lines[:pivots[0]])
            urls = clean_lines(lines[pivots[0] + 1:pivots[1]])
            hashes = clean_lines(lines[pivots[1] + 1:pivots[2]])
            article = clean_lines(lines[pivots[2] + 1:pivots[3]])
            summary = clean_lines(lines[pivots[3] + 1:])

            base_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=30))
            base_strings.append(base_string)

            text = '\n\n'.join(article)
            summary_lines = [x.strip() for x in re.split('</s>|<s>', summary[0]) if x.strip() != '']

            for line in summary_lines:
                text += f"\n\n@highlight\n\n{line}"
            
            with open(os.path.join(args.out, hashhex(base_string) + ".story"), 'w') as w:
                print(text, file=w)

            with open(os.path.join(args.out, hashhex(base_string) + ".story.aspect"), 'w') as w:
                print(''.join(aspect), file=w)

        with open((f"urls/mapping_{split}.txt"), "w") as f:
            print("\n".join(base_strings), file=f)
            print("")
        


    
