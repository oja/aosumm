from os.path import join as pjoin
import os
import argparse
import glob
import shutil
from tqdm import tqdm
import subprocess
import shlex
from collections import Counter

from nltk.tokenize import sent_tokenize


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-dir", required=True)
    parser.add_argument("-exclude_dir", required=True)

    args = parser.parse_args()

    failures = 0

    filepaths_to_exclude = []
    for f in tqdm(glob.glob(pjoin(args.exclude_dir, '*.story'))):
        article = open(f).read()
        tokenized_1st_sentence = sent_tokenize(article)[0][:75].strip()
        
        s = subprocess.run(["grep", "-Ril", tokenized_1st_sentence, "../data/raw/cnndm"], stdout=subprocess.PIPE)
        
        try:
            result = s.stdout.decode('utf-8').strip()
            result_split_length = len(result.split("\n"))
            
            if result_split_length != 1:
                if result_split_length == 2:
                    filepaths_to_exclude.append(result.split("\n")[0])
                    filepaths_to_exclude.append(result.split("\n")[1])
                else:
                    print("Failed")
                    print(sent_tokenize(article))
                    if result_split_length < 5:
                        print(result)
                    failures += 1
                    continue
            else:
                filepaths_to_exclude.append(result)
        except FileNotFoundError:
            failures += 1
            print("Could not find matching article in CNNDM")

    print(f"total number of failures {failures}")
    counter = Counter(filepaths_to_exclude)
    print(len(counter))

    for file in filepaths_to_exclude:
        print(f"Excluding {file}")
        os.rename(file, file + ".exclude")
