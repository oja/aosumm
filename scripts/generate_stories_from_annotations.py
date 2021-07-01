import csv
import pandas as pd
import json
from collections import Counter, defaultdict
import argparse
import os
import hashlib
import string
import random
import shutil
from tqdm import tqdm

import os
import re
import subprocess

from pathlib import Path

import random
random.seed(0)

import numpy as np
np.random.seed(0)

def grep(filepath, regex):
    regObj = re.compile(regex)
    with open(filepath) as f:
        data = f.read()
        if regObj.match(data):
            return data.split("@highlight")[1:]
    return None


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

    with open(args.input) as f:
        base_strings = []
        annotations = pd.read_csv(f)
        seen = {}
        article_ids = [seen.setdefault(x, x) for x in list(annotations["HITId"]) if x not in seen]

        id2article = {}
        for id in article_ids:
            id2article[id] = json.loads(list(annotations[annotations["HITId"] == id]["Input.articlejson"])[0])
        
        for id in tqdm(article_ids):
            #print(id2article[id]['paragraphs'])
            #for x in id2article[id]['paragraphs'][2:]:
            #    print((x[0]['text'] + x[1]['text']))
            #    print("")

            # limit to 10
            sentences = [(x[0]['text'] + x[1]['text']) for x in id2article[id]['paragraphs'][2:]][:11] # remove the "article" as the first sentence

            text = ""
            for x in sentences:
                text += x + "\n\n"
            
            s = subprocess.run(["grep", "-Ril", sentences[0], "../data/raw/cnndm"], stdout=subprocess.PIPE)
            filepath = (s.stdout.decode('utf-8').split('\n')[0])
            with open(filepath) as f:
                text += "@highlight" + f.read().split("@highlight", 1)[1]
            

            base_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=30))
            base_strings.append(base_string)
            with open(os.path.join(args.out, hashhex(base_string) + ".story"), 'w') as w:
                print(text, file=w)
            
            with open(os.path.join(args.out, "mapping_test.txt"), "w") as f:
                print("\n".join(base_strings), file=f)
