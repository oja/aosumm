import csv
import pandas as pd
from collections import Counter, defaultdict
import argparse
import os
import hashlib
import string
import random
import shutil
import json
import pprint

from pathlib import Path

import random
random.seed(0)

import numpy as np
np.random.seed(0)

pp = pprint.PrettyPrinter(indent=4)


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
        data = json.load(f)
        articles = data['text'].values()
        
        base_strings = []
    
        for article in articles:
            if article is not None:
                text = article
                for i in range(0, 2):
                    text += f"\n\n@highlight\n\nThis is dummy annotation {i + 1}."
                base_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=30))
                base_strings.append(base_string)
                with open(os.path.join(args.out, hashhex(base_string) + ".story"), 'w') as w:
                    print(text, file=w)
        
        with open(os.path.join(args.out, "mapping_test.txt"), "w") as f:
            print("\n".join(base_strings), file=f)
        
        # for id in article_ids:
        #     sentences = [(x[0]['text'] + x[1]['text']) for x in id2article[id]['paragraphs'][2:]][:11] # remove the "article" as the first sentence
        #     text = ""
        #     for x in sentences:
        #         text += x + "\n\n"

        #     text += f"@highlight\n\n{sentences[1]}"

        #     base_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=30))
        #     base_strings.append(base_string)
        #     with open(os.path.join(args.out, hashhex(base_string) + ".story"), 'w') as w:
        #         print(text, file=w)
            
        #     with open(os.path.join(args.out, "mapping_test.txt"), "w") as f:
        #         print("\n".join(base_strings), file=f)
