import json
import argparse
import hashlib
import os
import shutil
from pathlib import Path
from os import listdir
from os.path import isfile, join
import random
import string

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-space", required=True, type=str)
    parser.add_argument("-urls", required=True, type=str)
    parser.add_argument("-out", required=True, type=str)
    parser.add_argument('-overwrite', dest='overwrite', action='store_true', default=False)
    parser.add_argument('-size', type=int, required=True)
    args = parser.parse_args()
    
    if os.path.isdir(args.out):
        if not args.overwrite:
            print(f"found dir {args.out} already exists, pass -overwrite if you want")
            exit(1)
        else:
            shutil.rmtree(args.out)
    
    if os.path.isdir(args.urls):
        if not args.overwrite:
            print(f"found dir {args.urls} already exists, pass -overwrite if you want")
            exit(1)
        else:
            shutil.rmtree(args.urls)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.urls).mkdir(parents=True, exist_ok=True)

    with open(args.space) as f:
        data = json.load(f)

        base_strings = []
        summaries = []

        for row in data:
            review_sentences = []
            for sentences in row['reviews']:
                review_sentences.extend(sentences['sentences'])
            review_sentences = [x for x in review_sentences if len(x) > 10][:args.size]

            text = "\n".join(review_sentences)
            text += f"\n\n@highlight\n\nThis is a dummy annotation."

            base_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=30))
            base_strings.append(base_string)
            with open(os.path.join(args.out, hashhex(base_string) + ".story"), 'w') as w:
                print(text, file=w)

            summary_json = {"anchor": review_sentences[0]}
            for key in row['summaries'].keys():
                summary_json[key] = ' '.join(row['summaries'][key])
            summaries.append(summary_json)

        with open(os.path.join(args.urls, "mapping_test.txt"), "w") as f:
            print("\n".join(base_strings), file=f)

        with open(os.path.join(args.urls, "mapping_train.txt"), "w") as f:
            print("\n", file=f)

        with open(os.path.join(args.urls, "mapping_valid.txt"), "w") as f:
            print("\n", file=f)
        
        #with open(os.path.join(args.out, "summaries.json"), "w") as f:
        #    print(json.dumps(summaries), file=f)
