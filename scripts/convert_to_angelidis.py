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

def convert_raw_annotations_to_ids(raw_annotations):
    # TODO calculate shifts

    full_ids = []
    for annotation in raw_annotations:
        annotation.pop('comments', None)
        query_keys = [f"topic-segment-{x}" for x in range(2, 2*len(annotation) + 1, 2)]
        single_user_ids = []
        for i, k in enumerate(query_keys):
            #print(raw_annotation[k])
            sentence_annotation = annotation[k]
            if sentence_annotation == "not-in-summary":
                pass
            else:
                single_user_ids.append((i, int(sentence_annotation.split("-")[1])))

        single_user_ids.sort(key=lambda x: x[1], reverse=True)
        full_ids.append(single_user_ids)

    return full_ids


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
        encoded_articles = []

        id2article = {}
        for id in article_ids:
            id2article[id] = json.loads(list(annotations[annotations["HITId"] == id]["Input.articlejson"])[0])
        
        for id in tqdm(article_ids):
            sentences = [(x[0]['text'] + x[1]['text']) for x in id2article[id]['paragraphs'][2:]][:11] # remove the "article" as the first sentence

            text = ""
            for x in sentences:
                text += x + "\n\n"

            base_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=30))
            base_strings.append(base_string)

            article_id = hashhex(base_string)
            raw_annotations = [json.loads(x)[0] for x in list(annotations[annotations["HITId"] == id]["Answer.taskAnswers"])]
            annotation_ids = convert_raw_annotations_to_ids(raw_annotations) # [[(idx, importance), ...], ...]

            # helper functions
            first_elements = lambda l: [x[0] for x in l]
            collect_exists = lambda l: set([i for x in l for i in x])
            
            # code to throw out annotations that have no overlap with any of its peers
            annotation_kminus1 = [collect_exists([first_elements(x) for x in (annotation_ids[:i] + annotation_ids[i + 1:])]) for i in range(len(annotation_ids))] # annotations minus the annotator at that index, to discover bad annotators
            topop = []
            for i in range(len(annotation_ids)):
                if all([k[0] not in annotation_kminus1[i] for k in annotation_ids[i]]):
                    topop.append(i)

            annotation_ids = [i for j, i in enumerate(annotation_ids) if j not in topop]

            gold_summaries = []
            for annotation in annotation_ids:
                gold_summaries.append(" ".join([sentences[x[0]] for x in annotation]))

            encoded_article = {
                'entity_id': article_id,
                'split': 'test',
                'reviews': [
                    {
                        "review_id": f"{article_id}_review",
                        "rating": 3,
                        "sentences": sentences
                    },
                ],
                'summaries': {
                    'geo': gold_summaries,
                }
            }
            encoded_articles.append(encoded_article)
        
        with open(os.path.join(args.out, "geo_summ.json"), "w") as f:
            print(json.dumps(encoded_articles, indent=4), file=f)
    
        with open(os.path.join(args.out, "mapping_test.txt"), "w") as f:
            print("\n".join(base_strings), file=f)
