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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", required=True, type=str)
    parser.add_argument("-out", required=True, type=str)
    parser.add_argument('-overwrite', dest='overwrite', action='store_true', default=False)
    args = parser.parse_args()

    with open(args.input) as f:
        annotations = pd.read_csv(f)
        seen = {}
        w = open(args.out, 'w')
        article_ids = [seen.setdefault(x, x) for x in list(annotations["HITId"]) if x not in seen]

        id2article = {}
        for id in article_ids:
            id2article[id] = json.loads(list(annotations[annotations["HITId"] == id]["Input.articlejson"])[0])
        
        for id in article_ids:
            sentences = [(x[0]['text'] + x[1]['text']) for x in id2article[id]['paragraphs'][2:]][:11] # remove the "article" as the first sentence

            text = ""
            for i, x in enumerate(sentences):
                text += f"Sentence {i}: " + x.strip() + "\n"
            
            print(text, file=w)
            raw_annotations = [json.loads(x)[0] for x in list(annotations[annotations["HITId"] == id]["Answer.taskAnswers"])]
            annotation_ids = convert_raw_annotations_to_ids(raw_annotations) # [[(idx, importance), ...], ...]

            for annotator_id, annotator_annotation in enumerate(annotation_ids):
                description = "Sentences " + ', '.join([f"{index} (importance {priority})" for index, priority in annotator_annotation])
                print(f"Annotator {annotator_id}: {description}", file=w)
            print("\n", file=w)