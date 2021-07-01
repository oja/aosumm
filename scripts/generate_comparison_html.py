import pandas as pd
import json
import scipy.stats
import random
import numpy as np
from yattag import Doc
import itertools
from collections import defaultdict
import argparse
import os

random.seed(1)

import hashlib

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_dir", required=True, type=str)
    parser.add_argument("-names", required=True, type=str)
    parser.add_argument("-out", required=True, type=str)
    args = parser.parse_args()

    names = args.names.split(",")

    files = [json.load(open(os.path.join(args.input_dir, name))) for name in names]

    for x in range(0, 100):
        print(x)
        doc, tag, text = Doc().tagtext()
        doc.stag('link', rel='stylesheet', href='style.css')
        print(names)
        for i, data in enumerate(files):
            article = data[x]
            ids = set(article['ids'])
            with tag('h2'):
                text(names[i])
            with tag('body'):
                with tag('p'):
                    with tag('u'):
                        text("query tokens")
                    doc.stag('br')
                    text(f"{article['query']}")
                with tag('p'):
                    with tag('u'):
                        text("prediction")
                    doc.stag('br')
                    for i, sentence in enumerate(article['text']):
                        if i in ids:
                            with tag('span', klass="topic-2"):
                                text(sentence.capitalize())
                        else:
                            with tag('span'):
                                text(sentence.capitalize())
                        text(" ")
        with open(os.path.join(args.out, f'{x}.html'), 'w') as w:
                w.write(doc.getvalue())
