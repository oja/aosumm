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
    parser.add_argument("-input", required=True, type=str)
    parser.add_argument("-out", required=True, type=str)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
        for article in data:
            doc, tag, text = Doc().tagtext()
            doc.stag('link', rel='stylesheet', href='style.css')
            ids = set(article['ids'])
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

            with open(os.path.join(args.out, hashhex(" ".join(article['text'][0])) + '.html'), 'w') as w:
                w.write(doc.getvalue())