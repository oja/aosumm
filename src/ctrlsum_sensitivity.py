import argparse
import json
from collections import Counter
from re import L
from statistics import mean, stdev
from os import listdir
from os.path import isfile, join


import pandas as pd

from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-ctrlsum1", required=True, type=str)
    parser.add_argument("-ctrlsum2", required=True, type=str)
    parser.add_argument("-output1", required=True, type=str)
    parser.add_argument("-output2", required=True, type=str)
    args = parser.parse_args()

    ctrlsum1_filenames = [f for f in listdir(args.ctrlsum1) if isfile(join(args.ctrlsum1, f))]

    rouge_1s = []
    rouge_2s = []
    rouge_Ls = []

    for filename in ctrlsum1_filenames:
        with open(join(args.ctrlsum1, filename)) as file1, open(join(args.ctrlsum2, filename)) as file2:
            _rouge = scorer.score(file1.read(), file2.read())
            rouge_1s.append(_rouge['rouge1'].fmeasure)
            rouge_2s.append(_rouge['rouge2'].fmeasure)
            rouge_Ls.append(_rouge['rougeL'].fmeasure)
    
    print(f"ctrlsum: {mean(rouge_1s):.5f} {mean(rouge_2s):.5f} {mean(rouge_Ls):.5f}")
    
    rouge_1s = []
    rouge_2s = []
    rouge_Ls = []
    with open(args.output1) as model_outputs1, open(args.output2) as model_outputs2:
        model_outputs1_json = json.load(model_outputs1)
        model_outputs2_json = json.load(model_outputs2)

        for output1 in model_outputs1_json:
            # find matching article in other doc

            found = False
            for output2_candidate in model_outputs2_json:
                if output2_candidate['text'][0] == output1['text'][0]:
                    found = True
                    output2 = output2_candidate
                    break
            if not found:
                raise Exception("can't find matching output! maybe make sure casing is right?")

            ids1, ids2 = set(output1['ids']), set(output2['ids'])
            
            text1 = " ".join([output1['text'][i] for i in ids1])
            text2 = " ".join([output2['text'][i] for i in ids2])

            _rouge = scorer.score(text1, text2)
            rouge_1s.append(_rouge['rouge1'].fmeasure)
            rouge_2s.append(_rouge['rouge2'].fmeasure)
            rouge_Ls.append(_rouge['rougeL'].fmeasure)
    
    print(f"qfsumm: {mean(rouge_1s):.5f} {mean(rouge_2s):.5f} {mean(rouge_Ls):.5f}")
