import argparse
import json
from collections import Counter
from statistics import mean, stdev

import pandas as pd

from os import listdir
from os.path import isfile, join

from evaluate import convert_raw_annotations_to_ids
from prepro.data_builder import greedy_rouge_selection


def get_corresponding_cnndm_article(search_text, raw_path, tokenized_path):
    filenames = [f for f in listdir(raw_path) if isfile(join(raw_path, f)) if f.endswith(".story")]
    for filename in filenames:
        with open(join(raw_path, filename)) as raw_file:
            if ''.join(search_text.lower().split()) in ''.join(''.join(raw_file.read()).lower().split()):
                # return tokenized version of the article sentences
                with open(join(tokenized_path, filename + ".json")) as tokenized_file:
                    tokenized_text = json.load(tokenized_file)
                    sentences = ([([word['word'] for word in sentence['tokens']]) for sentence in tokenized_text['sentences']])
                    for i, x in enumerate(sentences):
                        if x[0] == "@highlight":
                            return sentences[:i], [w for w in sentences[i:] if w[0] != "@highlight"]        
    raise Exception("article not found in DB")


def extractive_oracle(sentences, summary):
    """
    returns indices
    """
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset1", required=True, type=str)
    parser.add_argument("-dataset2", required=True, type=str)
    parser.add_argument("-raw_location", required=True, type=str)
    parser.add_argument("-tokenized_location", required=True, type=str)
    args = parser.parse_args()

    annotations1 = pd.read_csv(open(args.dataset1))
    annotations2 = pd.read_csv(open(args.dataset2))

    seen = {}
    article_ids = [seen.setdefault(x, x) for x in list(annotations1["HITId"]) if x not in seen] # deterministic

    assert len(article_ids) == 100

    jaccard_scores_1_2 = []
    jaccard_scores_1_cnndm = []
    jaccard_scores_2_cnndm = []
    exact_matches_1_2 = 0
    exact_matches_1_cnndm = 0
    exact_matches_2_cnndm = 0

    for id in article_ids:
        raw_sentences1 = json.loads(list(annotations1[annotations1["HITId"] == id]["Input.articlejson"])[0])
        raw_annotations1 = [json.loads(x)[0] for x in list(annotations1[annotations1["HITId"] == id]["Answer.taskAnswers"])]
        annotation_ids1 = convert_raw_annotations_to_ids(raw_annotations1) # [[(idx, importance), ...], ...]
        sentence_1 = raw_sentences1['paragraphs'][2][0]['text']

        found = False
        for i, item in enumerate(list(annotations2["Input.articlejson"])):
            sentence_candidate = (json.loads(item)['paragraphs'][2][0]['text'])
            if sentence_1 == sentence_candidate:

                found = True
                second_id = (annotations2.loc[i]["HITId"])
                break

        if not found:
            raise Exception()
        
        raw_sentences2 = json.loads(list(annotations2[annotations2["HITId"] == second_id]["Input.articlejson"])[0])
        raw_annotations2 = [json.loads(x)[0] for x in list(annotations2[annotations2["HITId"] == second_id]["Answer.taskAnswers"])]
        annotation_ids2 = convert_raw_annotations_to_ids(raw_annotations2) # [[(idx, importance), ...], ...]
        
        first_elements = lambda l: [x[0] for x in l]
        collect_exists = lambda l: set([i for x in l for i in x])

        annotation1_counts = (Counter(sum([first_elements(x) for x in annotation_ids1], [])))
        annotation2_counts = (Counter(sum([first_elements(x) for x in annotation_ids2], [])))
        
        annotation1_counts_top3 = set(first_elements(annotation1_counts.most_common(3)))
        annotation2_counts_top3 = set(first_elements(annotation2_counts.most_common(3)))

        doc_text, abstractive_summary = get_corresponding_cnndm_article(sentence_1, args.raw_location, args.tokenized_location)
        cnndm_ids = set(greedy_rouge_selection(doc_text, abstractive_summary, 3))

        jaccard_scores_1_2.append(len(annotation1_counts_top3.intersection(annotation2_counts_top3)) / len(annotation1_counts_top3.union(annotation2_counts_top3)))
        jaccard_scores_1_cnndm.append(len(annotation1_counts_top3.intersection(cnndm_ids)) / len(annotation1_counts_top3.union(cnndm_ids)))
        jaccard_scores_2_cnndm.append(len(annotation2_counts_top3.intersection(cnndm_ids)) / len(annotation2_counts_top3.union(cnndm_ids)))
        
        if annotation1_counts_top3 == annotation2_counts_top3:
            exact_matches_1_2 += 1
        if annotation1_counts_top3 == cnndm_ids:
            exact_matches_1_cnndm += 1
        if annotation2_counts_top3 == cnndm_ids:
            exact_matches_2_cnndm += 1


    print("dataset1 vs dataset2")
    print(f"mean jaccard index: {mean(jaccard_scores_1_2):.3f}")
    print(f"{100 * exact_matches_1_2 / len(article_ids)}% of predictions were identical")

    print("dataset1 vs cnndm")
    print(f"mean jaccard index: {mean(jaccard_scores_1_cnndm):.3f}")
    print(f"{100 * exact_matches_1_cnndm / len(article_ids)}% of predictions were identical")

    print("dataset2 vs cnndm")
    print(f"mean jaccard index: {mean(jaccard_scores_2_cnndm):.3f}")
    print(f"{100 * exact_matches_2_cnndm / len(article_ids)}% of predictions were identical")


