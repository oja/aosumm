import os
import sys
import argparse
import json
from os import listdir
from os.path import isfile, join
from pathlib import Path
import shutil
from numpy import concatenate
from tqdm import tqdm
import time
import itertools
import operator

from bert_score import BERTScorer


def partition(alist, indices):
    return [alist[i:j] for i, j in zip([0]+indices, indices+[None])]

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_raw_scores(scorer, sentences1, sentences2, batch_size):
    _, _, scores = scorer.score(sentences1, sentences2, batch_size=batch_size)
    return scores

def get_indices(scores, splits, summary_size):
    assert sorted(splits) == splits

    paritioned_scores = [scores[i:j] for i, j in zip([0]+splits, splits)]
    batched_indices = []

    for scores in paritioned_scores:
        indexed_scores = [(idx, score.item()) for idx, score in enumerate(list(scores))]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        indices = [idx for idx, score in indexed_scores if score > 0] + [idx for idx, score in indexed_scores if score <= 0]
        batched_indices.append(indices[:summary_size])
    return batched_indices

def get_query_reference(abstract_sent_list, query=None, k=1):
    abstract_sent = sum(abstract_sent_list, [])

    if query:
        multiplier = 0
        while (multiplier * len(query)) / len(abstract_sent) < k:
            multiplier += 1
        abstract_sent = abstract_sent + (multiplier * query)
    
    return abstract_sent

def greedy_bertscore_selection(scorer, doc_sent_list, abstract_sent_list, summary_size, query=None, k=3):
    sents = [(' '.join(s)) for s in doc_sent_list]
    abstract_sent = sum(abstract_sent_list, [])

    if query:
        multiplier = 0
        while (multiplier * len(query)) / len(abstract_sent) < k:
            multiplier += 1
        abstract_sent = abstract_sent + (multiplier * query)

    start_time = time.time()
    _, _, scores = scorer.score(sents * 100, [' '.join(abstract_sent)] * len(sents) * 100)
    end_time = time.time()
    print(f"processed {len(sents) * 100} sentences in {end_time - start_time} sec")

    start_time = time.time()
    _, _, scores = scorer.score(sents, [' '.join(abstract_sent)] * len(sents))
    end_time = time.time()
    print(f"processed {len(sents)} sentences in {end_time - start_time} sec")
    

    indexed_scores = [(idx, score.item()) for idx, score in enumerate(list(scores))]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    indices = [idx for idx, score in indexed_scores if score > 0] + [idx for idx, score in indexed_scores if score <= 0]

    return indices[:summary_size]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-in_dir", required=True, type=str)
    parser.add_argument("-out_dir", required=True, type=str)
    parser.add_argument('-max_src_nsents', default=100, type=int)
    parser.add_argument('-intensity', default=1, type=float)
    parser.add_argument('-summary_size', default=3, type=int)
    parser.add_argument('-batch_size', default=2048, type=int)
    parser.add_argument('-gpu', default=0, type=int)

    args = parser.parse_args()
    scorer = BERTScorer(model_type="albert-base-v2", nthreads=4, batch_size=args.batch_size, device=f'cuda:{args.gpu}')

    input_filenames = [f for f in listdir(args.in_dir) if isfile(join(args.in_dir, f))]

    if os.path.isdir(args.out_dir):
        print(f"dir {args.out_dir} already exists, deleting")
        shutil.rmtree(args.out_dir)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    for input_filename in tqdm(input_filenames):
        with open(join(args.in_dir, input_filename)) as input_file:
            print(f"opening {input_filename}")
            input_file_json = json.load(input_file)
            new_json = []
            for documents in batch(input_file_json, 2048):
                num_docs = (len(documents))
                start_time = time.time()
                document_lengths = [len(document['src'][:args.max_src_nsents]) for document in documents]
                document_sentences = [' '.join(sentence) for sentence in sum([document['src'][:args.max_src_nsents] for document in documents], [])]
                assert sum(document_lengths) == len(document_sentences)

                reference_sentences = []
                for i, document in enumerate(documents):
                    query = None
                    if 'tokens' in document:
                        query = document['tokens']
                    reference_sentences.extend([' '.join(get_query_reference(document['tgt'], query, args.intensity))] * document_lengths[i])

                raw_scores = get_raw_scores(scorer, document_sentences, reference_sentences, args.batch_size)
                splits = list(itertools.accumulate(document_lengths, operator.add))
                indices = get_indices(raw_scores, splits, args.summary_size)
                end_time = time.time()

                print(f"evaluating {num_docs} took {end_time - start_time:.5f} sec, speed: {num_docs / (end_time - start_time):.5f} docs / sec")
                for i, document in enumerate(documents):
                    new_doc = {'src': document['src'][:args.max_src_nsents], 'tgt': document['tgt'], 'bertscore_idxs': indices[i]}
                    if 'tokens' in document:
                        new_doc['tokens'] = document['tokens']
                    new_json.append(new_doc)
                
            with open(join(args.out_dir, input_filename), 'w') as w:
                json.dump(new_json, w)
                
    

