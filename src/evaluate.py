from collections import Counter, defaultdict
import argparse
import random
import numpy as np
import string
import subprocess
import os
import json
import copy
from statistics import mean
import warnings
import itertools
import math
import time
from os import listdir
from os.path import isfile, join
from nltk.tokenize import word_tokenize

import pandas as pd
import hashlib
from pandas.core.indexing import convert_to_index_sliceable
from scipy import stats
import scipy 

from  rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

warnings.simplefilter("ignore", scipy.stats.SpearmanRConstantInputWarning)

# set random seeds
random.seed(42)
np.random.seed(42)

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


def build_dataset(args, name):
    qf_dataset = subprocess.run(["python", 
                        "build.py", 
                        "-raw", args.dataset,
                        "-root", args.dataset_dir,
                        "-map_path", args.map_path,
                        "-name", name,
                        "-min_src_nsents", "0",
                        "-max_src_nsents", "10",
                        "-min_src_ntokens_per_sent", "0",
                        "-min_tgt_ntokens", "0",
                        "-qf", "-keywords", args.keyword_set, '-overwrite', '-intensity', '0'], stdout=subprocess.PIPE)
    
    if args.nonqf_model is not None:
        nonqf_dataset = subprocess.run(["python", 
                            "build.py", 
                            "-raw", args.dataset,
                            "-root", args.dataset_dir,
                            "-map_path", args.map_path,
                            "-name", (name + "_nonqf"), 
                            "-min_src_nsents", "0",
                            "-max_src_nsents", "10",
                            "-min_src_ntokens_per_sent", "0", 
                            "-min_tgt_ntokens", "0", '-overwrite'], stdout=subprocess.PIPE)
        return qf_dataset, nonqf_dataset
    
    return qf_dataset, None

def do_inference(args, name):
    results = []
    for model in args.models:
        result_path = os.path.join(args.results_dir, f"{name}_{model}")
        s = subprocess.run(["python", "train.py", 
                            "-task", "ext", 
                            "-mode", "test", 
                            "-batch_size", "1", 
                            "-test_batch_size", "1", 
                            "-bert_data_path", f"../data/binary/{name}/t",
                            "-log_file", "../logs/temp",
                            "-sep_optim", "true",
                            "-use_interval", "true",
                            "-visible_gpus", "1",
                            "-max_pos", "512",
                            "-max_length", "200",
                            "-alpha", "0.95",
                            "-min_length", "50",
                            "-result_path", result_path,
                            "-test_from", f"../models/{model}/model_step_{args.step}.pt"], stdout=subprocess.PIPE)
        results.append(result_path)
    return results

def find_corresponding_model_output(raw_sentences, model_outputs):
    sentence = raw_sentences['paragraphs'][2][0]['text']
    for x in model_outputs:
        if ''.join(sentence.lower().split()) in ''.join(''.join(x['text']).lower().split()):
            return x
    return None


def get_hashfilename(text, dir):
    filenames_to_search = [f for f in listdir(dir) if isfile(join(dir, f)) if f.endswith(".story")]
    for filename in filenames_to_search:
        with open(join(dir, filename)) as file_to_search:
            if ''.join(text.lower().split()) in ''.join(''.join(file_to_search.read()).lower().split()):
                return filename
    return None


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

def fix_annotations(cleaned_text, annotation_ids):
    shifts = {}
    shift = 0
    new_cleaned_text = []
    for i, sentence in enumerate(cleaned_text):
        # dumb hardcoding stuff
        special_cases = ["The Japanese Meteorological Agency issued a tsunami advisory for the Japanese coastal areas including the Fukushima prefecture, warning people to leave the coast.\"",
                        "So we're here.\"",
                        "They've resorted to going in one man at a time on foot.\"",
                        "In every aspect of our operation there, we are running against time.\"",
                        "We\'ll assist if they request it.\"",
                        "In other words, the amount of stress released by this earthquake is minuscule compared to the amount that\'s built up and is building up for the Big One when it happens some day in the future.\"",
                        "What we do, we have to do fast.\"",
                        "Or someone trying to give me a shake down??\"",
                        "#kneesbuckled.\"",
                        "Is it a train?'",
                        "Everyone\'s eyes just kind of widened, and we all just ran to the front door.\""]
        shifts[i] = shift
        if sentence == "Key Concepts: Identify or explain these subjects you heard about in today's show:\n\n1. subduction zone\n\n2.":
            new_cleaned_text.extend(["Key Concepts: Identify or explain these subjects you heard about in today's show",
                                    "1. ",
                                    "subduction zone",
                                    "2."]) 
            shift += 3
        elif sentence == "A hydrogen explosion blew the roof and upper walls off the No 1. reactor building two days after the quake, and another blast two days later blew apart the No.":
            new_cleaned_text.extend(["A hydrogen explosion blew the roof and upper walls off the No 1.",
                                    "reactor building two days after the quake, and another blast two days later blew apart the No."])
            shift += 1
        elif sentence in special_cases:
            new_cleaned_text.extend([sentence[:-1], sentence[-1]])            
            shift += 1
        else:
            new_cleaned_text.extend(sentence.split("\n\n"))
            shift += len(sentence.split("\n\n")) - 1
        
    
    new_annotation_ids = []
    for _annotation_ids in annotation_ids:
        _new_annotation_ids = []
        for (idx, importance) in _annotation_ids:
            _new_annotation_ids.append((idx + shifts[idx], importance))
        new_annotation_ids.append(_new_annotation_ids)
    
    return new_cleaned_text, new_annotation_ids

def evaluate_output(annotations_filename, model_outputs_filename, visualize=None):
    """Return (f1_score, spearman correlation) for given annotations, model outputs"""
    # open files
    annotations = pd.read_csv(open(annotations_filename))
    model_outputs = json.load(open(model_outputs_filename))

    if visualize is not None:
        visualize_file = open("visualization.txt", 'w')

    # get unique articles
    seen = {}
    article_ids = [seen.setdefault(x, x) for x in list(annotations["HITId"]) if x not in seen] # deterministic
    
    # setup metrics
    num_correct = 0
    num_predicted = 0
    num_gold = 0

    max_possible_num_correct = 0

    spearmans = []

    rouge_1s = []
    rouge_2s = []
    rouge_Ls = []

    best_words = Counter()

    num_sents = []
    num_words = []

    histogram = Counter()

    for id in article_ids:
        # get annotation ids
        raw_sentences = json.loads(list(annotations[annotations["HITId"] == id]["Input.articlejson"])[0])
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

        # get model ids
        model_output = find_corresponding_model_output(raw_sentences, model_outputs)
        if model_output is None:
            print('Error: Corresponding model output not found...'.upper())
            print(raw_annotations)
            exit(1)
        prediction_ids = model_output['ids'][:3]

        cleaned_text = ([(x[0]['text'].strip() + x[1]['text'].strip()) for x in raw_sentences['paragraphs'][2:]])

        if (len(cleaned_text)) != (len(model_output['text'])):
            cleaned_text, annotation_ids = fix_annotations(cleaned_text, annotation_ids)
        
        # calculate F1 overlap score against annotation
        summed_annotations = Counter()
        summed_annotations.update([i[0] for x in annotation_ids for i in x])

        for key, value in summed_annotations.items():
            histogram[value] += 1

        num_annotators = len(annotation_ids)

        assert (len(cleaned_text)) == (len(model_output['text']))

        _num_correct = sum([summed_annotations[idx] for idx in prediction_ids])
        _num_predicted = len(prediction_ids) * num_annotators
        _num_gold = sum([x[1] for x in summed_annotations.items()])

        if visualize:
            print(f"article: {[(i, sentence) for i, sentence in enumerate(cleaned_text)]}", file=visualize_file)
            print("annotations (count / max possible):", file=visualize_file)
            for i in range(len(cleaned_text)):
                print(f"\t sentence {i}: {summed_annotations[i]} / {num_annotators}", end="", file=visualize_file)
                if i in prediction_ids:
                    print("<----------- prediction", file=visualize_file)
                else:
                    print("", file=visualize_file)
            print(f"predictions: {prediction_ids}", file=visualize_file)
            print(f"num_correct: {_num_correct}", file=visualize_file)
            print(f"num_predicted: {_num_predicted}", file=visualize_file)
            print(f"num_gold: {_num_gold}", file=visualize_file)
            print("", file=visualize_file)

        num_correct += _num_correct
        num_predicted += _num_predicted
        num_gold += _num_gold

        # get spearman
        prediction_ranking = [0] * len(cleaned_text)
        for priority, index in enumerate(prediction_ids):
            prediction_ranking[index] = priority

        _num_sents = []
        _num_words = []
        for _annotation_ids in annotation_ids:
            annotation_ranking = [0] * len(cleaned_text)
            _num_sents.append(len(_annotation_ids))
            for index, priority in _annotation_ids:
                annotation_ranking[index] = priority
                _num_words.append(len(word_tokenize(cleaned_text[index])))

            _spearman = stats.spearmanr(prediction_ranking, annotation_ranking)
            spearmans.append(_spearman.correlation)
        num_sents.append(mean(_num_sents))
        num_words.append(mean(_num_words))
        
        # get rouge
        annotator_top3_prediction_ids = [x[0] for x in summed_annotations.most_common(3)]
        annotator_prediction_text = ' '.join([cleaned_text[i] for i in annotator_top3_prediction_ids])

        for sentence in [cleaned_text[i] for i in annotator_top3_prediction_ids]:
            best_words.update([x for x in word_tokenize(sentence.lower()) if len(x) > 4])

        _max_possible_num_correct = sum([summed_annotations[id] for id in annotator_top3_prediction_ids])
        max_possible_num_correct += _max_possible_num_correct

        model_prediction_text = ' '.join([cleaned_text[i] for i in prediction_ids])

        _rouge = scorer.score(annotator_prediction_text, model_prediction_text)
        rouge_1s.append(_rouge['rouge1'].fmeasure)
        rouge_2s.append(_rouge['rouge2'].fmeasure)
        rouge_Ls.append(_rouge['rougeL'].fmeasure)

    precision = num_correct / num_predicted
    recall = num_correct / num_gold
    F1 = 2*((precision * recall) / (precision + recall))

    print(f"mean sent #:{mean(num_sents)}")
    print(f"mean word #:{mean(num_words)}")
    max_possible_precision = max_possible_num_correct / num_predicted
    max_possible_recall = max_possible_num_correct / num_gold
    max_possible_F1 = 2*((max_possible_precision * max_possible_recall) / (max_possible_precision + max_possible_recall))
    print(f"Best words were: {best_words.most_common(10)}")
    print(f"histogram: {histogram}")
    return 100 * F1, mean(spearmans), {'rouge1': mean(rouge_1s), 'rouge2': mean(rouge_2s), 'rougeL': mean(rouge_Ls)}, 100 * max_possible_F1


def evaluate_qa_output(annotations_filename, qa_dir, raw_dir):
    annotations = pd.read_csv(open(annotations_filename))

    # get unique articles
    seen = {}
    article_ids = [seen.setdefault(x, x) for x in list(annotations["HITId"]) if x not in seen] # deterministic
    
    # setup metrics
    num_correct = 0
    num_predicted = 0
    num_gold = 0

    spearmans = []

    rouge_1s = []
    rouge_2s = []
    rouge_Ls = []

    for id in article_ids:
        # get annotation ids
        raw_sentences = json.loads(list(annotations[annotations["HITId"] == id]["Input.articlejson"])[0])
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
        #print(raw_sentences['paragraphs'][2][0]['text'][:40])
        qa_output_filename = get_hashfilename(raw_sentences['paragraphs'][2][0]['text'][:40], raw_dir) + "_idx.txt"
        assert qa_output_filename is not None

        with open(join(qa_dir, qa_output_filename)) as qa_output_file:
            output_text = (qa_output_file.read())
            prediction_ids = ([int(x.strip()) for x in output_text.split("\n") if x])

        
        cleaned_text = ([(x[0]['text'].strip() + x[1]['text'].strip()) for x in raw_sentences['paragraphs'][2:]])
        prediction_ids = [i for i in prediction_ids if i < len(cleaned_text)]
        # calculate F1 overlap score against annotation
        summed_annotations = Counter()
        summed_annotations.update([i[0] for x in annotation_ids for i in x])

        num_annotators = len(annotation_ids)

        _num_correct = sum([summed_annotations[idx] for idx in prediction_ids])
        _num_predicted = len(prediction_ids) * num_annotators
        _num_gold = sum([x[1] for x in summed_annotations.items()])

        num_correct += _num_correct
        num_predicted += _num_predicted
        num_gold += _num_gold

        for _annotation_ids in annotation_ids:
            annotation_ranking = [0] * len(cleaned_text)
            for index, priority in _annotation_ids:
                annotation_ranking[index] = priority

        annotator_top3_prediction_ids = [x[0] for x in summed_annotations.most_common(3)]
        annotator_prediction_text = ' '.join([cleaned_text[i] for i in annotator_top3_prediction_ids])



        model_prediction_text = ' '.join([cleaned_text[i] for i in prediction_ids])

        _rouge = scorer.score(annotator_prediction_text, model_prediction_text)
        rouge_1s.append(_rouge['rouge1'].fmeasure)
        rouge_2s.append(_rouge['rouge2'].fmeasure)
        rouge_Ls.append(_rouge['rougeL'].fmeasure)

    precision = num_correct / num_predicted
    recall = num_correct / num_gold
    F1 = 2*((precision * recall) / (precision + recall))

    return 100 * F1, np.nan, {'rouge1': mean(rouge_1s), 'rouge2': mean(rouge_2s), 'rougeL': mean(rouge_Ls)}, None


def evaluate_ctrlsum_output(annotations_filename, ctrlsum_dir, raw_dir):
    annotations = pd.read_csv(open(annotations_filename))

    # get unique articles
    seen = {}
    article_ids = [seen.setdefault(x, x) for x in list(annotations["HITId"]) if x not in seen] # deterministic
    
    # setup metrics
    num_correct = 0
    num_predicted = 0
    num_gold = 0

    spearmans = []

    rouge_1s = []
    rouge_2s = []
    rouge_Ls = []

    for id in article_ids:
        # get annotation ids
        raw_sentences = json.loads(list(annotations[annotations["HITId"] == id]["Input.articlejson"])[0])
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
        #print(raw_sentences['paragraphs'][2][0]['text'][:40])
        ctrlsum_output_filename = get_hashfilename(raw_sentences['paragraphs'][2][0]['text'][:40], raw_dir) + ".txt"
        assert ctrlsum_output_filename is not None

        with open(join(ctrlsum_dir, ctrlsum_output_filename)) as ctrlsum_output_file:
            output_text = (ctrlsum_output_file.read())

        cleaned_text = ([(x[0]['text'].strip() + x[1]['text'].strip()) for x in raw_sentences['paragraphs'][2:]])
        
        # calculate F1 overlap score against annotation
        summed_annotations = Counter()
        summed_annotations.update([i[0] for x in annotation_ids for i in x])

        annotator_top3_prediction_ids = [x[0] for x in summed_annotations.most_common(3)]
        annotator_prediction_text = ' '.join([cleaned_text[i] for i in annotator_top3_prediction_ids])

        model_prediction_text = output_text

        _rouge = scorer.score(annotator_prediction_text, model_prediction_text)
        rouge_1s.append(_rouge['rouge1'].fmeasure)
        rouge_2s.append(_rouge['rouge2'].fmeasure)
        rouge_Ls.append(_rouge['rougeL'].fmeasure)

    return None, None, {'rouge1': mean(rouge_1s), 'rouge2': mean(rouge_2s), 'rougeL': mean(rouge_Ls)}, None

    
# we accept a list of models, a list of keywords,t and an annotated dataset, and a raw dataset, and give spearman and overlap calculations, and visualizations
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-models", required=True, type=str)
    parser.add_argument("-models_dir", default='../models/', type=str)
    parser.add_argument("-keywords", required=False, type=str)
    parser.add_argument("-keyword_set", required=False, default=None, type=str)
    parser.add_argument("-annotations", required=True, type=str)
    parser.add_argument("-dataset", required=True, type=str)
    parser.add_argument("-dataset_dir", default="../data/", type=str)
    parser.add_argument("-map_path", required=True, type=str)
    parser.add_argument("-results_dir", default="../results/temp/", type=str)
    parser.add_argument("-step", default=28000, type=int)
    parser.add_argument('-overwrite', action='store_true')

    parser.add_argument("-visualize", action='store_true')

    parser.add_argument("-nonqf_model", default=None, type=str)
    parser.add_argument("-qa_dir", default=None, type=str)
    parser.add_argument("-raw_dir", default=None, type=str)
    parser.add_argument("-ctrlsum_dir", default=None, type=str)

    parser.add_argument("-raw_outputs_filename", default=None, type=str)
    args = parser.parse_args()

    if args.keyword_set:
        args.keywords = f"custom{','.join(args.keyword_set.split(','))}"
    elif not args.keyword_set:
        if args.keywords == 'geo':
            args.keyword_set = ",".join(["region", "location", "country", "geography", "miles"])
        elif args.keywords == 'rescue':
            args.keyword_set = ",".join(["recovery", "aid", "survivor", "injury", "death"])
        elif args.keywords == 'penalty':
            args.keyword_set = ",".join(['penalty', 'consequences', 'jailed', 'fined', 'court'])
        elif args.keywords == 'nature':
            args.keyword_set = ",".join(['amount', 'money', 'bank', 'stolen', 'time'])
    else:
        print(f"Must supply either -keywords or -keyword_set!")
        exit(1)
    print(args)
    time.sleep(2)
    args.models = args.models.split(',')
    name = f"{args.dataset}_{args.keywords}"

    build_dataset(args, name)
    result_paths = do_inference(args, name)

    outputs = []

    # evaluate the non-query model
    if args.nonqf_model:
        tempargs = copy.deepcopy(args)
        tempargs.models = [tempargs.nonqf_model]
        result_paths_nonqf = do_inference(tempargs, name + "_nonqf")
        F1, spearman, rouge12L, _ = evaluate_output(args.annotations, result_paths_nonqf[0] + ".outputs")
        outputs.append(["non-query model", F1, spearman, rouge12L['rouge1'], rouge12L['rouge2'], rouge12L['rougeL']])

    # evaluate the query models
    for model, result_path in zip(args.models, result_paths):
        F1, spearman, rouge12L, max_possible_F1 = evaluate_output(args.annotations, result_path + ".outputs", visualize=args.visualize)
        outputs.append([f"query ({model})", F1, spearman, rouge12L['rouge1'], rouge12L['rouge2'], rouge12L['rougeL']])
        

    cnndm_baseline_F1, _, cnndm_rouge12L, _ = evaluate_output(args.annotations, result_paths[0] + ".outputs.cnndm_baseline")
    keyword_baseline_F1, _, keyword_rouge12L, _ = evaluate_output(args.annotations, result_paths[0] + ".outputs.keyword_baseline")
    outputs.append(["cnndm baseline", cnndm_baseline_F1, np.nan, cnndm_rouge12L['rouge1'], cnndm_rouge12L['rouge2'], cnndm_rouge12L['rougeL']])
    outputs.append(["keyword baseline", keyword_baseline_F1, np.nan, keyword_rouge12L['rouge1'], keyword_rouge12L['rouge2'], keyword_rouge12L['rougeL']])

    # qa baseline
    if args.qa_dir:
        qa_baseline_F1, qa_baseline_spearman, qa_baseline_rouges, _ = evaluate_qa_output(args.annotations, args.qa_dir, args.raw_dir)
        outputs.append(["qa baseline", qa_baseline_F1, qa_baseline_spearman, qa_baseline_rouges['rouge1'], qa_baseline_rouges['rouge2'], qa_baseline_rouges['rougeL']])

    # ctrlsum baseline
    if args.ctrlsum_dir:
        _, _, ctrlsum_baseline_rouges, _ = evaluate_ctrlsum_output(args.annotations, args.ctrlsum_dir, args.raw_dir)
        outputs.append(["ctrlsum baseline", np.nan, np.nan, ctrlsum_baseline_rouges['rouge1'], ctrlsum_baseline_rouges['rouge2'], ctrlsum_baseline_rouges['rougeL']])

    if args.raw_outputs_filename:
        aspect_baseline_F1, aspect_baseline_spearman, aspect_baseline_rouges, _ = evaluate_output(args.annotations, args.raw_outputs_filename)
        outputs.append(["aspect", aspect_baseline_F1, aspect_baseline_spearman, aspect_baseline_rouges['rouge1'], aspect_baseline_rouges['rouge2'], aspect_baseline_rouges['rougeL']])
    print("\n\n")
    outputs = pd.DataFrame(outputs, columns=["Model", "F1", "Spearman", "ROUGE-1", "ROUGE-2", "ROUGE-L"]).round(3)
    print(outputs.to_string(index=False))
    print(f"max possible F1: {max_possible_F1:.3f}")
    
