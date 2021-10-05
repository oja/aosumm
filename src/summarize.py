import argparse
import subprocess
import random
import string
import os
import hashlib
import json
from pathlib import Path



def make_dir(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()

def create_singleton_dataset(text, name, directory, map_path):
    """
    Create a .story file within a raw dataset containing the given text.
    """
    make_dir(os.path.join(directory, name))
    with open(os.path.join(directory, name, hashhex(name)) + ".story", 'w') as f:
        f.write(text + "\n\n@highlight\nEmpty highlight.\n")

    make_dir(os.path.join(map_path, name))
    with open(os.path.join(map_path, name, "mapping_train.txt"), 'w') as f:
        f.write("")

    with open(os.path.join(map_path, name, "mapping_valid.txt"), 'w') as f:
        f.write("")

    with open(os.path.join(map_path, name, "mapping_test.txt"), 'w') as f:
        f.write(name + "\n")
    

def build_dataset(dataset, dataset_directory, map_path, keywords, debug=False):
    return subprocess.run(["python", 
                        "build.py", 
                        "-raw", dataset,
                        "-root", dataset_directory,
                        "-map_path", map_path,
                        "-name", dataset,
                        "-min_src_nsents", "0",
                        "-max_src_nsents", "10",
                        "-min_src_ntokens_per_sent", "0",
                        "-min_tgt_ntokens", "0",
                        "-qf", "-keywords", (keywords), '-overwrite'], stdout=subprocess.PIPE if debug else subprocess.DEVNULL, stderr=subprocess.PIPE if debug else subprocess.DEVNULL)

def do_inference(model, name, binary_data_directory, results_directory, debug=False, max_pos=512):
    result_path = os.path.join(results_directory, name)
    s = subprocess.run(["python", "train.py", 
                        "-task", "ext", 
                        "-mode", "test", 
                        "-batch_size", "1", 
                        "-test_batch_size", "1", 
                        "-bert_data_path", f"{binary_data_directory}/t",
                        "-log_file", "../logs/temp",
                        "-sep_optim", "true",
                        "-use_interval", "true",
                        "-visible_gpus", "1",
                        "-max_pos", f"{max_pos}",
                        "-max_length", "200",
                        "-alpha", "0.95",
                        "-min_length", "50",
                        "-result_path", result_path,
                        "-test_from", model], stdout=subprocess.PIPE if debug else subprocess.DEVNULL, stderr=subprocess.PIPE if debug else subprocess.DEVNULL)
    return result_path

def load_ctrlsum():
    from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast
    model = AutoModelForSeq2SeqLM.from_pretrained("hyunwoongko/ctrlsum-cnndm")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/ctrlsum-cnndm")

    return tokenizer, model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize a document conditioned on query keywords.')
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument("--model", default='qfsumm', type=str, help="name of the model to use. Can be 'qfsumm', 'ctrlsum', or a filepath to to a QFSumm-like pytorch checkpoint.", required=True)
    required.add_argument("--text", type=str, help="input document", required=True)
    required.add_argument("--keywords", help="comma-seperated keywords to use for query", required=True)
    optional.add_argument('--map-path', type=str, default="../urls", help="where to store temporary mapping")
    optional.add_argument("--dataset-dir", default="../data/", type=str, help="where to store raw, tokenized, and binarized data.")
    optional.add_argument("--results-dir", default="../results/", type=str, help="where to write model outputs.")
    optional.add_argument("--logs-dir", default="../logs/", type=str, help="where to store logs")
    optional.add_argument('--debug', action='store_true', help="print more verbose output for debugging")

    args = parser.parse_args()

    make_dir(args.dataset_dir)
    make_dir(os.path.join(args.dataset_dir, "raw"))
    make_dir(os.path.join(args.dataset_dir, "json"))
    make_dir(os.path.join(args.dataset_dir, "tokenized"))
    make_dir(os.path.join(args.dataset_dir, "binary"))

    make_dir(args.results_dir)

    make_dir(args.logs_dir)
    
    if args.model.lower() == 'ctrlsum':
        tokenizer, model = load_ctrlsum()
        keywords = " | ".join(args.keywords.split(","))
        data = tokenizer(f"{keywords} - {args.text}", return_tensors="pt")
        input_ids, attention_mask = data["input_ids"], data["attention_mask"]
        decoded = model.generate(input_ids, attention_mask=attention_mask, num_beams=5, min_length=100,max_length=150)
        summary = tokenizer.decode(decoded[0],skip_special_tokens=True)
        print(summary)
    else: 
        # Generate a random name for this run.
        name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
        
        if args.debug:
            print(f"Generated name: {name}")

        create_singleton_dataset(args.text, name, os.path.join(args.dataset_dir, "raw"), args.map_path)
        build_dataset(name, args.dataset_dir, os.path.join(args.map_path, name), args.keywords, args.debug)

        if args.model.lower() == "qfsumm":
            args.model = "../models/model_step_28000.pt"

        make_dir(os.path.join(args.results_dir, name))
        result_path = do_inference(args.model, name, os.path.join(args.dataset_dir, "binary", name), 
                                    os.path.join(args.results_dir, name), args.debug) + ".outputs"
        with open(result_path) as f:
            x = json.loads(f.read())[0]
            summary = ' '.join([x['text'][id] for id in x['ids']])
        print(summary)

