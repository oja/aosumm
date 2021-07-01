# QFSumm


## Setup
Dependencies can be installed with `pip3 install -r requirements.txt`.

## Inference
`summarize.py` is a wrapper script located within the `src/` directory. It allows for summarization of documents using either QFSumm, CTRLSum, or a custom PyTorch checkpoint. Usage is as follows:

```
usage: summarize.py [-h] --model MODEL --text TEXT --keywords KEYWORDS [--map-path MAP_PATH] [--dataset-dir DATASET_DIR] [--results-dir RESULTS_DIR] [--logs-dir LOGS_DIR] [--debug]

Summarize a document conditioned on query keywords.

required arguments:
  --model MODEL         name of the model to use. Can be 'qfsumm', 'ctrlsum', or a filepath to to a QFSumm-like pytorch checkpoint.
  --text TEXT           input document
  --keywords KEYWORDS   comma-seperated keywords to use for query

optional arguments:
  --map-path MAP_PATH   where to store temporary mapping
  --dataset-dir DATASET_DIR
                        where to store raw, tokenized, and binarized data.
  --results-dir RESULTS_DIR
                        where to write model outputs.
  --logs-dir LOGS_DIR   where to store logs
  --debug               print more verbose output for debugging
```

Note that this script calls other subprocesses and must be run directly from the same working directory `src`. Sample usage of the script: 
```shell
python summarize.py --model qfsumm --keywords disturbance,islands,hurricane \
 --text "The 2021 Atlantic hurricane season is heating up early as the National Hurricane Center (NHC) monitors two different areas of potential development, and it's still June. Next up: Tropical Storm Elsa. Social media is awash with memes of the next named storm, which shares the same name as Disney's fictional character from the movie 'Frozen.' It may crack a smile for some parents, or even the weather-savvy 5-year-old, but this is one to watch closely. While the nearest area of activity (currently identified as invest 95L) has major hurdles to overcome in the days ahead, the NHC designated the next wave as 'Potential Tropical Cyclone Five' Wednesday afternoon. This tropical disturbance is currently about 1,200 miles east of the Windward Islands. Although any potential interaction with the US mainland wouldn't occur until the early to middle part of next week, this disturbance is becoming better organized as the hours pass and currently poses a more immediate threat to the Windward and Leeward Islands." 

The 2021 Atlantic hurricane season is heating up early as the National Hurricane Center ( NHC ) monitors two different areas of potential development , and it 's still June . This tropical disturbance is currently about 1,200 miles east of the Windward Islands . Although any potential interaction with the US mainland would n't occur until the early to middle part of next week , this disturbance is becoming better organized as the hours pass and currently poses a more immediate threat to the Windward and Leeward Islands .
```

## Training
Training procedure is similar to [BertSum](https://github.com/nlpyang/PreSumm), but instead we train on query-augmented data. To generate this data, place your raw training data, consisting of .story files into a directory, such as `./data/raw` (or one you specify). 

Call the wrapper script `build.py` with the query-focused flag `-qf`. This will create a binary dataset in `../data/binary` that a model can be trained on.

```
usage: build.py [-h] [-pretrained_model PRETRAINED_MODEL] [-map_path MAP_PATH] -root ROOT -raw RAW -name NAME [-overwrite] [-shard_size SHARD_SIZE] [-min_src_nsents MIN_SRC_NSENTS] [-max_src_nsents MAX_SRC_NSENTS]
                [-min_src_ntokens_per_sent MIN_SRC_NTOKENS_PER_SENT] [-max_src_ntokens_per_sent MAX_SRC_NTOKENS_PER_SENT] [-min_tgt_ntokens MIN_TGT_NTOKENS] [-max_tgt_ntokens MAX_TGT_NTOKENS] [-summary_size SUMMARY_SIZE]
                [-lower [LOWER]] [-use_bert_basic_tokenizer [USE_BERT_BASIC_TOKENIZER]] [-log_file LOG_FILE] [-n_cpus N_CPUS] [-qf [QF]] [-keywords KEYWORDS] [-contrastive {none,binary}] [-intensity INTENSITY]
                [-bertscore [BERTSCORE]] [-dataset DATASET]

Create a query-focused dataset by preprocessing, tokenizing, and binarizing a given raw dataset.

optional arguments:
  -h, --help            show this help message and exit
  -pretrained_model PRETRAINED_MODEL
                        which pretrained model to use
  -map_path MAP_PATH
  -root ROOT            location of root directory for data
  -raw RAW              name of raw directory within the root directory
  -name NAME            name of the generated datset
  -overwrite            overwrite existing datasets that have the same name
  -shard_size SHARD_SIZE
  -min_src_nsents MIN_SRC_NSENTS
  -max_src_nsents MAX_SRC_NSENTS
  -min_src_ntokens_per_sent MIN_SRC_NTOKENS_PER_SENT
  -max_src_ntokens_per_sent MAX_SRC_NTOKENS_PER_SENT
  -min_tgt_ntokens MIN_TGT_NTOKENS
  -max_tgt_ntokens MAX_TGT_NTOKENS
  -summary_size SUMMARY_SIZE
  -lower [LOWER]
  -use_bert_basic_tokenizer [USE_BERT_BASIC_TOKENIZER]
  -log_file LOG_FILE
  -n_cpus N_CPUS
  -qf [QF]              generate a query-focused dataset
  -keywords KEYWORDS    (useful for eval) train on these supplied keywords, otherwise use TF-IDF keywords
  -contrastive {none,binary}
                        whether to use contrastive training
  -intensity INTENSITY  intensity of oracle summary modification
  -bertscore [BERTSCORE]
                        whether to use bertscore instead of rougescore
  -dataset DATASET
```
