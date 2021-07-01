import json
from collections import defaultdict
import itertools
from scipy import stats
import statistics

from pytorch_transformers import file_utils

filepaths = {
    "empty": "k15_eqempty.outputs",
    "manual": "k15_eqmanual.outputs",
    "tfidf": "k15_eqtfidf.outputs",
    "rescue": "k15_eqrescue.outputs",
    "geo": "k15_eqgeo.outputs"
}

predictions = defaultdict(list)

max_id = float('-inf')
for klass in filepaths.keys():
    filepath = filepaths[klass]
    with open(filepath) as f:
        data = json.load(f)
        for article in data:
            ids = article['ids']
            max_id = max(max_id, max(ids))
            predictions[klass].append(ids)

def scatter(ids):
    s = [max_id] * (max_id + 1)
    for i, idx in enumerate(ids):
        s[idx] = i
    return s

for x, y in (itertools.combinations(filepaths.keys(), 2)):
    spearman_scores = []
    jaccard_scores = []
    pred_x, pred_y = predictions[x], predictions[y]
    exact_match = 0
    for i, j in zip(pred_x, pred_y):
        spearman_scores.append(stats.spearmanr(scatter(i), scatter(j))[0])
        jaccard_scores.append(len(set(i).intersection(set(j))) / len(set(i).union(set(j))))
        if set(i) == set(j):
            exact_match += 1
    print(f"({x}, {y}) \n\t spearman: mean {statistics.mean(spearman_scores):.5f}, stddev {statistics.stdev(spearman_scores):.5f}")
    print(f"\t jaccard: mean {statistics.mean(jaccard_scores):.5f}, stddev {statistics.stdev(jaccard_scores):.5f}")
    print(f"\t exact matches: {exact_match / len(pred_x):.5f}")
    
    