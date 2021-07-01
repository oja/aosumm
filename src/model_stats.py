import argparse
import json
from statistics import mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-output1", required=True, type=str)
    parser.add_argument("-output2", required=True, type=str)
    args = parser.parse_args()

    jaccard_scores = []
    set_matches = 0
    exact_matches = 0

    with open(args.output1) as output1, open(args.output2) as output2:
        output1_json = json.load(output1)
        output2_json = json.load(output2)

        assert(len(output1_json) == len(output2_json))

        for i, sample1 in enumerate(output1_json):
            for sample2 in output2_json:
                if sample2['text'] == sample1['text']:
                    sample2 = sample2
                    break
            assert(sample1['text'] == sample2['text'])
            set1 = set(sample1['ids'])
            set2 = set(sample2['ids'])

            jaccard_scores.append(len(set1.intersection(set2)) / len(set1.union(set2)))
            if set1 == set2:
                set_matches += 1

            if sample1['ids'] == sample2['ids']:
                exact_matches += 1

        print(f"mean jaccard index: {mean(jaccard_scores):.3f}")
        print(f"{100 * set_matches / len(output1_json)}% of predictions were identical")
