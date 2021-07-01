import json
import pandas as pd
import argparse
import os

articles = []
n = 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", required=True, type=str)
    parser.add_argument("-out", required=True, type=str)
    args = parser.parse_args()

    # verify that args.in is a directory 
    if not os.path.isdir(args.input):
        print("input is not a directory")
        exit(1)

    articles = []
    for filename in ([x for x in os.listdir(args.input) if "tfidf-tokens" not in x]):
        filepath = os.path.join(args.input, filename)
        article = {}
        article["paragraphs"] = [[]]

        with open(filepath) as f:
            j = json.load(f)
            done = False
            if j is not None:
                article["paragraphs"].append([{"number": 1,
                                            "text": "Article",
                                            "label": "title"}])

                number = 2
                for sentence in j['sentences'][:10]:
                    sentence_txt = ""
                    for token in (sentence['tokens']):
                        if token['word'] == "@highlight":
                            done = True
                            break
                        sentence_txt += token['word'] + token['after']
                    if done:
                        break
                    article["paragraphs"].append([{"number": number,
                                                 "text": sentence_txt.strip('\n'),
                                                 "label": "sentence"},
                                                 {"number": number + 1,
                                                 "text": " ",
                                                 "label": "no-unit"}])
                    number += 2
                articles.append(json.dumps(article))

    articles = pd.DataFrame({"articlejson": pd.Series(articles)})
    articles.to_csv(args.out, index=False)
    
