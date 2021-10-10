import json
import argparse
import hashlib
import os
import shutil
from pathlib import Path
from os import listdir
from os.path import isfile, join, isdir
import random
import string
import xml.etree.ElementTree as ET


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-tac_docs", required=True, type=str)
    parser.add_argument("-tac_summs", required=True, type=str)
    parser.add_argument("-urls", required=True, type=str)
    parser.add_argument("-out", required=True, type=str)
    parser.add_argument('-overwrite', dest='overwrite', action='store_true', default=False)
    parser.add_argument('-size', type=int, required=True)
    args = parser.parse_args()
    
    if os.path.isdir(args.out):
        if not args.overwrite:
            print(f"found dir {args.out} already exists, pass -overwrite if you want")
            exit(1)
        else:
            shutil.rmtree(args.out)
    
    if os.path.isdir(args.urls):
        if not args.overwrite:
            print(f"found dir {args.urls} already exists, pass -overwrite if you want")
            exit(1)
        else:
            shutil.rmtree(args.urls)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.urls).mkdir(parents=True, exist_ok=True)

    

    def get_files_in_dir(dir):
        return [f for f in listdir(dir) if isfile(join(dir, f))]

    def get_dirs_in_dir(dir):
        return [f for f in listdir(dir) if isdir(join(dir, f))]
    
    summary_filenames = get_files_in_dir(args.tac_summs)

    summaries = []
    base_strings = []

    for document_group_name in get_dirs_in_dir(args.tac_docs):
        for document_group_AB_name in get_dirs_in_dir(os.path.join(args.tac_docs, document_group_name)):
            # get the whole document input
            base_strings = []
            summaries_ = []

            document_sentences = []
            for document_name in get_files_in_dir(os.path.join(args.tac_docs, document_group_name, document_group_AB_name)):
                filepath = os.path.join(args.tac_docs, document_group_name, document_group_AB_name, document_name)
                # map the document to the appropriate summaries, append it to json
                
                doc_text = '''<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
                            "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd" [
                            <!ENTITY Cx1f ' '>
                            <!ENTITY Cx13 ' '>
                            <!ENTITY Cx11 ' '>
                            ]>'''  # You can define more entities here, if needed
                with open(filepath, 'r') as f:
                    doc_text += f.read()

                root = ET.fromstring(doc_text)
                for body in root:
                    if body.tag == 'BODY':
                        for text in body:
                            if text.tag == 'TEXT':
                                for p in text:
                                    if p.tag == 'P':
                                        document_sentences.append(p.text.strip())

            document_sentences = [x for x in document_sentences][:args.size]
            
            if len(document_sentences) > 5:
                text = "\n".join(document_sentences)
                text += f"\n\n@highlight\n\nThis is a dummy annotation."

                # get the summaries
                spliced_name = (document_group_AB_name[:-3] + document_group_AB_name[-2:])
                matched_summary_filenames = ([x for x in summary_filenames if spliced_name in x])

                summaries_ = []
                for filename in matched_summary_filenames:
                    with open(os.path.join(args.tac_summs, filename), 'r') as f:
                        summaries_.append(f.read().strip())

                base_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=30))
                base_strings.append(base_string)
                with open(os.path.join(args.out, hashhex(base_string) + ".story"), 'w') as w:
                    print(text, file=w)

                summary_json = {"anchor": document_sentences[0], "summaries": summaries_}
                summaries.append(summary_json)
        
    with open(os.path.join(args.urls, "mapping_test.txt"), "w") as f:
        print("\n".join(base_strings), file=f)

    with open(os.path.join(args.urls, "mapping_train.txt"), "w") as f:
        print("\n", file=f)

    with open(os.path.join(args.urls, "mapping_valid.txt"), "w") as f:
        print("\n", file=f)
    
    with open(os.path.join(args.out, "summaries.json"), "w") as f:
        print(json.dumps(summaries), file=f)
