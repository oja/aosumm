import argparse

from os import listdir
from os.path import isfile, join


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir1", required=True, type=str)
    parser.add_argument("-dir2", required=True, type=str)
    args = parser.parse_args()

    filenames = [f for f in listdir(args.dir1) if isfile(join(args.dir1, f))]

    for filename in filenames:
        print(f"from: {filename}")
        print("================ Article =================")

        with open(join(args.dir2, filename.replace(".txt", ""))) as f:
            print(f.read().split("@highlight")[0])

        print("================ CTRLSum =================")

        with open(join(args.dir1, filename)) as f:
            print(f.read())

        
        print("\n")
