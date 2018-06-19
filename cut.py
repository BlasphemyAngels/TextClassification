import os
import sys

from utils import read
from websearch import Segmenter
import argparse

if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument("--use_stop", type=bool, default=False, help="use or not stop words")

    args, _ = parser.parse_known_args()

    seg = Segmenter()
    texts, labels = read("./data/data.txt")
    cut_texts = [[token.word() for token in list(seg.cut_token(text)) if args.use_stop or (not token.is_stopword())] for text in texts]
    cut_texts = list(map(lambda x: " ".join(x), cut_texts))

    L = len(texts)

    lines = []
    for i in range(L):
        lines.append(cut_texts[i] + "\t" + str(labels[i]))

    with open("data/cut_data.txt", "w") as f:
        f.write("\n".join(lines))
    
