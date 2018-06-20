import os
import sys

import re
import logging
import argparse
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np

from utils import read
from utils import tokenize_and_padding

from model import Model

if __name__ == '__main__':

    logging.basicConfig(format="%s(asctime)s : %(levelname)s : %(message)s")
    logging.root.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True, help="path of data")
    parser.add_argument("--checkpoints", type=str, default="checkpoints", help="checkpoings directory")
    parser.add_argument("--k_fold", type=int, default=8, help="the k of KFold")
    parser.add_argument("--vocab_size", type=int, default=8000, help="the size of the vocabulary")
    parser.add_argument("--model", type=str, required=True, help="model name")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--embedding_size", type=int, default=200, help="embedding size")
    parser.add_argument("--epoch", type=int, default=200, help="epoch")

    args, _ = parser.parse_known_args()


    # read data
    texts, labels = read(args.data_path)

    # tokenize
    max_length = max(list(map(lambda text: len(text.split()), texts)))
    tokenized_texts = tokenize_and_padding(texts, max_length, args.vocab_size)
    labels = labels

    print("source data size:", len(tokenized_texts))

    # clean
    data = zip(tokenized_texts, labels)
    data_ = []
    for d in data:
        if(np.max(d[0]) != 0):
            data_.append(d)

    print("data size:", len(data_))
    tokenized_texts, labels = zip(*data_)

    tokenized_texts = np.array(tokenized_texts)
    labels = np.array(labels)

    kf = KFold(n_splits=args.k_fold)

    if not os.path.exists(args.checkpoints):
        os.mkdir(args.checkpoints)

    model_dir_basename = "model-0"

    model = Model(args.model, max_length, args.vocab_size, args.embedding_size, [2, 3, 5], [20, 20, 20], 0.3, 0.5, batch_size=args.batch_size, epoch=args.epoch)

    counter = 1
    for train_index, test_index in kf.split(tokenized_texts):

        print("Train:", train_index, "Test:", test_index)

        model_dir = model_dir_basename + str(counter)

        counter += 1

        text_train, text_test = tokenized_texts[train_index], tokenized_texts[test_index]
        label_train, label_test = labels[train_index], labels[test_index]

        model.train(text_train, label_train, text_test, label_test, model_dir)
