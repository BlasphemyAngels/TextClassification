import os
import sys

import re
import logging
import argparse
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

from utils import read
from utils import tokenize_and_padding
from utils import labels_smooth
from utils import parser_config
from utils import Config
from utils import f1
from utils import precision
from utils import merge_two_dicts
from utils import recall

if __name__ == '__main__':

    logging.basicConfig(format="%s(asctime)s : %(levelname)s : %(message)s")
    logging.root.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True, help="path of data")
    parser.add_argument("--model", type=str, required=True, help="model name")
    parser.add_argument("--config_filename", type=str, default="config", help="the filename of config file")

    args, _ = parser.parse_known_args()

    # get config

    model_config_name = args.model

    if "lstm" in args.model:
        model_config_name = "lstm"
    main_config = parser_config(args.config_filename, "main")
    model_config = parser_config(args.config_filename, model_config_name)

    #  configs = {**main_config, **model_config}
    configs = merge_two_dicts(main_config, model_config)
    configs = Config(configs)


    texts, labels = read(args.data_path)

    with open("./tmp_test_content", "r") as f:
        pred = f.readlines()
        pred = list(map(lambda x: x.rstrip("\n"), pred))

    #  vocab = {}
#
    #  for tt in texts + pred:
        #  for ttt in tt.split():
            #  if(ttt not in vocab):
                #  vocab[ttt] = 1
            #  else:
                #  vocab[ttt] += 1
    #  print(len(vocab))

    # tokenize

    #  max_length1 = max(list(map(lambda text: len(text.split()), texts)))
    #  max_length2 = max(list(map(lambda x: len(x.split()), pred)))
    #  max_length = max(max_length1, max_length2)

    tokenized_texts = tokenize_and_padding(texts + pred, int(configs.length), int(configs.vocab_size))
    tokenized_pred = tokenized_texts[len(texts):]
    tokenized_texts = tokenized_texts[:len(texts)]
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

    tokenized_pred = np.array(tokenized_pred)

    labels = labels_smooth(labels, int(configs.class_nums), float(configs.label_smooth_eps))

    models = __import__("models")
    if not hasattr(models, args.model):
        logging.error("The model %s not exist" % args.model)
        sys.exit(1)
    model_fn = getattr(models, args.model)

    model = model_fn(configs)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[precision, recall, f1])

    print('Train...')

    x_train, x_test, y_train, y_test = train_test_split(tokenized_texts, labels, test_size=0.2, random_state=0)

    x_train = tokenized_texts
    y_train = labels

    model.fit(x_train, y_train, batch_size=int(configs.batch_size), epochs=int(configs.epochs), validation_data=(x_test, y_test))

    score, prec, rec, fs = model.evaluate(x_test, y_test, batch_size=int(configs.batch_size))

    pred = model.predict(tokenized_pred)

    #  pred = np.argmax(pred, axis=1)
#
    #  pred = list(pred)

    bad_list = [29, 35, 46, 53, 61, 93, 177, 189, 210, 227, 288]

    #  print("pred", pred)

    print('Test precision:', prec)
    print('Test recall:', rec)
    print('Test f1 score:', fs)

    #  with open("./tmp_test_content.txt", "r") as f:
        #  pred_data = f.readlines()
        #  pred_data = list(map(lambda x: x.rstrip("\n"), pred_data))
        #  for id_, _ in enumerate(pred):
            #  if _ == 1:
                #  print(pred_data[id_])
    res = []
    with open("./tmp_test_content.txt", "r") as f:
        pred_data = f.readlines()
        pred_data = list(map(lambda x: x.rstrip("\n"), pred_data))
        for _ in bad_list:
            print(pred_data[_ - 1].decode("utf-8"))
            print(pred[_ - 1])
            if(pred[_ - 1][0] > pred[_ - 1][1]):
                res.append(0)
            else:
                res.append(1)

    true_pred = sum(res)

    val_recall = true_pred * 1.0 / len(res)

    pred_class = np.argmax(pred, axis=1)

    pred_pos = sum(pred_class)

    val_precision = 1.0 * true_pred / (pred_pos + 0.0001)

    print("val recall:", val_recall)
    print("val precision:", val_precision)

    for i, p in enumerate(pred):
        if p < 0.99:
            print(pred_data[i])
