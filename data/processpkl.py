import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import pickle as pkl


with open("./cut_data.txt", "r") as f:
    data = f.readlines()
    data = list(map(lambda x: x.rstrip(), data))

    data = list(map(lambda x: x.split("\t"), data))

    texts, labels = zip(*data)

    labels = list(map(lambda x: int(x), labels))

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = []
    for seq in tokenizer.texts_to_sequences_generator(texts):
        sequences.append(seq)

    train_x, test_x, train_y, test_y = train_test_split(sequences, labels, test_size=0.2, random_state=0)


    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    train = np.array([train_x, train_y])
    test = np.array([test_x, test_y])

    with open("train", "wb") as f:
        pkl.dump(train, f)
        pkl.dump(test, f)
