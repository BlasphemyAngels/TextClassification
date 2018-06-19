import os
import sys

import functools
import argparse
import tensorflow as tf
import numpy as np


def cut(text):
    return jieba.cut(text)

def tokenize_and_padding(texts, max_length, vocab_size):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts)
    sequences = []
    for seq in tokenizer.texts_to_sequences_generator(texts):
        sequences.append(seq)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
    return sequences


def read(filename, cut=False):
    with open(filename, "r") as f:
        lines = f.readlines();
        lines = list(map(lambda x: x.rstrip("\n"), lines))
        texts = []
        labels = []
        for line in lines:
            lineSplit = line.split("\t")
            if cut:
                texts.append(lineSplit[0].split(" "))
            else:
                texts.append(lineSplit[0])
            labels.append(int(lineSplit[1]))
    return texts, labels

def doublewrap(function):
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

def gpu_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config

def labels_smooth(labels, class_nums, label_smooth_eps):

    n_sample = labels.shape[0]

    labels_ = np.eye(class_nums)[np.reshape(labels, (n_sample, ))]

    labels_ = list(labels_)
    def smooth_fn(x):
        return (1.0 - label_smooth_eps) * x + label_smooth_eps * 1.0 / class_nums

    labels_ = list(map(lambda x: [smooth_fn(x[0]), smooth_fn(x[1])], labels_))

    labels_ = np.array(labels_)
    return labels_
