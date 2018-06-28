import os
import sys

import functools
import argparse
import tensorflow as tf
import numpy as np
import jieba
import configparser
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

def cut(texts, use_stop=True, stop_list=None):

    if(not use_stop):
        assert(stop_list is not None)
    cut_texts = []
    for text in texts:
        cut_text = [word for word in jieba.cut(text) if use_stop or (word not in stop_list)]
        cut_texts.append(" ".join(cut_text))
    return cut_texts

def tokenize_and_padding(texts, max_length, vocab_size):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts)
    sequences = []
    for seq in tokenizer.texts_to_sequences_generator(texts):
        sequences.append(seq)
    sequences = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
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
                texts.append(lineSplit[0].split())
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

def load_stop_list(stop_list_filename="data/stop.txt"):
    with open(stop_list_filename, "r") as f:
        stops = f.readlines()
        stops = list(map(lambda x: x.rstrip("\n"), stops))

    return stops


def parser_config(config_filename, session):

    config = configparser.ConfigParser()
    config.read(config_filename)

    keys = config.options(session)

    confs = {}

    for key in keys:
        confs[key] = config.get(session, key)

    return confs

def str_to_list(s):
    target_list = [int(x) for x in s.split(',')]
    return target_list

def write(texts, labels, filename):
    L = len(texts)
    lines = []
    for i in range(L):
        lines.append(texts[i] + "\t" + str(labels[i]))

    with open(filename, "w") as f:
        f.write("\n".join(lines))

def dense_to_one_hot(labels_dense, num_classes):  
  num_labels = labels_dense.shape[0]  
  index_offset = np.arange(num_labels) * num_classes  
  labels_one_hot = np.zeros((num_labels, num_classes))  
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1  
  return labels_one_hot

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))

class Config(object):

    def __init__(self, config):
        for key in config:
            setattr(self, key, config[key])

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

class Attention_layer(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        super(Attention_layer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)

        a = K.exp(uit)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
