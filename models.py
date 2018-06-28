from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras import regularizers
from keras.layers import Flatten
from keras.layers import Convolution1D
from keras.layers import MaxPool1D
from keras.layers import Dropout
from keras.layers import concatenate
from keras.layers import GRU
from keras.layers import Conv1D
from keras.layers import Bidirectional
from keras.layers import Permute
from keras.layers import merge
from keras.models import Model
from keras.layers import Reshape
from keras.layers import RepeatVector
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Lambda
import keras.backend as K
K.set_image_dim_ordering("tf")

from utils import dense_to_one_hot
from utils import labels_smooth
from utils import Attention_layer
from utils import str_to_list

def lstm(args):
    print('Build model...')    
    vocab_size = int(args.vocab_size)
    embedding_size = int(args.embedding_size)
    dropout_prob = float(args.dropout_prob)
    l2_reg_scala = float(args.l2_reg_scala)
    units = int(args.lstm_units)
    print('Build model...')    
    model = Sequential()
    model.add(Embedding(vocab_size + 2, embedding_size, mask_zero=False))
    model.add(LSTM(units, dropout=dropout_prob, recurrent_dropout=dropout_prob))
    model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(l2_reg_scala), bias_regularizer=regularizers.l2(l2_reg_scala)))
    model.summary()

    return model

def bilstm(args):
    print('Build model...')    
    vocab_size = int(args.vocab_size)
    embedding_size = int(args.embedding_size)
    units = int(args.lstm_units)
    dropout_prob = float(args.dropout_prob)
    l2_reg_scala = float(args.l2_reg_scala)
    length = int(args.length)
    model = Sequential()
    model.add(Embedding(vocab_size + 1, embedding_size, input_length=length, mask_zero=False))
    model.add(Bidirectional(LSTM(units, dropout=dropout_prob, recurrent_dropout=dropout_prob, return_sequences=True)))
    model.add(Permute([2, 1]))
    model.add(Conv1D(1, 1, padding='valid'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(l2_reg_scala), bias_regularizer=regularizers.l2(l2_reg_scala)))
    model.summary()
    return model

def bilstm_att(args):
    print('Build model...')
    vocab_size = int(args.vocab_size)
    embedding_size = int(args.embedding_size)
    units = int(args.lstm_units)
    dropout_prob = float(args.dropout_prob)
    l2_reg_scala = float(args.l2_reg_scala)
    length = int(args.length)

    _input = Input(shape=[length], name='input', dtype='int32')

    embed = Embedding(vocab_size+1, embedding_size)(_input)
    activations = Bidirectional(LSTM(units, dropout=dropout_prob, recurrent_dropout=dropout_prob, return_sequences=True))(embed)

    attention = Dense(1, activation='tanh')(activations)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)

    attention = RepeatVector(2*units)(attention)
    activations = Permute([2, 1])(activations)

    sent_representation = merge([activations, attention], mode='mul')

    sent_representation = Lambda(lambda xin: K.sum(xin, axis=2), output_shape=(2*units,))(sent_representation)

    prob = Dense(2, activation='softmax')(sent_representation)

    model = Model(inputs=_input, outputs=prob)

    model.summary()
    return model

def gru(args):
    print('Build model...')    
    vocab_size = int(args.vocab_size)
    embedding_size = int(args.embedding_size)
    dropout_prob = float(args.dropout_prob)
    l2_reg_scala = float(args.l2_reg_scala)
    units = int(args.lstm_units)
    model = Sequential()
    model.add(Embedding(vocab_size+1, embedding_size, mask_zero=False))
    model.add(GRU(embedding_size, dropout=dropout_prob, recurrent_dropout=dropout_prob))
    model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(l2_reg_scala), bias_regularizer=regularizers.l2(l2_reg_scala)))
    model.summary()
    return model

def bilstm_att2(args):
    print('Build model')
    vocab_size = int(args.vocab_size)
    embedding_size = int(args.embedding_size)
    units = int(args.lstm_units)
    dropout_prob = float(args.dropout_prob)
    l2_reg_scala = float(args.l2_reg_scala)
    length = int(args.length)
    model = Sequential()
    model.add(Embedding(vocab_size + 1, embedding_size, input_length=length, mask_zero=False))
    model.add(Bidirectional(GRU(units, dropout=dropout_prob, recurrent_dropout=dropout_prob, return_sequences=True)))
    model.add(Attention_layer())
    model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(l2_reg_scala), bias_regularizer=regularizers.l2(l2_reg_scala)))
    model.summary()
    return model

def mlp(args):
    print('Build model')
    model = Sequential()
    vocab_size = int(args.vocab_size)
    units = int(args.units)
    dropout_prob = float(args.dropout_prob)
    l2_reg_scala = float(args.l2_reg_scala)
    model.add(Dense(units, input_shape=(vocab_size+1,), activation='relu'))
    model.add(Dropout(dropout_prob))
    model.add(Dense(2, activation='softmax'))
    model.summary()

    return model

def textcnn(args):
    print('Build model')
    vocab_size = int(args.vocab_size)
    embedding_size = int(args.embedding_size)
    dropout_prob = float(args.dropout_prob)
    l2_reg_scala = float(args.l2_reg_scala)
    length = int(args.length)

    filter_sizes = str_to_list(args.filter_sizes)
    filter_nums = str_to_list(args.filter_nums)

    main_input = Input(shape=(length,),dtype='int32')
    embed = Embedding(vocab_size+1, embedding_size)(main_input)

    cnn_outs = []

    for i, filter_size in enumerate(filter_sizes):
        filter_num = filter_nums[i]
        cnn = Convolution1D(filter_num, filter_size, padding='valid', strides=1, activation='relu')(embed)
        cnn = MaxPool1D(pool_size=length-filter_size+1)(cnn)
        cnn_outs.append(cnn)

    out = concatenate(cnn_outs, axis=-1)
    flat = Flatten()(out)

    drop = Dropout(dropout_prob)(flat)
    main_output = Dense(2, activation='softmax')(drop)

    model = Model(inputs=main_input, outputs=main_output)
    model.summary()
    return model
