import os
import sys

import tensorflow as tf
from layers import *
from utils import labels_smooth

def batch_iter(data,batch_size):

    x, y = data
    x = np.array(x)
    y = np.array(y)
    data_size = len(x)
    num_batches_per_epoch = int((data_size - 1) / batch_size)
    for batch_index in range(num_batches_per_epoch):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_size)
        return_x = x[start_index:end_index]
        return_y = y[start_index:end_index]
        yield return_x, return_y

def train(x_train, y_train, x_predict, length, vocab_size, batch_size=512,hidden=96, head_nums=1, word_embedding_size=128, l2_reg_lambda=0.01, class_nums=2, init_learning_rate=0.0001, learning_rate_decay=0.9, epoches=5):

    inputs = tf.placeholder(tf.int32, [None, length], name="inputs")
    target = tf.placeholder(tf.int32, [None, 2], name="target")
    keep_prob = tf.placeholder(tf.float32)
    inputs_mask = tf.cast(inputs, tf.bool)
    inputs_len = tf.reduce_sum(tf.cast(inputs_mask, tf.int32), axis=1)
#  word_mat = tf.get_variable("word_mat", initializer=tf.constant(
    #  word_mat, dtype=tf.float32), trainable=False)

    word_mat = tf.Variable(tf.random_normal([vocab_size, word_embedding_size],
                                                -1.0, 1.0), name="embedding_matrix")

    input_embed = tf.nn.dropout(tf.nn.embedding_lookup(word_mat, inputs), 1.0 - keep_prob)
    input_embed = highway(input_embed, size=word_embedding_size, scope ="highway", dropout=keep_prob, reuse=None)



    out = residual_block(input_embed,
        num_blocks = 1,
        num_conv_layers = 2,
        kernel_size = 7,
        mask = inputs_mask,
        num_filters = word_embedding_size,
        num_heads = head_nums,
        seq_len = inputs_len,
        scope = "Encoder_Residual_Block",
        bias = False,
        dropout = keep_prob)

    out = tf.reshape(out, [-1, length * word_embedding_size])
    out = tf.layers.dense(out, class_nums, use_bias=True,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.constant_initializer(0.1),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda),
                                name="fc")

    y_train = labels_smooth(y_train, class_nums, 0.2)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=target, name="loss")

    loss = tf.reduce_mean(loss)
    loss += tf.losses.get_regularization_loss()

    global_step = tf.Variable(0, trainable=False)

    data_len = len(y_train)
    batch_nums = data_len // batch_size
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step=global_step, decay_steps=data_len // batch_size, decay_rate=learning_rate_decay)
    train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, global_step=global_step, name="adam-attn")
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        for counter in range(epoches):

            for batch_x, batch_y in batch_iter((x_train, y_train), batch_size):
                batch_loss, _ = sess.run([loss, train_op], feed_dict={inputs: batch_x, target: batch_y, keep_prob: 0.2})
            print(batch_loss)

            pred_label = tf.argmax(out, axis=1)
            pred = sess.run(pred_label, feed_dict={inputs:x_predict, keep_prob: 1.0})
            print(pred)
