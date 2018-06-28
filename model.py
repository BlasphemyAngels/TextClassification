import os
import sys
import numpy as np
import tensorflow as tf

from utils import define_scope
from utils import gpu_config
from utils import labels_smooth
from utils import parser_config
from config import Config

from methods import Methods

class Model(object):
    """
        Text classification.
    """
    def __init__(self, model_name, config_filename):
        """
            initialization function
        """

        self._config_filename = config_filename

        self._graph = tf.Graph()
        self._model_name = model_name
        self._mode = "train"

        main_config = parser_config(self._config_filename, "main")
        model_config = parser_config(self._config_filename, self._model_name)

        configs = {**main_config, **model_config}

        self._transforms_method = Methods("transform")

        self._transforms_method.register("textcnn", "transform.textcnn")
        self._transforms_method.register("main", "transform.main")

        self._transforms_method.exe(self._model_name, config=configs, config_filename=self._config_filename)
        self._transforms_method.exe("main", config=configs, config_filename=self._config_filename)

        self._config = Config(configs)

        with self._graph.as_default():
            self._input = tf.placeholder(tf.int32, [None, self._config.text_length], name="input")
            self._target = tf.placeholder(tf.float32, [None, 2], name="target")



    @define_scope
    def embedding(self):
        with self._graph.as_default():
            embed_matrix = tf.Variable(tf.random_normal([self._config.vocab_size, self._config.embedding_size],
                                                        -1.0, 1.0), name="embedding_matrix")
            embed_input = tf.nn.embedding_lookup(embed_matrix, self._input)
            embed_input = tf.expand_dims(embed_input, axis=-1)
        return embed_input

    def textcnn(self):

        with self._graph.as_default():
            conv_and_pool_outs = []
            # conv
            for filter_id, filter_size in enumerate(self._config.filters_size):
                f_shape = [filter_size, self._config.embedding_size, 1, self._config.filters_num[filter_id]]
                filter_variable = tf.Variable(tf.truncated_normal(f_shape, stddev=0.1), name="filter"+str(filter_id)+"-variable")
                filter_bias = tf.Variable(tf.constant(0.1, shape=[self._config.filters_num[filter_id]]), name="filter"+str(filter_id)+"-bias")
                conv = tf.nn.conv2d(self.embedding, filter_variable, strides=[1, 1, 1, 1], padding="VALID", name="conv"+str(filter_id))

                conv = tf.nn.bias_add(conv, filter_bias, name="conv-biasadd"+str(filter_id))

                conv = tf.nn.relu(conv, name="conv-relu-"+str(filter_id))

                pool = tf.nn.max_pool(conv, ksize=[1, self._config.text_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool-"+str(filter_id))

                conv_and_pool_outs.append(pool)
            out = tf.concat(conv_and_pool_outs, 3)
            num_of_filters = sum(self._config.filters_num)
            out = tf.reshape(out, [-1, num_of_filters], name="reshape")

            if self._mode == "train":
                dropout_out = tf.nn.dropout(out, keep_prob=self._config.dropout_prob, name="dropout")

            final_out = tf.layers.dense(out, self._config.class_nums, use_bias=True,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        bias_initializer=tf.constant_initializer(0.1),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._config.l2_reg_lambda),
                                        bias_regularizer=tf.contrib.layers.l2_regularizer(scale=self._config.l2_reg_lambda),
                                        name="fc")
        return final_out
    @define_scope
    def bilstm(self):
        pass

    def lstm(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=10, state_is_tuple=True)

        #  if self.mode == "train":
          #  lstm_cell = tf.contrib.rnn.DropoutWrapper(
              #  lstm_cell,
              #  input_keep_prob=self._lstm_config.lstm_dropout_keep_prob,
              #  output_keep_prob=self._lstm_config.lstm_dropout_keep_prob)
#
        #  initial_state = lstm_cell.zero_state(self._batch_size, dtype=tf.float32)

        lstm_cell.reuse_variables()



    @define_scope
    def prediction(self):
        return getattr(self, self._model_name)()

    @define_scope
    def loss(self):
        with self._graph.as_default():
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction, labels=self._target, name="loss")
            loss = tf.reduce_mean(loss, name="reduce_mean")
            loss += tf.losses.get_regularization_loss()
        return loss

    @define_scope
    def accurcy(self):
        with self._graph.as_default():
            predict_class = tf.argmax(self.prediction, 1, name="predict_class")
            real_class = tf.argmax(self._target, 1, name="real_class")
            correct = tf.equal(predict_class, real_class)
            correct = tf.cast(correct, tf.float32, name="cast")
        return tf.reduce_mean(correct, name="acc")


    def train(self, text_train, label_train, text_test, label_test, model_dir):

        label_test = np.reshape(label_test, [-1, 1])
        label_test = labels_smooth(label_test, self._config.class_nums, self._config.label_smooth_eps)
        with self._graph.as_default():
            dataset = tf.data.Dataset.from_tensor_slices((text_train, label_train))
            batch_dataset = dataset.batch(batch_size=self._config.batch_size)
            repeat_dataset = batch_dataset.repeat(self._config.epoch)
            data_iterator = repeat_dataset.make_one_shot_iterator()
            next_batch_text, next_batch_label = data_iterator.get_next()

            global_step = tf.Variable(0, trainable=False)
            data_len = len(label_train)
            batch_nums = data_len // self._config.batch_size
            learning_rate = tf.train.exponential_decay(self._config.init_learning_rate, global_step=global_step, decay_steps=data_len // self._config.batch_size, decay_rate=self._config.learning_rate_decay) 
            train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.loss, global_step=global_step, name="adam-textcnn")
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            with tf.Session(config=gpu_config()) as sess:
                sess.run(init_op)
                counter = 1
                while True:
                    try:
                        counter += 1
                        self._mode = "train"
                        x, y = sess.run([next_batch_text, next_batch_label])
                        y = np.reshape(y, [-1, 1]).astype(np.int32)
                        y = labels_smooth(y, self._config.class_nums, self._config.label_smooth_eps)
                        loss, _ = sess.run([self.loss, train_op], feed_dict={self._input: x, self._target: y})
                        if counter % batch_nums == 0:
                            print("Epoch %d loss: %lf" % ((counter // batch_nums), loss))
                            self._mode = "test"
                            accurcy = sess.run(self.accurcy, feed_dict={self._input: text_test, self._target: label_test})
                            print("Test accurcy:", accurcy)
                    except tf.errors.OutOfRangeError:
                        break
