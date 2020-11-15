import tensorflow as tf
import numpy as np


############# Changes from Original Paper #######################
'''
1) Only included L2 regularization to kernel, not bias
2) Used Leaky-Relu instead of normal relu
3) Made this weird custom layer to handle all the different convolutional stuff
4) Didn't get predictions, only got scores, added get_predictions function that takes the logits
5) Changed from 2 class softmax to sigmoid output binary_crossentropy loss cuz it seemed pointless
6) This one was a bit confusing, so there's probably tons of errors in there

'''

class Highway(tf.keras.layers.Layer):
    def __init__(self, size, leaky_relu_alpha = 0.1, carry_bias=-2.0):
        super(Highway, self).__init__()

        self.leaky_relu_alpha = leaky_relu_alpha
        self.w_t = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight_transform")
        self.b_t = tf.Variable(tf.constant(carry_bias, shape=[size]), name="bias_transform")

        self.w = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight")
        self.b = tf.Variable(tf.constant(0.1, shape=[size]), name="bias")


    def call(self, inputs):

        t = tf.sigmoid(tf.matmul(inputs, self.w_t) + self.b_t, name="transform_gate")
        h = tf.keras.activations.relu((tf.matmul(inputs, self.w) + self.b),alpha = self.leaky_relu_alpha)
        c = tf.sub(1.0, t, name="carry_gate")
        # LOL thc

        return tf.add(tf.mul(self.h, self.t), tf.mul(x, self.c), "y")


class Convolutional_Bullshit(tf.keras.layers.Layer):
    def __init__(self, filter_sizes, num_filters, kernel_regularizer, table_len,
                 sequence_length, carry_bias, leaky_relu_alpha):
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.kernel_regularizer = kernel_regularizer
        self.table_len = table_len
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha = leaky_relu_alpha)
        self.seq_len = sequence_length
        self.carry_bias = carry_bias
    
    def call(self, inputs):
      pooled_outputs = []
      for filter_size, num_filters in zip(self.filter_sizes, self.num_filters):
        
        conv = tf.keras.layers.Conv2D(
                    filters = num_filters,
                    kernel_size = filter_size,
                    strides = 1,
                    padding='valid',
                    kernel_regularizer = self.kernel_regularizer)(inputs)
        h = self.leaky_relu(conv)
        pooled = tf.keras.layers.MaxPool2D(
            pool_sizes = (filter_size, self.table_len),
            strides = 1,
            padding = 'valid'
        )
        pooled_outputs.append(pooled)
      num_filters_total = sum(self.num_filters)
      h_pool = tf.concat(pooled_outputs, 3)
      h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
      h_highway = Highway(h_pool_flat, size = h_pool_flat.get_shape()[1], leaky_relu_alpha = 0.1, carry_bias=self.carry_bias)
      return h_highway



class Discriminator(object):
    def __init__(self, table_len, filter_sizes, num_filters, sequence_len = 128,
                    l2_reg_lambda=0.0, dropout_keep_prob = 1.0, learning_rate = 1e-4, 
                    leaky_relu_alpha = 0.1, carry_bias = 0, metrics = ['loss','val_loss']):
            self.table_len = table_len
            self.input_shape = [None, self.table_len]
            self.filter_sizes = filter_sizes
            self.num_filters = num_filters,
            self.dropout_keep_prob = dropout_keep_prob
            self.leaky_relu_alpha = leaky_relu_alpha
            self.seq_len = sequence_len
            self.opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
            self.metrics = metrics
            self.kernel_regularizer = tf.keras.regularizers.l2(l2 = l2_reg_lambda)
            self.carry_bias = carry_bias,
            self.loss = tf.keras.losses.categorical_crossentropy
            self.model = self.buid_model()

    def buid_model(self):
        model = tf.keras.Sequential()
        model.add(
              Convolutional_Bullshit(
                    filter_sizes= self.filter_size,
                    num_filters = self.num_filters,
                    kernel_regularizer = self.kernel_regularizer,
                    table_len = self.table_len,
                    sequence_length = self.seq_len,
                    carry_bias = self.carry_bias,
                    leaky_relu_alpha = self.leaky_relu_alpha
                )
            )
        model.add(
            tf.keras.layers.Dropout(rate = self.dropout_keep_prob)
        )
        model.add(
            tf.keras.layers.Dense(
                units = 1,
                activation='sigmoid'
            )
        )

        model.compile(
            optimizer = self.opt,
            loss = 'binary_crossentropy',
            metrics = self.metrics
        )
        return model

    def train(self, logits, labels, num_epochs, verbose = 1, callbacks=[], save_weights = False, filepath = ''):
        history = self.model.fit(
            x = logits,
            y = labels,
            epochs = num_epochs,
            verbose = verbose,
            callbacks = callbacks,
        )
        if (save_weights):
            save_weights(filepath)
        return history

    def load_weights(self, filepath):
        print("Loading weights from " + str(filepath))
        self.model.load_weights(filepath)
        print("Weights Loaded")
    
    def save_weights(self, filepath):
        print("Saving weights to " + str(filepath))
        self.model.save_weights(filepath)
        print("Weights Saved")
        
    def get_predictions(logits):

        return np.round(logits)

