from dataloader_new import SmilesTokenizer
import numpy as np
import tensorflow as tf





############# Changes from Original Paper #######################
'''
1) I basically just copy and pasted the Generator class, because it basically works the same as generator pre-training
'''

class Oracle(object):
    def __init__(self, train_data_loader, validation_data_loader, units = 256, leaky_relu_alpha = 0.1, 
                num_layers = 1, opt = tf.keras.optimizers.Adam(lr=0.01), dropout_keep_prob = 1.0, l2_reg_lambda=0.0,
                loss = 'categorical_crossentropy',metrics = ['loss','val_loss']):
        
        assert num_layers >= 1

        self.st = SmilesTokenizer()
        self.train_dl = train_data_loader
        self.val_dl = validation_data_loader
        self.table_len = self.st.table_len
        self.opt = opt
        self.model = self.build_model()
        self.rewards = []
        self.num_layers = num_layers
        self.units = units
        self.leaky_alpha = leaky_relu_alpha
        self.loss = loss
        self.metrics = metrics
        self.dropout_keep_prob = dropout_keep_prob
        self.kernel_regularizer = tf.keras.regularizers.l2(l2 = l2_reg_lambda)
        
        
        
        
    def build_model(self, metrics = ['loss','val_loss'] ):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(
            input_shape = [None, self.table_len]
            ))
        if self.num_layers == 1:
            model.add(tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                        units = self.units,
                        return_sequences=False,
                        activation = 'tanh',
                        recurrent_activation = 'tanh',
                        kernel_regularizer = self.kernel_regularizer
            )))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.LeakyReLU(alpha = self.leaky_alpha))
            model.add( tf.keras.layers.Dropout(rate = self.dropout_keep_prob))
            model.add(tf.keras.layers.Dense(
                    units = self.table_len,
                    activation = 'softmax',
                    kernel_regularizer = self.kernel_regularizer
            ))
        else:
            for i in range(self.num_layers - 1):
                model.add(tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                            units = self.units,
                            return_sequences=True,
                            activation = 'tanh',
                            recurrent_activation = 'tanh',,
                            kernel_regularizer = self.kernel_regularizer
                )))
                model.add(tf.keras.layers.BatchNormalization())
                model.add( tf.keras.layers.Dropout(rate = self.dropout_keep_prob))
            model.add(tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                            units = self.units,
                            return_sequences=False,
                            activation = 'tanh',
                            recurrent_activation = 'tanh',
                            kernel_regularizer = self.kernel_regularizer
                )))
            model.add( tf.keras.layers.Dropout(rate = self.dropout_keep_prob))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.LeakyReLU(alpha = self.leaky_alpha))
            model.add(tf.keras.layers.Dense(
                    units = self.table_len,
                    activation = 'softmax',
                    kernel_regularizer = self.kernel_regularizer
            ))

        
        model.compile(
            optimizer = self.opt,
            loss = self.loss,
            metrics = self.metrics
        )
        return model
    
    def compile_model(self,model, loss = "categorical_crossentropy", metrics = ['loss','val_loss'] ):
        self.model.compile(
            optimizer = self.opt,
            loss = "categorical_crossentropy",
            metrics = metrics
        )
        print("self.model compiled!")

    def load_weights(self, filepath):
        print("Loading weights from " + str(filepath))
        self.model.load_weights(filepath)
        print("Weights Loaded")
    
    def save_weights(self, filepath):
        print("Saving weights to " + str(filepath))
        self.model.save_weights(filepath)
        print("Weights Saved")
    
    def train(self, num_epochs, verbose = 1, callbacks=[], save_weights = False, filepath = ''):
        history = self.model.fit(
            self.train_dl,
            steps_per_epoch = self.train_dl.__len__(),
            epochs = num_epochs,
            verbose = verbose,
            validation_data = self.valid_dl,
            validation_steps = self.valid_dl.__len__(),
            shuffle = True,
            callbacks = callbacks,
        )
        if (save_weights):
            save_weights(filepath)
        return history
            

        