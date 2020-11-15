import tensorflow as tf
import numpy as np

from Dataloader import *
from Discriminator import *
from Generator import *
from Oracle import *
from Discriminator_DataLoader import *


############# Changes from Original Paper #######################
'''
1) I think my Keras Sequence DataLoader should be able to replace rollout.py if i add the get_reward function here
2) I'm not using the Oracle to generate samples because that's dumb, I'm just going to use real examples


'''

##################################################################
########################## Variables #############################

# Directories
data_dir = ''
oracle_checkpoint_dir = ''
generator_checkpoint_dir = ''
discriminator_checkpoint_dir = ''

# Session Settings
load_oracle_weights = False
load_generator_weights = False
load_discriminator_weights = False

train_oracle = True
save_oracle_training_weights = True
oracle_training_epochs = 3
oracle_training_callbacks = []
oracle_training_verbose = 1

pre_train_generator = True
save_generator_training_weights = True
generator_pre_training_verbose = 1
generator_pre_training_epochs = 3
generator_pre_training_callbacks = []
generator_pre_training_verbose = 1

pre_train_discriminator = True
save_discriminator_training_weights = True
discriminator_pre_training_epochs = 3
discriminator_pre_training_callbacks = []
discriminator_pre_training_verbose = 1

# Hyperparameters
batch_size = 128
sequence_length = 128
validation_split=0.1
seed=None
use_word_vectors=False

# Oracle
oracle_hidden_units = 256
oracle_leaky_relu_alpha = 0.1
oracle_layers = 1
oracle_optimizer = tf.keras.optimizers.Adam(lr=0.01)
oracle_dropout_keep_prob = 1.0
oracle_l2_regularization_lambda = 0.0
oracle_loss = 'categorical_crossentropy'
oracle_metrics = ['loss', 'val_loss']

# Generator
generator_hidden_units = 256
generator_leaky_relu_alpha = 0.1
generator_layers = 1
generator_optimizer = tf.keras.optimizers.Adam(lr=0.01)
generator_dropout_keep_prob = 1.0
generator_l2_regularization_lambda = 0.0
generator_loss = 'categorical_crossentropy'
generator_metrics = ['loss', 'val_loss']

# Discriminator
filter_sizes_by_layer = []
number_of_filters_by_layer = []
discriminator_leaky_relu_alpha = 0.1
discriminator_dropout_keep_prob = 1.0
discriminator_l2_regularization_lambda = 0.0
discriminator_learning_rate = 1e-4
carry_bias = 0
discriminator_metrics = ['loss', 'val_loss']
discriminator_pre_training_fake_batch_size = batch_size

##################################################################


# Initialize DataLoaders
train_dl = DataLoader(
    data_filename = data_dir,
    batch_size = batch_size,
    sequence_length = sequence_length,
    validation_split = validation_split,
    seed = seed,
    data_type = "train",
    use_word_vectors=use_word_vectors
)
val_dl = DataLoader(
    data_filename = data_dir,
    batch_size = batch_size,
    sequence_length = sequence_length,
    validation_split = validation_split,
    seed = seed,
    data_type = "val",
    use_word_vectors=use_word_vectors
)
disc_dl = Discriminator_DataLoader(
    data_filename = data_dir,
    batch_size = batch_size,
    sequence_length = sequence_length,
    validation_split = validation_split,
    fake_batch_size = discriminator_pre_training_fake_batch_size,
    seed = seed,
    data_type = "val",
    use_word_vectors=use_word_vectors
)

# Initialize Models
oracle = Oracle(
    train_data_loader = train_dl,
    validation_data_loader = val_dl,
    units = oracle_hidden_units,
    leaky_relu_alpha = oracle_leaky_relu_alpha,
    num_layers = oracle_layers,
    opt = oracle_optimizer,
    dropout_keep_prob = oracle_dropout_keep_prob,
    l2_reg_lambda = oracle_l2_regularization_lambda,
    sequence_length = sequence_length,
    loss = oracle_loss,
    metrics = oracle_metrics
)
gen = Generator(
    train_data_loader = train_dl,
    validation_data_loader = val_dl,
    units = generator_hidden_units,
    leaky_relu_alpha=generator_leaky_relu_alpha,
    num_layers = generator_layers,
    opt = generator_optimizer,
    dropout_keep_prob = generator_dropout_keep_prob,
    l2_reg_lambda = generator_l2_regularization_lambda,
    sequence_length = sequence_length,
    loss = generator_loss,
    metrics = generator_metrics
)
disc = Discriminator(
    table_len = len(train_dl.st.table),
    filter_sizes = filter_sizes_by_layer,
    num_filters = number_of_filters_by_layer,
    sequence_len = sequence_length,
    l2_reg_lambda = discriminator_l2_regularization_lambda,
    dropout_keep_prob = discriminator_dropout_keep_prob,
    learning_rate = discriminator_learning_rate,
    leaky_relu_alpha = discriminator_leaky_relu_alpha,
    carry_bias=carry_bias,
    metrics = discriminator_metrics
)


# Load Model Weights if you have them
if load_oracle_weights:
    oracle.load_weights(oracle_checkpoint_dir)
if load_generator_weights:
    gen.load_weights(generator_checkpoint_dir)
if load_discriminator_weights:
    disc.load_weights(discriminator_checkpoint_dir)

# Train Oracle
if train_oracle:
    oracle_history = oracle.train(
        epochs = oracle_training_epochs,
        verbose = oracle_training_verbose,
        callbacks = oracle_training_callbacks,
        save_weights = save_oracle_training_weights,
        filepath = oracle_checkpoint_dir
    )

# Pretrain Generator
if pre_train_generator:
    generator_history = gen.pretrain(
        epochs = generator_pre_training_epochs,
        verbose = generator_pre_training_verbose,
        callbacks = generator_pre_training_callbacks,
        save_weights = save_generator_training_weights,
        filepath = generator_checkpoint_dir
    )

# Pretrain Discriminator
if pre_train_discriminator:
    for i in range(discriminator_pre_training_epochs):
        print("AYYYYYY starting epoch "+str(i+1) + '/' + str(discriminator_pre_training_epochs))
        print("Training History: " + str(history))
        epoch_finished = False
        batch_num = 1
        while not epoch_finished:
            epoch_finished, x, y = disc_dl.get_batch(gen)
            history = disc.model.train_on_batch(
                x = x,
                y = y,
                reset_metrics = False,
                return_dict = True
            )
            if batch_num % 20 == 0:
                print("Finished batch " + str(batch_num))
            batch_num += 1

