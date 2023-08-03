# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np


def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"
    sentences = bbc["text"].values.tolist()
    labels = bbc["category"].values.tolist()

    # training_size = int(len(sentences) * training_portion)

    training_sentences, validation_sentences = train_test_split(sentences, train_size=training_portion, shuffle=False)
    training_labels, validation_labels = train_test_split(labels, train_size=training_portion, shuffle=False)

    # Fit your tokenizer with training data
    tokenizer_sentences = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer_sentences.fit_on_texts(training_sentences)  # YOUR CODE HERE

    train_sequences = tokenizer_sentences.texts_to_sequences(training_sentences)
    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    valid_sequences = tokenizer_sentences.texts_to_sequences(validation_sentences)
    valid_padded = pad_sequences(valid_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # You can also use Tokenizer to encode your label.
    tokenizer_label = Tokenizer()
    tokenizer_label.fit_on_texts(labels)

    train_labels_sequences = tokenizer_label.texts_to_sequences(training_labels)
    valid_labels_sequences = tokenizer_label.texts_to_sequences(validation_labels)

    train_labels_sequences = np.array(train_labels_sequences)
    valid_labels_sequences = np.array(valid_labels_sequences)

    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(256, activation='relu'),
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    model.fit(train_padded,
              train_labels_sequences,
              epochs=20,
              validation_data=(valid_padded,
                               valid_labels_sequences))

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
