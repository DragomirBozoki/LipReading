import os
import gdown
import cv2
import dlib
import tensorflow as tf
from tensorflow.keras.backend import ctc_batch_cost
import numpy as np
from typing import List
from matplotlib import pyplot as plt
from collections import defaultdict
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from preprocessing import vocabulary
from model import LipReadingModel
from preprocessing import *
from datapipeline import *

char_to_num, num_to_char = vocabulary()

class SpeakerControl(tf.keras.callbacks.Callback):
    def __init__(self, numspeaker):
        super().__init__()
        self.numspeaker = numspeaker

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 50 == 0:
            self.numspeaker += 1
            print(f'Epoch {epoch+1}: Switching to speaker {self.numspeaker}')
        if self.numspeaker > 10:
            print('Training done')
            self.model.stop_training = True

# Use the custom callback in model.f

def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return float(lr*tf.math.exp(-0.1))

class ProduceExample(tf.keras.callbacks.Callback):

    def __init__(self, dataset) -> None:
        self.dataset = dataset.as_numpy_iterator()

    def on_epoch_end(self, epoch, logs=None) -> None:
        data = self.dataset.next()
        yhat = self.model.predict(data[0])
        decoded = tf.keras.backend.ctc_decode(yhat, [75,75], greedy=False)[0][0].numpy()
        for x in range(len(yhat)):
            print('Original is: ', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))
            print('Prediction is: ', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
            print('~'*100)


def CTCLoss(y_true, y_pred):

    batch_len = tf.cast(tf.shape(y_true)[0], dtype = "int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype = "int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype = "int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype = "int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype = "int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def learning():

    model = LipReadingModel()

    model.compile(optimizer = Adam(learning_rate = 0.0001), loss = CTCLoss)

    speaker_control_callback = SpeakerControl(numspeaker=1)

    checkpoint_callback = ModelCheckpoint(os.path.join('models', 'checkpoint_full.weights.h5'), monitor = 'loss', save_weights_only = True)

    schedule_callback = LearningRateScheduler(scheduler)

    train_data = data(speaker_control_callback.numspeaker)

    example_callback = ProduceExample(train_data)

    model.build((None, 75, 64, 64, 1))
    history = model.fit(train_data, epochs = 50*9, callbacks = [speaker_control_callback, checkpoint_callback, schedule_callback, example_callback])

    return history

def __main__():
    learning()
__main__()
