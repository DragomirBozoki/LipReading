import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, Dense, GRU, Dropout, Bidirectional, MaxPool3D
from tensorflow.keras.layers import Flatten, Activation, TimeDistributed
from tensorflow.keras.initializers import Orthogonal


class LipReadingModel(tf.keras.Model):
    def __init__(self):
        super(LipReadingModel, self).__init__()

        self.conv1 = Conv3D(128, 3, padding='same', input_shape=(75, 64, 64))
        self.act1 = Activation('relu')
        self.pool1 = MaxPool3D((1, 2, 2))

        self.conv2 = Conv3D(256, 3, padding='same')
        self.act2 = Activation('relu')
        self.pool2 = MaxPool3D((1, 2, 2))

        self.conv3 = Conv3D(64, 3, padding='same')
        self.act3 = Activation('relu')
        self.pool3 = MaxPool3D((1, 2, 2))

        self.flatten = TimeDistributed(Flatten())

        self.gru1 = Bidirectional(GRU(128, kernel_initializer=Orthogonal(), return_sequences=True))
        self.dropout1 = Dropout(0.5)

        self.gru2 = Bidirectional(GRU(128, kernel_initializer=Orthogonal(), return_sequences=True))
        self.dropout2 = Dropout(0.5)

        self.dense = Dense(41, kernel_initializer='he_normal',
                           activation='softmax')  # 41 klasa izlaza, prilagodi po potrebi

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)

        x = self.flatten(x)

        x = self.gru1(x)
        x = self.dropout1(x)

        x = self.gru2(x)
        x = self.dropout2(x)

        return self.dense(x)