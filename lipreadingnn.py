from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from preprocessing import *
from model import LipReadingModel
from datapipeline import *

strategy = tf.distribute.MirroredStrategy()
char_to_num, num_to_char = vocabulary()

config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 4
config.inter_op_parallelism_threads = 2

session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


#podesiva brzina ucenja
def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    else:
        cycle = (epoch - 50) // 50
        if cycle % 2 == 0:
            return float(lr * tf.math.exp(-0.1))
        else:
            return lr

#sacuvaj history u backup na svakih 5 epoha
class SaveHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path, interval=10):
        super(SaveHistoryCallback, self).__init__()
        self.save_path = save_path
        self.interval = interval
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key, value in logs.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
        if (epoch + 1) % 5 == 0:
            self.save_history()

    def save_history(self):
        # Save the history as a file
        save_file = os.path.join(self.save_path, f'history_epoch_{len(self.history["loss"])}.txt')
        with open(save_file, 'w') as f:
            for key, values in self.history.items():
                f.write(f'{key}: {values}\n')
        print(f'Saved history at epoch {len(self.history["loss"])}')

#predikcija mreze nakon svake epohe
class ProduceExample(tf.keras.callbacks.Callback):

    def __init__(self, dataset) -> None:
        self.dataset = dataset.as_numpy_iterator()

    def on_epoch_end(self, epoch, logs=None) -> None:

        data = self.dataset.next()
        yhat = self.model.predict(data[0])
        batch_size = yhat.shape[0]

        sequence_lengths = [75] * batch_size
        decoded = tf.keras.backend.ctc_decode(yhat, sequence_lengths, greedy=False)[0][0].numpy()

        for x in range(len(yhat)):
            print('Original is: ', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))
            print('Prediction is: ', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
            print('~' * 100)

#CTCloss funkcija gubitka za merenje kvaliteta mreže
def CTCLoss(y_true, y_pred):

    batch_len = tf.cast(tf.shape(y_true)[0], dtype = "int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype = "int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype = "int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype = "int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype = "int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

#podesi učenje
def learning():

    with strategy.scope():
        model = LipReadingModel()
        model.compile(optimizer = Adam(learning_rate = 0.0001), loss = CTCLoss)
        model.build((None, 75, 64, 64, 1))

    checkpoint_callback = ModelCheckpoint(os.path.join('Weights', 'checkpoint_full.weights.h5'), monitor = 'loss', save_weights_only = True)

    schedule_callback = LearningRateScheduler(scheduler)

    train_data, test_data = datapipeline()

    example_callback = ProduceExample(train_data)

    save_history_callback = SaveHistoryCallback(save_path='history_logs', interval=10)

    history = model.fit(train_data, epochs = 30*22, callbacks = [checkpoint_callback, schedule_callback, example_callback, save_history_callback])

    return history
