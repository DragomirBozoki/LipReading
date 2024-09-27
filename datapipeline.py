from preprocessing import *

def data(numspeaker):

    data = tf.data.Dataset.list_files(f'./data/s{numspeaker}/*.mpg')
    data = data.shuffle(1000, reshuffle_each_iteration=False)
    data = data.map(lambda file: mappable_function(numspeaker, file))

    data = data.padded_batch(10, padded_shapes=([75,64,64, 1],[40]))
    data = data.prefetch(tf.data.AUTOTUNE)

    return data

def mappable_function(file, numspeaker):
    result = tf.py_function(load_data, [file, numspeaker], (tf.float32, tf.int64))

    return result