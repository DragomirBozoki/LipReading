from preprocessing import *


def datapipeline():

    speaker_list = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
                    's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19',
                    's20', 's21']


    for i in range(len(speaker_list)):

        data = tf.data.Dataset.list_files(f'data/s{speaker_list[i]}/*.mpg')
        data = data.map(lambda file: mappable_function(speaker_list[i], file))
        data = data.shuffle(1000, reshuffle_each_iteration=False)

    data = data.padded_batch(10, padded_shapes=([75, 64, 64, 1], [40]))
    data = data.prefetch(tf.data.AUTOTUNE)

    train = data.take(20800)
    test = data.skip(20800)

    return train, test

#poravnanje alignmentsa i video snimaka
def mappable_function(file, numspeaker):

    result = tf.py_function(load_data, [file, numspeaker], (tf.float32, tf.int64))

    return result
