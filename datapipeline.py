from preprocessing import *


def datapipeline(global_batch_size = 80, num_sub_batches = 10):

    local_batch_size = global_batch_size // num_sub_batches

    speaker_list = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
                    's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19',
                    's20', 's21']

    #on each 900 videos, speaker is changing
    train_all_speakers = None

    #on each 100 videos, speakers is changing
    test_all_speakers = None

    for i in range(len(speaker_list)):

        data = tf.data.Dataset.list_files(f'data/s{speaker_list[i]}/*.mpg')
        data = data.map(lambda file: mappable_function(speaker_list[i], file))
        data = data.shuffle(1000, reshuffle_each_iteration=False)
        train = data.take(900)
        test = data.skip(900)

        if train_all_speakers is None:
            train_all_speakers = train
        else:
            train_all_speakers = train_all_speakers.concatenate(train)

        if test_all_speakers is None:
            test_all_speakers = test
        else:
            test_all_speakers = train_all_speakers.concatenate(train)

    train_all_speakers = train_all_speakers.padded_batch(local_batch_size, padded_shapes=([75, 64, 64, 1], [40]))
    train_all_speakers = train_all_speakers.prefetch(tf.data.AUTOTUNE)

    test_all_speakers = test_all_speakers.padded_batch(local_batch_size, padded_shapes=([75, 64, 64, 1], [40]))
    test_all_speakers = test_all_speakers.prefetch(tf.data.AUTOTUNE)

    return train_all_speakers, test_all_speakers

#poravnanje alignmentsa i video snimaka
def mappable_function(file, numspeaker):

    result = tf.py_function(load_data, [file, numspeaker], (tf.float32, tf.int64))

    return result
