from preprocessing import *

#sacuvati snimke nakon sto smo ih jednom izmešali
def save_dataset_as_npz(dataset, filename):
    # Lists to store inputs and labels
    inputs, labels = [], []

    # Iterate through the dataset and extract inputs and labels
    for input_data, label_data in dataset:
        inputs.append(input_data.numpy())
        labels.append(label_data.numpy())

    # Convert lists to arrays
    inputs = np.array(inputs)
    labels = np.array(labels)

    # Save the arrays to an npz file
    np.savez_compressed(filename, inputs=inputs, labels=labels)
    print(f"Dataset saved to {filename}.npz")



def datapipeline(global_batch_size = 512, num_sub_batches = 8):

    local_batch_size = global_batch_size // num_sub_batches

    speaker_list = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
                    's11', 's12', 's13', 's15', 's16', 's17', 's18', 's19',
                    's20', 's22']


    train_all_speakers = None

    #on each 100 videos, speakers is changing
    test_all_speakers = None

    for i in range(len(speaker_list)):

        data = tf.data.Dataset.list_files(f'data/{speaker_list[i]}/*.mpg')
        data = data.map(lambda file: mappable_function(speaker_list[i], file))
        data = data.shuffle(1000, reshuffle_each_iteration= False)

        train = data.take(900)
        test = data.skip(900)

        #print('TRAIN', len(train))
        #print('DATA', len(data))
        #print('TEST', len(test))

        if train_all_speakers is None:
            train_all_speakers = train
        else:
            train_all_speakers = train_all_speakers.concatenate(train)

        if test_all_speakers is None:
            test_all_speakers = test
        else:
            test_all_speakers = test_all_speakers.concatenate(test)
            print(len(test_all_speakers))

    #print('Train: ', len(train_all_speakers))
    #print('Test: ', len(test_all_speakers))

    #shuffle all speakers so we train neural network on random video order from random speaker order
    train_all_speakers = train_all_speakers.shuffle(3000, reshuffle_each_iteration= False).padded_batch(local_batch_size, padded_shapes=([75, 64, 64, 1], [40]))
    train_all_speakers = train_all_speakers.prefetch(tf.data.AUTOTUNE)

    test_all_speakers = test_all_speakers.padded_batch(local_batch_size, padded_shapes=([75, 64, 64, 1], [40]))
    test_all_speakers = test_all_speakers.prefetch(tf.data.AUTOTUNE)


    save_dataset_as_npz(train_all_speakers, 'train_all_speakers')
    save_dataset_as_npz(test_all_speakers, 'test_all_speakers')

    print("SAVED!")
    return train_all_speakers, test_all_speakers

#poravnanje alignmentsa i video snimaka
def mappable_function(file, numspeaker):

    result = tf.py_function(load_data, [file, numspeaker], (tf.float32, tf.int64))

    return result
