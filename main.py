from lipreadingnn import *
import json

def __main__():

    config = tf.compat.v1.ConfigProto()
    config.intra_op_parallelism_threads = 4
    config.inter_op_parallelism_threads = 2

    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    history = learning()

    #sacuvati history u json file
    with open('training_history.json', 'w') as file:
        json.dump(history.history, file)

__main__()