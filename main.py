from lipreadingnn import *
import json

def __main__():
    history = learning()

    #sacuvati history u json file
    with open('training_history.json', 'w') as file:
        json.dump(history.history, file)

__main__()