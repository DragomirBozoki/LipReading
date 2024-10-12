from lipreadingnn import *
import json

def __main__():

    #pocni ucenje
    history = learning()

    #sacuvaj history u json fajl, za analizu toka obuke
    with open('training_history.json', 'w') as file:
        json.dump(history.history, file)

__main__()