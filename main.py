from lipreadingnn import *
from data_download_googledrive import check_download
import json

def __main__():

    #pocni ucenje
    history = learning()

    #sacuvaj history u json fajl, za analizu toka obuke
    with open('training_history.json', 'w') as file:
        json.dump(history.history, file)

__main__()