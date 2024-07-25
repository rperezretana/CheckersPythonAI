from Enumerators import Engines
import numpy as np


DEBUG_ON = False # set to false if not interested on the ouputs
TRAINING = False # set False if not interested on training the nn
RANDOM_FIRST_PLAYS = 2 # This activates the random play for a few plays at the start of the game
PLAYER_1_ENGINE = Engines.MC
PLAYER_2_ENGINE = Engines.RANDOM # -1
EXECUTE_SAVE_ASYNC = False # enables multi threading to save files, sueful most of the time.
                          # To stop the proccess without damaging the file, create a file called "stop.txt", 
                          # this will ask the app to stop after the file has been saved, stopping the app safe might take a few minutes
SAVES_INTERVAL =  100000  # this number hast to be big (>10000) if the async is enabled and debug is False.
STOP_CHECK_INTERVAL = 5000 # this always has to be lower than SAVES_INTERVAL

    
def debug_print(*args, end=None):
    if DEBUG_ON:
        if end:
            print(*args, end=end)
        else:
            print(*args, end=end)
