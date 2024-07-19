from Enumerators import Engines
import numpy as np


DEBUG_ON = False # set to false if not interested on the ouputs
TRAINING = True # set False if not interested on training the nn
RANDOM_FIRST_PLAYS = 2 # This activates the random play for a few plays at the start of the game
PLAYER_1_ENGINE = Engines.MC
PLAYER_2_ENGINE = Engines.RANDOM # -1

    
def debug_print(*args, end=None):
    if DEBUG_ON:
        if end:
            print(*args, end=end)
        else:
            print(*args, end=end)
