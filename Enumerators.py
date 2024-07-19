from enum import Enum


class Engines(Enum):
    NN = 1  # Neural network
    MC = 2  # Monte Carlo
    RANDOM = 3 # Random

    
class Player(Enum):
    PLAYER_1 = 1
    PLAYER_2 = -1