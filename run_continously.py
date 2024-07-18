import ctypes

from CheckersTraining import CheckersTraining

# Define constants
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

# Prevent sleep
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

try:
    game = CheckersTraining()
    game.run_simulation()

    # game.load_status()
    # new_dict = game.clean_dict_keys(game.monte_carlo_scoring)
    # game.monte_carlo_scoring = new_dict
    # game.save_status()

finally:
    # Restore the original state when done
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)