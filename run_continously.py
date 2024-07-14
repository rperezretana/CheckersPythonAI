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
finally:
    # Restore the original state when done
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)