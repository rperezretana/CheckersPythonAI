"""
This intends to create a structure that stores items and keep taps on the frequency they are used.
After some time, the least used can be removed based on conditions ( minimum usage) or the 50% least used.
"""

class Memo:
    def __init__(self):
        self.data_dict = {}
        self.freq_dict = {}

    def __setitem__(self, key, value):
        self.data_dict[key] = value
        self.freq_dict[key] = 0

    def __getitem__(self, key):
        if key in self.data_dict:
            self.freq_dict[key] += 1
            return self.data_dict[key]
        raise KeyError(f'Key {key} not found')
    
    def __contains__(self, key):
        return key in self.data_dict

    def __delitem__(self, key):
        if key in self.data_dict:
            del self.data_dict[key]
            del self.freq_dict[key]
        else:
            raise KeyError(f'Key {key} not found')

    def insert(self, key, value):
        self.__setitem__(key, value)

    def get(self, key):
        return self.__getitem__(key)

    def remove_least_used(self):
        # Convert freq_dict to a list of (key, frequency) pairs
        freq_list = list(self.freq_dict.items())
        
        # Sort freq_list by frequency
        freq_list.sort(key=lambda x: x[1])
        
        # Determine the number of elements to remove (90%)
        remove_precentage = 0.9
        num_to_remove = int(len(freq_list) * remove_precentage)
        print(f"Removing {num_to_remove} items from the cache, leaving {int(len(freq_list) * (1-remove_precentage))}.")
        
        # Remove the least used keys
        for i in range(num_to_remove):
            key_to_remove = freq_list[i][0]
            del self[key_to_remove]



    