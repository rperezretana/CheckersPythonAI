import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class CheckersNN:
    def __init__(self):
        self.model = self.build_model()
    
    def build_model(self):
        # Define the model
        model = models.Sequential()
        model.add(layers.Input(shape=(65,)))  # 1 for player number + 64 for the board state
        model.add(layers.Dense(128, activation='linear'))
        model.add(layers.Dense(64, activation='linear'))
        model.add(layers.Dense(1, activation='linear'))  # Linear output layer
        
        # Compile the model
        model.compile(optimizer='adam',
                      loss='mean_squared_error')
        return model
    
    def train(self, X, y, epochs=10, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
    
    def predict(self, X):
        return self.model.predict(X)

# Example usage
# if __name__ == "__main__":
#     # Initialize the neural network
#     checkers_nn = CheckersNN()
    
#     # Dummy data for demonstration purposes
#     # X will be the player number + flattened board state
#     X = np.random.rand(1000, 65)  # 1000 samples of 65 features
#     y = np.random.rand(1000) * 100  # 1000 samples of labels scaled to 0-100
    
#     # Train the model
#     checkers_nn.train(X, y, epochs=10)
    
#     # Make a prediction
#     sample_input = np.random.rand(1, 65)  # Single sample of 65 features
#     prediction = checkers_nn.predict(sample_input)
#     print(f"Predicted score: {prediction[0][0]}")
