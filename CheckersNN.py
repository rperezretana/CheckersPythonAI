import os
import tensorflow as tf

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {0, 1, 2, 3} based on the verbosity level

class CheckersNN:
    def __init__(self):
        self.model = self.build_model()
    
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(33,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def train(self, x, y):
        # x = x.reshape(-1, 65)
        self.model.fit(x, y, epochs=10, verbose=0)

    def predict(self, x):
        return self.model.predict(x)
    
    def save_model(self, path):
        self.model.save(path)

    def load(self, path):
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path, compile=False)
            self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
            print(f"Model loaded from {path}")
        else:
            self.model = self.build_model()  # Create a new model
            print(f"No model found at {path}. Created a new model.")

