import cupy as cp

# Define a class for the neural network that can use multiple GPUs
class NeuralNetworkMultiGPU:
    def __init__(self, layers, gpus=[0, 1]):
        """
        Initialize the neural network with specified layers and GPUs.
        
        Parameters:
        layers (list): Number of neurons in each layer.
        gpus (list): List of GPU device IDs to use for training.
        """
        self.layers = layers
        self.gpus = gpus
        self.weights = []
        self.biases = []
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights and biases for each layer with random values.
        """
        for i in range(len(self.layers) - 1):
            # Randomly initialize weights and biases for each layer
            weight = cp.random.randn(self.layers[i], self.layers[i + 1])
            bias = cp.random.randn(self.layers[i + 1])
            self.weights.append(weight)
            self.biases.append(bias)

    def sigmoid(self, x):
        """
        Apply the sigmoid activation function.
        
        Parameters:
        x (array): Input array.

        Returns:
        array: Output after applying the sigmoid function.
        """
        return 1 / (1 + cp.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Compute the derivative of the sigmoid function.
        
        Parameters:
        x (array): Input array.

        Returns:
        array: Derivative of the sigmoid function.
        """
        return x * (1 - x)

    def forward(self, X):
        """
        Perform a forward pass through the network.
        
        Parameters:
        X (array): Input data.

        Returns:
        array: Output of the network.
        """
        self.a = [X]
        for i in range(len(self.weights)):
            # Compute the weighted sum of inputs and add bias
            z = cp.dot(self.a[-1], self.weights[i]) + self.biases[i]
            # Apply the sigmoid activation function
            a = self.sigmoid(z)
            self.a.append(a)
        return self.a[-1]

    def backward(self, X, y, learning_rate):
        """
        Perform a backward pass through the network and update weights and biases.
        
        Parameters:
        X (array): Input data.
        y (array): Target output data.
        learning_rate (float): Learning rate for weight updates.
        """
        m = y.shape[0]  # Number of training examples
        delta = self.a[-1] - y  # Error at the output layer
        for i in reversed(range(len(self.weights))):
            # Compute the gradient of the loss with respect to weights and biases
            dw = cp.dot(self.a[i].T, delta) / m
            db = cp.sum(delta, axis=0) / m
            # Compute the error for the next layer (going backward)
            delta = cp.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.a[i])
            # Update weights and biases
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

    def train(self, X, y, epochs, learning_rate):
        """
        Train the neural network using the specified number of epochs and learning rate.
        
        Parameters:
        X (array): Training input data.
        y (array): Training target data.
        epochs (int): Number of training iterations.
        learning_rate (float): Learning rate for weight updates.
        """
        # Split data across the available GPUs
        X_splits = cp.array_split(X, len(self.gpus))
        y_splits = cp.array_split(y, len(self.gpus))

        for epoch in range(epochs):
            # Parallel processing on multiple GPUs
            for gpu_id, (X_part, y_part) in enumerate(zip(X_splits, y_splits)):
                with cp.cuda.Device(self.gpus[gpu_id]):
                    # Perform forward and backward pass on each GPU
                    output = self.forward(X_part)
                    self.backward(X_part, y_part, learning_rate)
            if epoch % 100 == 0:
                # Calculate and print the loss every 100 epochs
                loss = cp.mean(cp.square(y - self.forward(X)))
                print(f'Epoch {epoch}, Loss: {loss}')

