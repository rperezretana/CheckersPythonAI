import cupy as cp

class NeuralNetworkMultiGPU:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = self._initialize_weights()
    
    def _initialize_weights(self):
        weights = []
        for i in range(len(self.layers) - 1):
            weight = cp.random.randn(self.layers[i], self.layers[i + 1])
            weights.append(weight)
        return weights
    
    def forward(self, x):
        x = cp.array(x)
        activations = [x]
        for weight in self.weights:
            x = cp.dot(x, weight)
            activations.append(x)
        return activations

    def backward(self, activations, y):
        y = cp.array(y)
        output_error = activations[-1] - y
        deltas = [output_error * self._sigmoid_derivative(activations[-1])]
        
        for i in range(len(activations) - 2, 0, -1):
            delta = deltas[-1].dot(self.weights[i].T) * self._sigmoid_derivative(activations[i])
            deltas.append(delta)
        
        deltas.reverse()
        
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * activations[i].T.dot(deltas[i])
    
    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, x, y):
        activations = self.forward(x)
        self.backward(activations, y)

    def predict(self, x):
        x = cp.array(x)
        return self.forward(x)[-1]
