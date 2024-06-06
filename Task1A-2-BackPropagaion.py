import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Activation function: Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Loss function: Mean Squared Error and its derivative
def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(1, hidden_size)
        self.bias_output = np.random.rand(1, output_size)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, output):
        error = mse_derivative(y, output)
        d_output = error * sigmoid_derivative(output)

        error_hidden_layer = d_output.dot(self.weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output -= self.hidden_output.T.dot(d_output)
        self.weights_input_hidden -= X.T.dot(d_hidden_layer)

        self.bias_output -= np.sum(d_output, axis=0)
        self.bias_hidden -= np.sum(d_hidden_layer, axis=0)

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
        return self

    def predict(self, X):
        return self.forward(X)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One hot encoding the target
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train neural network
nn = NeuralNetwork(input_size=4, hidden_size=5, output_size=3)
nn.train(X_train, y_train, epochs=5000, learning_rate=0.1)

# Testing the neural network
predictions = nn.predict(X_test)
predictions = np.argmax(predictions, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = np.mean(predictions == y_test_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")
