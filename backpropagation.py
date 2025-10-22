import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='sigmoid', learning_rate=0.01):
        """
        Initialize neural network with given architecture.
        
        Parameters:
        - layer_sizes: list of layer sizes [input, hidden1, hidden2, ..., output]
        - activation: 'sigmoid' or 'relu'
        - learning_rate: learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.losses = []
        
        # Initialize weights and biases using He initialization
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward_propagation(self, X):
        """
        Forward pass through the network.
        Returns activations for each layer.
        """
        activations = [X]
        
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            
            # Use sigmoid for output layer, chosen activation for hidden layers
            if i == len(self.weights) - 1:
                a = sigmoid(z)
            else:
                if self.activation == 'relu':
                    a = relu(z)
                else:
                    a = sigmoid(z)
            
            activations.append(a)
        
        return activations
    
    def backward_propagation(self, X, y, activations):
        """
        Backward pass (backpropagation) to compute gradients.
        """
        m = X.shape[0]
        deltas = [None] * len(self.weights)
        
        # Output layer error
        output_error = activations[-1] - y
        deltas[-1] = output_error
        
        # Backpropagate error through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            error = np.dot(deltas[i+1], self.weights[i+1].T)
            
            if self.activation == 'relu':
                deltas[i] = error * relu_derivative(activations[i+1])
            else:
                deltas[i] = error * sigmoid_derivative(activations[i+1])
        
        # Compute gradients
        weight_gradients = []
        bias_gradients = []
        
        for i in range(len(self.weights)):
            dw = np.dot(activations[i].T, deltas[i]) / m
            db = np.sum(deltas[i], axis=0, keepdims=True) / m
            weight_gradients.append(dw)
            bias_gradients.append(db)
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        """
        Update weights and biases using gradient descent.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def compute_loss(self, y_true, y_pred):
        """
        Binary cross-entropy loss.
        """
        m = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the neural network using backpropagation.
        """
        for epoch in range(epochs):
            # Forward propagation
            activations = self.forward_propagation(X)
            y_pred = activations[-1]
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # Backward propagation
            weight_gradients, bias_gradients = self.backward_propagation(X, y, activations)
            
            # Update parameters
            self.update_parameters(weight_gradients, bias_gradients)
            
            if verbose and (epoch + 1) % 100 == 0:
                accuracy = self.evaluate(X, y)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
    
    def predict(self, X):
        """
        Make predictions on input data.
        """
        activations = self.forward_propagation(X)
        return (activations[-1] > 0.5).astype(int)
    
    def evaluate(self, X, y):
        """
        Evaluate accuracy on dataset.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

# Generate classification dataset
print("Generating classification dataset...")
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                           n_redundant=5, n_classes=2, random_state=42)
y = y.reshape(-1, 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Create and train neural network
print("\nTraining Neural Network with Backpropagation...")
print("Architecture: [20, 16, 8, 1]")
nn = NeuralNetwork(layer_sizes=[20, 16, 8, 1], activation='relu', learning_rate=0.1)
nn.train(X_train, y_train, epochs=1000, verbose=True)

# Evaluate on test set
test_accuracy = nn.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Visualize loss convergence
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(nn.losses, linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (Binary Cross-Entropy)', fontsize=12)
plt.title('Loss Convergence During Training', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(nn.losses[50:], linewidth=2, color='orange')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (Binary Cross-Entropy)', fontsize=12)
plt.title('Loss Convergence (After 50 epochs)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('backpropagation_loss_convergence.png', dpi=300, bbox_inches='tight')
print("\nLoss convergence plot saved as 'backpropagation_loss_convergence.png'")
plt.show()
