# Neural Network from Scratch - Backpropagation & Gradient Descent

This project implements backpropagation and gradient descent algorithms from scratch in Python, with visualizations and detailed analysis.

## Contents

1. **backpropagation.py** - Complete backpropagation implementation with classification
2. **gradient_descent.py** - Gradient descent optimizer with learning rate analysis
3. **requirements.txt** - Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Backpropagation Implementation

```bash
python backpropagation.py
```

**What it does:**
- Implements a neural network from scratch with forward and backward propagation
- Uses a synthetic classification dataset (1000 samples, 20 features)
- Architecture: [20, 16, 8, 1] with ReLU activation
- Trains using backpropagation algorithm
- Visualizes loss convergence during training
- Reports test accuracy

**Key Features:**
- Custom implementation of forward propagation
- Backpropagation with chain rule for gradient computation
- Support for both sigmoid and ReLU activations
- Binary cross-entropy loss
- He weight initialization
- Training/test split with standardization

### 2. Gradient Descent Simulation

```bash
python gradient_descent.py
```

**What it does:**
- Implements gradient descent optimizer from scratch
- Tests on multiple mathematical functions (quadratic, Rosenbrock)
- Demonstrates learning rate effects with 4 different rates
- Provides 1D and 2D optimization visualizations
- Comprehensive learning rate analysis with 20 different values
- Shows convergence paths and loss curves

**Key Features:**
- 1D optimization on quadratic function (f(x) = x²)
- 2D optimization on Rosenbrock function (non-convex)
- Contour plots showing optimization paths
- Learning rate sweep analysis
- Convergence speed vs learning rate comparison

## Output

The scripts generate the following visualizations:

1. **backpropagation_loss_convergence.png**
   - Loss curve during neural network training
   - Shows convergence behavior

2. **gradient_descent_1d_learning_rates.png**
   - 1D gradient descent with different learning rates
   - Demonstrates speed and stability trade-offs

3. **gradient_descent_2d_learning_rates.png**
   - 2D optimization paths on Rosenbrock function
   - Shows how learning rate affects convergence trajectory

4. **learning_rate_analysis.png**
   - Comprehensive analysis of learning rate effects
   - Convergence speed, final loss, and success rate

## Key Concepts Demonstrated

### Backpropagation Algorithm
1. Forward pass: compute activations layer by layer
2. Compute loss using binary cross-entropy
3. Backward pass: compute gradients using chain rule
4. Update weights using gradient descent

### Gradient Descent
- **Learning Rate Impact:**
  - Too small: slow convergence, many iterations needed
  - Optimal: balanced speed and stability
  - Too large: oscillation, overshooting, divergence

### Mathematical Functions Used
- **Quadratic (1D):** Simple convex function for basic demonstration
- **Rosenbrock (2D):** Classic non-convex test function with challenging landscape

## Interpretation

### Learning Rate Effects:
- **< 0.001**: Very slow but stable
- **0.001 - 0.01**: Optimal range for most problems
- **> 0.1**: Risk of instability and divergence

### Practical Insights:
- Monitor loss curves to detect instability
- Use learning rate decay for better convergence
- Consider adaptive optimizers (Adam, RMSprop) for complex problems
- The optimal learning rate is problem-dependent

## Implementation Details

### Backpropagation Steps:
1. Initialize weights randomly (He initialization)
2. Forward propagation through network
3. Calculate loss
4. Compute output layer error
5. Backpropagate error to hidden layers
6. Calculate gradients for all weights and biases
7. Update parameters using gradient descent

### Gradient Descent Algorithm:
```
while not converged:
    gradient = compute_gradient(current_position)
    current_position = current_position - learning_rate * gradient
```

## Architecture

**Neural Network Structure:**
- Input layer: 20 features
- Hidden layer 1: 16 neurons (ReLU)
- Hidden layer 2: 8 neurons (ReLU)
- Output layer: 1 neuron (Sigmoid)

## Requirements
- Python 3.7+
- NumPy
- Matplotlib
- scikit-learn
