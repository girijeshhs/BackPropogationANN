import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define test functions for gradient descent
def quadratic_function(x):
    """Simple quadratic function: f(x) = x^2"""
    return x ** 2

def quadratic_gradient(x):
    """Gradient of quadratic function"""
    return 2 * x

def rosenbrock_function(x, y):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(x, y):
    """Gradient of Rosenbrock function"""
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

def beale_function(x, y):
    """Beale function - another common test function"""
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def beale_gradient(x, y):
    """Gradient of Beale function"""
    dx = 2*(1.5 - x + x*y)*(-1 + y) + 2*(2.25 - x + x*y**2)*(-1 + y**2) + 2*(2.625 - x + x*y**3)*(-1 + y**3)
    dy = 2*(1.5 - x + x*y)*x + 2*(2.25 - x + x*y**2)*(2*x*y) + 2*(2.625 - x + x*y**3)*(3*x*y**2)
    return np.array([dx, dy])

class GradientDescentOptimizer:
    """Gradient Descent Optimizer Implementation"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = []
        self.loss_history = []
    
    def optimize_1d(self, gradient_func, loss_func, x_init):
        """Optimize 1D function"""
        x = x_init
        self.history = [x]
        self.loss_history = [loss_func(x)]
        
        for i in range(self.max_iterations):
            grad = gradient_func(x)
            x_new = x - self.learning_rate * grad
            
            self.history.append(x_new)
            self.loss_history.append(loss_func(x_new))
            
            if abs(x_new - x) < self.tolerance:
                break
            
            x = x_new
        
        return x, self.loss_history
    
    def optimize_2d(self, gradient_func, loss_func, x_init, y_init):
        """Optimize 2D function"""
        position = np.array([x_init, y_init], dtype=float)
        self.history = [position.copy()]
        self.loss_history = [loss_func(*position)]
        
        for i in range(self.max_iterations):
            grad = gradient_func(*position)
            position_new = position - self.learning_rate * grad
            
            self.history.append(position_new.copy())
            self.loss_history.append(loss_func(*position_new))
            
            if np.linalg.norm(position_new - position) < self.tolerance:
                break
            
            position = position_new
        
        return position, self.loss_history

# 1. Demonstrate gradient descent on 1D quadratic function
print("=" * 60)
print("1D GRADIENT DESCENT SIMULATION")
print("=" * 60)

learning_rates_1d = [0.01, 0.1, 0.5, 0.9]
x_init = 10.0

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Effect of Learning Rate on 1D Gradient Descent (f(x) = x²)', 
             fontsize=16, fontweight='bold')

for idx, lr in enumerate(learning_rates_1d):
    optimizer = GradientDescentOptimizer(learning_rate=lr, max_iterations=100)
    x_final, losses = optimizer.optimize_1d(quadratic_gradient, quadratic_function, x_init)
    
    row, col = idx // 2, idx % 2
    ax = axes[row, col]
    
    # Plot loss convergence
    ax.plot(losses, linewidth=2, color=f'C{idx}')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title(f'Learning Rate = {lr}\nFinal x = {x_final:.6f}, Iterations = {len(losses)-1}', 
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    print(f"\nLearning Rate: {lr}")
    print(f"  Initial x: {x_init}")
    print(f"  Final x: {x_final:.6f}")
    print(f"  Final loss: {losses[-1]:.6e}")
    print(f"  Iterations: {len(losses)-1}")

plt.tight_layout()
plt.savefig('gradient_descent_1d_learning_rates.png', dpi=300, bbox_inches='tight')
print(f"\n1D gradient descent plot saved as 'gradient_descent_1d_learning_rates.png'")

# 2. Demonstrate gradient descent on 2D Rosenbrock function
print("\n" + "=" * 60)
print("2D GRADIENT DESCENT SIMULATION (Rosenbrock Function)")
print("=" * 60)

learning_rates_2d = [0.0001, 0.0005, 0.001, 0.005]
x_init, y_init = -1.0, 1.0

fig = plt.figure(figsize=(16, 12))
fig.suptitle('Effect of Learning Rate on 2D Gradient Descent (Rosenbrock Function)', 
             fontsize=16, fontweight='bold')

# Create contour plot data
x_range = np.linspace(-2, 2, 400)
y_range = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock_function(X, Y)

for idx, lr in enumerate(learning_rates_2d):
    optimizer = GradientDescentOptimizer(learning_rate=lr, max_iterations=1000)
    final_pos, losses = optimizer.optimize_2d(rosenbrock_gradient, rosenbrock_function, 
                                               x_init, y_init)
    
    # Plot convergence path on contour
    ax1 = plt.subplot(2, 4, idx + 1)
    contour = ax1.contour(X, Y, Z, levels=np.logspace(-1, 3.5, 20), cmap='viridis', alpha=0.6)
    
    # Plot optimization path
    history = np.array(optimizer.history)
    ax1.plot(history[:, 0], history[:, 1], 'r.-', linewidth=2, markersize=4, alpha=0.7)
    ax1.plot(history[0, 0], history[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(history[-1, 0], history[-1, 1], 'r*', markersize=15, label='End')
    ax1.plot(1, 1, 'b*', markersize=15, label='Global Min')
    
    ax1.set_xlabel('x', fontsize=10)
    ax1.set_ylabel('y', fontsize=10)
    ax1.set_title(f'LR = {lr}\nIterations: {len(losses)-1}', fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot loss convergence
    ax2 = plt.subplot(2, 4, idx + 5)
    ax2.plot(losses, linewidth=2, color=f'C{idx}')
    ax2.set_xlabel('Iteration', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.set_title(f'Loss Convergence\nFinal Loss: {losses[-1]:.4f}', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    print(f"\nLearning Rate: {lr}")
    print(f"  Initial position: ({x_init}, {y_init})")
    print(f"  Final position: ({final_pos[0]:.6f}, {final_pos[1]:.6f})")
    print(f"  Final loss: {losses[-1]:.6e}")
    print(f"  Iterations: {len(losses)-1}")
    print(f"  Distance from optimum: {np.linalg.norm(final_pos - np.array([1, 1])):.6f}")

plt.tight_layout()
plt.savefig('gradient_descent_2d_learning_rates.png', dpi=300, bbox_inches='tight')
print(f"\n2D gradient descent plot saved as 'gradient_descent_2d_learning_rates.png'")

# 3. Detailed analysis of learning rate effects
print("\n" + "=" * 60)
print("LEARNING RATE ANALYSIS AND INTERPRETATION")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Comprehensive Learning Rate Analysis', fontsize=16, fontweight='bold')

# Test multiple learning rates
test_lrs = np.logspace(-3, 0, 20)
iterations_to_converge = []
final_losses = []
converged = []

for lr in test_lrs:
    optimizer = GradientDescentOptimizer(learning_rate=lr, max_iterations=500, tolerance=1e-8)
    final_pos, losses = optimizer.optimize_2d(rosenbrock_gradient, rosenbrock_function, 
                                               x_init, y_init)
    iterations_to_converge.append(len(losses) - 1)
    final_losses.append(losses[-1])
    converged.append(losses[-1] < 1.0)  # Consider converged if loss < 1.0

# Plot 1: Iterations vs Learning Rate
axes[0].semilogx(test_lrs, iterations_to_converge, 'b.-', linewidth=2, markersize=6)
axes[0].set_xlabel('Learning Rate', fontsize=12)
axes[0].set_ylabel('Iterations to Converge', fontsize=12)
axes[0].set_title('Convergence Speed vs Learning Rate', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Plot 2: Final Loss vs Learning Rate
axes[1].loglog(test_lrs, final_losses, 'r.-', linewidth=2, markersize=6)
axes[1].set_xlabel('Learning Rate', fontsize=12)
axes[1].set_ylabel('Final Loss', fontsize=12)
axes[1].set_title('Final Loss vs Learning Rate', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Plot 3: Convergence Success Rate
colors = ['red' if not c else 'green' for c in converged]
axes[2].scatter(test_lrs, final_losses, c=colors, s=80, alpha=0.7)
axes[2].set_xscale('log')
axes[2].set_yscale('log')
axes[2].set_xlabel('Learning Rate', fontsize=12)
axes[2].set_ylabel('Final Loss', fontsize=12)
axes[2].set_title('Convergence Success (Green) vs Failure (Red)', fontsize=13, fontweight='bold')
axes[2].axhline(y=1.0, color='blue', linestyle='--', linewidth=2, label='Threshold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_rate_analysis.png', dpi=300, bbox_inches='tight')
print("\nLearning rate analysis plot saved as 'learning_rate_analysis.png'")

# Print interpretation
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("""
LEARNING RATE EFFECTS ON CONVERGENCE:

1. TOO SMALL LEARNING RATE (< 0.001):
   - Very slow convergence
   - Requires many iterations
   - Stable but inefficient
   - May get stuck in local minima

2. OPTIMAL LEARNING RATE (0.001 - 0.01):
   - Balanced convergence speed
   - Stable descent
   - Reaches good solution efficiently
   - Best trade-off between speed and stability

3. TOO LARGE LEARNING RATE (> 0.1):
   - Unstable optimization
   - May overshoot the minimum
   - Can oscillate or diverge
   - May never converge to optimal solution

4. KEY OBSERVATIONS:
   - The optimal learning rate depends on the problem
   - Non-convex problems (like Rosenbrock) are more sensitive
   - Adaptive learning rates (Adam, RMSprop) can help
   - Learning rate schedules can improve convergence

5. PRACTICAL RECOMMENDATIONS:
   - Start with lr = 0.001 or 0.01
   - Use learning rate decay during training
   - Monitor loss curves for signs of instability
   - Consider adaptive optimizers for complex problems
""")

plt.show()

print("\n" + "=" * 60)
print("SIMULATION COMPLETE")
print("=" * 60)
print("\nAll plots have been saved:")
print("  1. gradient_descent_1d_learning_rates.png")
print("  2. gradient_descent_2d_learning_rates.png")
print("  3. learning_rate_analysis.png")
