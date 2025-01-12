#%%
import numpy as np
# LinearSVM
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param  # Regularization strength
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Train the SVM on input data X and labels y.
        """
        n_samples, n_features = X.shape
        
        # Convert labels to -1 and 1
        y = np.where(y <= 0, -1, 1)

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # Gradient for correctly classified points
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # Gradient for misclassified points
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.learning_rate * y[idx]

    def predict(self, X):
        """
        Predict class labels for input data X.
        """
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)

# Generate a simple dataset for testing
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Create 2 classes with 2 features
X, y = make_blobs(n_samples=100, centers=2, random_state=6)
y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1

# Train the SVM
svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X, y)

# Predict labels
y_pred = svm.predict(X)

# Plot the decision boundary
def plot_svm(X, y, model):
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x - b + offset) / w[1]

    fig, ax = plt.subplots()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')

    # Plot decision boundary
    x0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
    x1 = get_hyperplane_value(x0, model.w, model.b, 0)
    x1_margin_positive = get_hyperplane_value(x0, model.w, model.b, 1)
    x1_margin_negative = get_hyperplane_value(x0, model.w, model.b, -1)

    ax.plot(x0, x1, 'k--', label="Decision Boundary")
    ax.plot(x0, x1_margin_positive, 'r--', label="Margin (+1)")
    ax.plot(x0, x1_margin_negative, 'b--', label="Margin (-1)")

    plt.legend()
    plt.show()

plot_svm(X, y, svm)


#%%
# Kernel SVM
import numpy as np

class KernelSVM:
    def __init__(self, kernel='rbf', C=1.0, gamma=0.5, learning_rate=0.001, epochs=1000):
        self.kernel = kernel
        self.C = C  # Regularization parameter
        self.gamma = gamma  # For RBF kernel
        self.learning_rate = learning_rate  # Gradient descent learning rate
        self.epochs = epochs  # Number of epochs
        self.alpha = None  # Lagrange multipliers
        self.W = None  # Weight vector
        self.b = None  # Bias term
        self.X = None  # Training data
        self.y = None  # Training labels

    def _kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(x1, x2) + 1) ** 3
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError("Unsupported kernel type")

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.X = X
        self.y = np.where(y <= 0, -1, 1)  # Convert labels to -1 and 1
        self.W = np.zeros(n_features)  # Initialize weights
        self.b = 0  # Initialize bias

        # Kernel matrix computation (for faster computation in the dual space)
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])

        # Gradient descent optimization for SVM parameters
        for epoch in range(self.epochs):
            for i in range(n_samples):
                condition = self.y[i] * (np.dot(self.W, X[i]) + self.b) >= 1
                if condition:
                    # If correctly classified, only regularization term matters
                    grad_W = self.W
                    grad_b = 0
                else:
                    # If misclassified, include the hinge loss gradient
                    grad_W = self.W - self.C * self.y[i] * X[i]
                    grad_b = -self.C * self.y[i]
                
                # Update the parameters
                self.W -= self.learning_rate * grad_W
                self.b -= self.learning_rate * grad_b

    def predict(self, X):
        predictions = []
        for x in X:
            prediction = np.dot(self.W, x) + self.b
            predictions.append(np.sign(prediction))
        return np.array(predictions)

# Generate a dataset
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

# Create a non-linearly separable dataset
X, y = make_circles(n_samples=200, factor=0.5, noise=0.1)
y = np.where(y == 0, -1, 1)

# Train the Kernel SVM
model = KernelSVM(kernel='rbf', C=1.0, gamma=0.5, learning_rate=0.01, epochs=1000)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot the decision boundary
def plot_decision_boundary(X, y, model):
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title("Kernel SVM Decision Boundary")
    plt.show()

plot_decision_boundary(X, y, model)