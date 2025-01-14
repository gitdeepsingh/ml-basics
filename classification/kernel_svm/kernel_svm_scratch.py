#%%
import numpy as np

class KernelSVM:
    def __init__(self, kernel='linear', C=1.0, degree=3, gamma=1.0, coef0=1):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
    
    def fit(self, X, y, max_iter=1000, tol=1e-3):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.X = X
        self.y = y
        self.support_vectors = None
        
        K = self._compute_kernel_matrix(X)
        for _ in range(max_iter):
            alpha_prev = np.copy(self.alpha)
            for i in range(n_samples):
                error_i = self._decision_function(X[i]) - y[i]
                if (y[i] * error_i < -tol and self.alpha[i] < self.C) or \
                   (y[i] * error_i > tol and self.alpha[i] > 0):
                    j = self._choose_second_alpha(i, n_samples)
                    error_j = self._decision_function(X[j]) - y[j]
                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
                    L, H = self._compute_bounds(y[i], y[j], alpha_i_old, alpha_j_old)
                    if L == H:
                        continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    self.alpha[j] -= y[j] * (error_i - error_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    if abs(self.alpha[j] - alpha_j_old) < tol:
                        continue
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    b1 = self.b - error_i - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - error_j - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
            if np.allclose(self.alpha, alpha_prev, atol=tol):
                break
        self.support_vectors = X[self.alpha > 1e-4]
    
    def predict(self, X):
        return np.sign(self._decision_function(X))
    
    def _compute_kernel_matrix(self, X):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])
        return K
    
    def _kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'polynomial':
            return (self.gamma * np.dot(x1, x2) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError("Unsupported kernel")
    
    def _decision_function(self, x):
        decision = 0
        for i in range(len(self.alpha)):
            decision += self.alpha[i] * self.y[i] * self._kernel_function(self.X[i], x)
        return decision + self.b
    
    def _choose_second_alpha(self, i, n_samples):
        j = i
        while j == i:
            j = np.random.randint(0, n_samples)
        return j
    
    def _compute_bounds(self, yi, yj, alpha_i, alpha_j):
        if yi != yj:
            return (max(0, alpha_j - alpha_i), min(self.C, self.C + alpha_j - alpha_i))
        else:
            return (max(0, alpha_j + alpha_i - self.C), min(self.C, alpha_j + alpha_i))

# Generate some synthetic data
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=100, centers=2, random_state=6)
y = np.where(y == 0, -1, 1)

# Train SVM with RBF kernel
svm = KernelSVM(kernel='rbf', C=1.0, gamma=0.5)
svm.fit(X, y)

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))
Z = np.array([svm._decision_function(np.array([xx_, yy_])) for xx_, yy_ in zip(np.ravel(xx), np.ravel(yy))])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.8, colors=['blue', 'purple', 'red'])
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], edgecolors='k', marker='o', facecolors='none')
plt.title("Kernel SVM with RBF Kernel")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# %%
