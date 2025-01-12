#%%
import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """Store the training data."""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predict the labels for the given data."""
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        """Predict the label of a single data point."""
        # Calculate distances from x to all training samples
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Get the indices of the k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get the labels of the k-nearest neighbors
        k_neighbor_labels = [self.y_train[i] for i in k_indices]
        # Majority vote for classification
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_blobs(n_samples=300, centers=2, random_state=42, cluster_std=2.5)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Initialize the KNN model with k=5
knn = KNN(k=5)
knn.fit(X_train, y_train)

# Predict on test data
predictions = knn.predict(X_test)


# Calculate accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")


# %%
