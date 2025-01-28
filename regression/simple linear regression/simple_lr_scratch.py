#%%
import numpy as np
import matplotlib.pyplot as plt

# Generate some synthetic data
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Independent variable (features)
y = 3.5 * X + np.random.randn(100, 1) * 5 + 10  # Dependent variable (target)

# Step 1: Calculate mean of X and y
X_mean = np.mean(X)
y_mean = np.mean(y)

# Step 2: Calculate coefficients
numerator = np.sum((X - X_mean) * (y - y_mean))
denominator = np.sum((X - X_mean) ** 2)
m = numerator / denominator  # Slope
c = y_mean - m * X_mean      # Intercept

# Step 3: Predictions
y_pred = m * X + c

# Step 4: Visualization
plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X, y_pred, color="red", label="Regression line")
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.legend()
plt.title("Simple Linear Regression")
plt.show()

# Step 5: Calculate Mean Squared Error (MSE)
mse = np.mean((y - y_pred) ** 2)
print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")
print(f"Mean Squared Error (MSE): {mse}")

# %%
