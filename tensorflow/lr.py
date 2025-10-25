#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# synthetic data
X = np.linspace(0,4,400)
y = 2 * X + 3 + np.random.randn(*X.shape)*1.0

# model params: weights and bias
w = tf.Variable(np.random.randn(), dtype=tf.float32)
b = tf.Variable(np.random.randn(), dtype=tf.float32)

learning_rate = 0.01
epochs = 200

losses = []
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = w * X + b
        loss = tf.reduce_mean(tf.square(y_pred-y))
    
    gradient = tape.gradient(loss, [w,b])

    # update weights
    w.assign_sub(learning_rate * gradient[0])
    b.assign_sub(learning_rate * gradient[1])

    losses.append(loss.numpy())

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss={loss.numpy():.2f}, w={w.numpy():.2f}, b={b.numpy():.2f}")

#plots
plt.scatter(X, y, label="True Data", color="blue")
plt.plot(X, w.numpy()*X+b.numpy(), label="Regression Line", color="red")
plt.legend()
plt.show()

plt.plot(losses)
plt.title("Loss Curve")
plt.show()

# %%
