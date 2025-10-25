#%%
import torch
import matplotlib.pyplot as plt

seed = torch.manual_seed(42)

# synthetic data: y = 2x + 3 + noise
X = torch.linspace(0, 10, 400).unsqueeze(1)
y = 2 * X + 3 + torch.randn(X.size()) * 0.5

plt.figure(figsize=(7, 5))
plt.scatter(X.numpy(), y.numpy(), color='blue', label='Data points')
plt.title("Synthetic Linear Data")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# initialize weights and biases
w = torch.randn(1, requires_grad=True, dtype=torch.float32)
b = torch.randn(1, requires_grad=True, dtype=torch.float32)

# training
learning_rate = 0.01
epochs = 200
losses=[]

for epoch in range(epochs):
    y_pred = w*X + b

    loss = torch.mean((y_pred-y)**2)
    losses.append(loss.item())
    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    w.grad.zero_()
    b.grad.zero_()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, w = {w.item():.3f}, b = {b.item():.3f}")

print(f"\nLearned Parameters → w = {w.item():.3f}, b = {b.item():.3f}")

plt.figure(figsize=(8, 6))
plt.scatter(X.detach(), y.detach(), label="Data (True)", color="blue", alpha=0.4)
plt.scatter(X.detach(), (w*X+b).detach(), label="Regression Line", color="red")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(losses, label='MSE Loss')
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()





# %%
