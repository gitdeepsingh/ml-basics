#%%
# imports and Generate synthetic dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import models, layers, optimizers, callbacks

# 
x = np.linspace(-3, 3, 2000)
y = np.sin(x) + 0.1 * np.random.randn(*x.shape)
plt.plot(y)
data = np.vstack((x, y)).T

# Preprocessing: normalization + train/test split
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print(data_scaled.shape, data_scaled[:5])

train_data, test_data = train_test_split(data_scaled, test_size=0.2, random_state=42)
# %%
# Build Autoencoder Model
def build_autoencoder(input_dim):
    encoder = models.Sequential([
        layers.Input(shape=input_dim),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(2, activation='linear', name='latent_space')
    ], name="Encoder")

    decoder = models.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ], name="Decoder")

    autoencoder = models.Sequential([encoder, decoder], name="Autoencoder")
    return autoencoder, encoder, decoder

autoencoder, encoder, decoder = build_autoencoder(input_dim=2)

# Compile the model
autoencoder.compile(optimizer=optimizers.Adam(learning_rate=0.01),
                    loss='mse',
                    metrics=['mae'])
#%%
# Train the model
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

training_results = autoencoder.fit(
    train_data, train_data,
    epochs=400,
    batch_size=16,
    shuffle=True,
    validation_data=(test_data, test_data),
    callbacks=[early_stop],
    verbose=1
)
print(training_results.history.keys())
#%%
# performance evaluation
train_loss = autoencoder.evaluate(train_data, train_data, verbose=0)
test_loss = autoencoder.evaluate(test_data, test_data, verbose=0)
print('train_loss=', train_loss)
print('test_loss=', test_loss)
#%%
# visualization
reconstructed = autoencoder.predict(test_data)
test_data_orig = scaler.inverse_transform(test_data)
reconstructed_orig = scaler.inverse_transform(reconstructed)
plt.figure(figsize=(7,5))
plt.scatter(test_data_orig[:, 0], test_data_orig[:, 1], label='Original (Test Data)', color='blue')
plt.scatter(reconstructed_orig[:, 0], reconstructed_orig[:, 1], label='Reconstructed', color='red', alpha=0.7)
plt.title("Original vs Reconstructed (Test Data)")
plt.legend()
plt.show()

# ==============================
# 9️⃣ Visualize Latent Space
# ==============================
latent_repr = encoder.predict(test_data)
plt.figure(figsize=(5,4))
plt.scatter(latent_repr[:, 0], latent_repr[:, 1], c='green')
plt.title("Learned Latent Space (2D)")
plt.xlabel("Latent Dim 1")
plt.ylabel("Latent Dim 2")
plt.show()

# %%
#testing
# Encode and decode test data
encoded_data = encoder.predict(test_data)
decoded_data = decoder.predict(encoded_data)

# Compare visually
import matplotlib.pyplot as plt

n = 5  # show 5 random samples
plt.figure(figsize=(10, 6))
for i in range(n):
    # Original signal
    plt.subplot(2, n, i + 1)
    plt.plot(test_data[i])
    plt.title("Original")
    plt.axis('off')
    
    # Reconstructed signal
    plt.subplot(2, n, i + 1 + n)
    plt.plot(decoded_data[i])
    plt.title("Reconstructed")
    plt.axis('off')

plt.tight_layout()
plt.show()


# %%
