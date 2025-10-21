# %% [markdown]
# # ERA5 Temperature Analysis for Delhi and Ahmedabad (2014-2023)
# 
# This notebook performs data preprocessing, statistical analysis, and temperature prediction using machine learning models on an ERA5 temperature dataset. We extract data for Delhi and Ahmedabad, generate basic statistics, visualize trends, perform linear regression for annual trends, and finally predict the monthly average temperature for 2023 using both linear regression and random forest models.
# 
# **Cities & Approximate Coordinates:**
# - **Delhi:** Latitude ≈ 28.70°N, Longitude ≈ 77.10°E
# - **Ahmedabad:** Latitude ≈ 23.03°N, Longitude ≈ 72.58°E

# %% [markdown]
# ## (A) Data Preprocessing and Exploration
# 
# 1. Load the ERA5 dataset using xarray and convert it into a Pandas DataFrame.
# 2. Extract temperature data for Delhi and Ahmedabad.
# 3. Generate a table of yearly basic statistics (mean, median, min, max, std).
# 4. Plot a time-series graph of monthly mean, min and max temperature trends (2014-2023).

# %%
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For machine learning and regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# %% [markdown]
# ### 1. Load the ERA5 Dataset and Convert to Pandas DataFrame
# 
# We assume that the dataset has a time coordinate (with timestamps) and a temperature variable (e.g., "t" or "temperature").
# Adjust the variable name if needed.

# %%
# Load the dataset (update the file path if necessary)
file_path = "ERA5_T_2014_2023.nc"
ds = xr.open_dataset(file_path, engine="netcdf4")
print(ds)
# Inspect variable names (assume temperature is in variable 't')
print(ds.data_vars)

# %%
# Convert the entire dataset into a DataFrame (flatten spatial dimensions)
df_full = ds.to_dataframe().reset_index()
df_full.head()

# %% [markdown]
# ### 2. Extract Temperature Data for Delhi and Ahmedabad
# 
# We select the nearest grid point using `.sel()` with the method `nearest` for each city.

# %%
# Define approximate coordinates
delhi_lat, delhi_lon = 28.70, 77.10
ahmedabad_lat, ahmedabad_lon = 23.03, 72.58

# Extract temperature data at the nearest grid point for each city
# Adjust variable name if temperature is stored under a different name
delhi_data = ds['t'].sel(latitude=delhi_lat, longitude=delhi_lon, method="nearest")
ahmedabad_data = ds['t'].sel(latitude=ahmedabad_lat, longitude=ahmedabad_lon, method="nearest")

# Convert to DataFrame and reset index (time becomes a column)
df_delhi = delhi_data.to_dataframe().reset_index()
df_ahmedabad = ahmedabad_data.to_dataframe().reset_index()

# Ensure that the time coordinate is in datetime format
df_delhi['time'] = pd.to_datetime(df_delhi['time'])
df_ahmedabad['time'] = pd.to_datetime(df_ahmedabad['time'])

# %% [markdown]
# ### 3. Generate Yearly Basic Statistics for Temperature
# 
# For each city, we compute the yearly mean, median, min, max, and standard deviation.

# %%
# Set time as index and group by year
df_delhi.set_index('time', inplace=True)
df_ahmedabad.set_index('time', inplace=True)

# Group by year and compute statistics
stats_delhi = df_delhi['t'].groupby(df_delhi.index.year).agg(['mean', 'median', 'min', 'max', 'std'])
stats_ahmedabad = df_ahmedabad['t'].groupby(df_ahmedabad.index.year).agg(['mean', 'median', 'min', 'max', 'std'])

print("Delhi Temperature Statistics (Yearly)")
print(stats_delhi)
print("\nAhmedabad Temperature Statistics (Yearly)")
print(stats_ahmedabad)

# %% [markdown]
# ### 4. Plot Monthly Temperature Trends
# 
# We compute the monthly mean, min, and max temperatures and plot the time series (2014-2023).

# %%
# Reset index to have time as a column again for resampling
df_delhi = df_delhi.reset_index()
df_ahmedabad = df_ahmedabad.reset_index()

# Resample monthly (using mean, min, max) for Delhi
df_delhi['time'] = pd.to_datetime(df_delhi['time'])
delhi_monthly = df_delhi.set_index('time').resample('M')['t'].agg(['mean','min','max']).reset_index()

# Resample monthly for Ahmedabad
df_ahmedabad['time'] = pd.to_datetime(df_ahmedabad['time'])
ahmedabad_monthly = df_ahmedabad.set_index('time').resample('M')['t'].agg(['mean','min','max']).reset_index()

# Plotting monthly trends for Delhi
plt.figure(figsize=(14, 6))
plt.plot(delhi_monthly['time'], delhi_monthly['mean'], label='Mean Temperature')
plt.plot(delhi_monthly['time'], delhi_monthly['min'], label='Min Temperature')
plt.plot(delhi_monthly['time'], delhi_monthly['max'], label='Max Temperature')
plt.title("Delhi Monthly Temperature Trends (2014-2023)")
plt.xlabel("Time")
plt.ylabel("Temperature (K)")  # Adjust unit if needed
plt.legend()
plt.grid(True)
plt.show()

# Plotting monthly trends for Ahmedabad
plt.figure(figsize=(14, 6))
plt.plot(ahmedabad_monthly['time'], ahmedabad_monthly['mean'], label='Mean Temperature')
plt.plot(ahmedabad_monthly['time'], ahmedabad_monthly['min'], label='Min Temperature')
plt.plot(ahmedabad_monthly['time'], ahmedabad_monthly['max'], label='Max Temperature')
plt.title("Ahmedabad Monthly Temperature Trends (2014-2023)")
plt.xlabel("Time")
plt.ylabel("Temperature (K)")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# **Comment:**  
# From the monthly trend plots, you can observe seasonal fluctuations in temperature. Typically, temperatures peak in the summer months and drop during winter. Over the period 2014-2023, one might notice subtle shifts in the monthly averages which could hint at a warming or cooling trend—this will be further explored in the next section.

# %% [markdown]
# ## (B) Statistical Analysis
# 
# 1. Plot the annual mean temperature trend for both cities and assess if there is a warming or cooling pattern.
# 2. Fit a linear regression model to the annual mean temperature values and plot the trendline.
# 3. Interpret the slope of the regression line.

# %%
# Compute annual mean temperatures
annual_delhi = df_delhi.set_index('time')['t'].resample('A').mean()
annual_ahmedabad = df_ahmedabad.set_index('time')['t'].resample('A').mean()

# Convert index to year for plotting/regression
annual_delhi.index = annual_delhi.index.year
annual_ahmedabad.index = annual_ahmedabad.index.year

plt.figure(figsize=(10, 5))
plt.plot(annual_delhi.index, annual_delhi.values, marker='o', label='Delhi')
plt.plot(annual_ahmedabad.index, annual_ahmedabad.values, marker='o', label='Ahmedabad')
plt.title("Annual Mean Temperature Trend (2014-2023)")
plt.xlabel("Year")
plt.ylabel("Mean Temperature (K)")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Prepare data for linear regression (reshape X to 2D array)
years = np.array(annual_delhi.index).reshape(-1, 1)

# Delhi regression
lr_delhi = LinearRegression()
lr_delhi.fit(years, annual_delhi.values)
trend_delhi = lr_delhi.predict(years)

# Ahmedabad regression
lr_ahmedabad = LinearRegression()
lr_ahmedabad.fit(years, annual_ahmedabad.values)
trend_ahmedabad = lr_ahmedabad.predict(years)

# Plot with regression trendline
plt.figure(figsize=(10, 5))
plt.plot(annual_delhi.index, annual_delhi.values, 'o-', label='Delhi')
plt.plot(annual_delhi.index, trend_delhi, 'r--', label=f'Delhi Trend (slope={lr_delhi.coef_[0]:.4f})')
plt.plot(annual_ahmedabad.index, annual_ahmedabad.values, 'o-', label='Ahmedabad')
plt.plot(annual_ahmedabad.index, trend_ahmedabad, 'g--', label=f'Ahmedabad Trend (slope={lr_ahmedabad.coef_[0]:.4f})')
plt.title("Annual Mean Temperature with Linear Regression Trendline")
plt.xlabel("Year")
plt.ylabel("Mean Temperature (K)")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Interpret the slope:
print("Delhi Linear Regression Slope:", lr_delhi.coef_[0])
print("Ahmedabad Linear Regression Slope:", lr_ahmedabad.coef_[0])
# A positive slope indicates a warming trend while a negative slope indicates a cooling trend.

# %% [markdown]
# **Interpretation:**  
# The regression slope for each city quantifies the annual change in mean temperature (in Kelvin per year).  
# - If the slope is positive, it suggests a warming trend over the period 2014-2023.
# - If the slope is negative, it suggests a cooling trend.
# The magnitude of the slope indicates how rapidly the temperature is changing. In our analysis, you would compare the slopes for Delhi and Ahmedabad to assess which city is experiencing a more pronounced warming (or cooling) trend.

# %% [markdown]
# ## (C) Temperature Prediction Using Machine Learning Models
# 
# We will predict the monthly average temperature for 2023 using both a linear regression model and a random forest model.
# 
# Steps:
# 1. Extract monthly mean temperatures from 2014 to 2023.
# 2. Split the dataset into training (80%) and testing (20%) sets.
# 3. Train linear regression and random forest models, then compare their performance using Mean Absolute Error (MAE) and R² score.
# 4. Compare predictions with the actual 2023 data.

# %%
# Extract monthly mean temperatures for Delhi (similar steps can be repeated for Ahmedabad or both)
monthly_delhi = df_delhi.set_index('time')['t'].resample('M').mean().reset_index()

# Use the year and month to create a time index feature
monthly_delhi['year'] = monthly_delhi['time'].dt.year
monthly_delhi['month'] = monthly_delhi['time'].dt.month

# Create a numeric time feature for regression: e.g., number of months since start (2014-01)
monthly_delhi = monthly_delhi.sort_values('time')
monthly_delhi['time_index'] = np.arange(len(monthly_delhi))
monthly_delhi.head()

# %%
# Separate data for training (2014-2022) and testing (2023)
train = monthly_delhi[monthly_delhi['year'] < 2023]
test = monthly_delhi[monthly_delhi['year'] == 2023]

# Features (we use time_index as our predictor) and target (temperature)
X_train = train[['time_index']].values
y_train = train['t'].values
X_test = test[['time_index']].values
y_test = test['t'].values

# %% [markdown]
# ### Model 1: Linear Regression

# %%
# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions on test set
pred_lr = lr_model.predict(X_test)

# Compute performance metrics
mae_lr = mean_absolute_error(y_test, pred_lr)
r2_lr = r2_score(y_test, pred_lr)
print("Linear Regression - MAE:", mae_lr, "R2:", r2_lr)

# %% [markdown]
# ### Model 2: Random Forest Regression

# %%
# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions on test set
pred_rf = rf_model.predict(X_test)

# Compute performance metrics
mae_rf = mean_absolute_error(y_test, pred_rf)
r2_rf = r2_score(y_test, pred_rf)
print("Random Forest - MAE:", mae_rf, "R2:", r2_rf)

# %% [markdown]
# ### Comparison of Predictions vs Actual 2023 Data
# 
# Let's plot the actual 2023 monthly mean temperatures along with predictions from both models.

# %%
plt.figure(figsize=(12, 6))
plt.plot(test['time'], y_test, 'o-', label='Actual 2023')
plt.plot(test['time'], pred_lr, 's--', label='Linear Regression Prediction')
plt.plot(test['time'], pred_rf, 'd--', label='Random Forest Prediction')
plt.title("Monthly Mean Temperature Predictions for 2023 (Delhi)")
plt.xlabel("Time")
plt.ylabel("Temperature (K)")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# **Discussion:**
# - The MAE and R² values indicate the prediction accuracy of both models.  
# - Compare the performance metrics: a lower MAE and higher R² score suggest a better model.
# - In the plot, the predictions are overlaid on the actual 2023 data.  
# - Analyze whether the models capture the seasonal variation correctly and which model performs better.
# 
# You can repeat similar analysis for Ahmedabad or create a combined dataset if desired.
# 
# ---
# 
# **Conclusion:**
# 
# This notebook demonstrated how to load ERA5 temperature data, extract city-specific information for Delhi and Ahmedabad, and perform basic statistical analysis and trend visualization. Additionally, we applied machine learning models (linear regression and random forest) to predict monthly average temperature for 2023, comparing model performance using MAE and R² score. The analysis offers insights into long-term temperature trends and serves as a basis for further climate or urban heat studies.



#%%
'''
This Python code implements Stochastic Gradient Descent (SGD) from scratch for a 
simple linear regression problem. The goal is to find the best-fit line (\(y=m\cdot x+b\)) 
by minimizing the Mean Squared Error (MSE) using randomly selected data points to update 
the parameters \(m\) (slope) and \(b\) (intercept). 
'''
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate synthetic data for linear regression
def generate_data(num_samples):
    np.random.seed(42)  # for reproducibility
    X = 2 * np.random.rand(num_samples, 1)
    y = 4 + 3 * X + np.random.randn(num_samples, 1) # y = 4 + 3x + noise
    return X, y

# 2. Implement the Stochastic Gradient Descent algorithm
def sgd_from_scratch(X, y, learning_rate=0.01, epochs=50):
    num_samples, num_features = X.shape
    
    # Initialize weights (slope) and bias (intercept) randomly
    m = np.random.randn(num_features, 1)
    b = np.random.randn(1, 1)
    
    cost_history = []
    
    for epoch in range(epochs):
        # Shuffle the data before each epoch for true stochasticity
        indices = np.random.permutation(num_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(num_samples):
            # Pick a single random data point (mini-batch size of 1)
            x_i = X_shuffled[i:i+1]
            y_i = y_shuffled[i:i+1]
            
            # Predict the value
            y_pred = x_i.dot(m) + b
            
            # Calculate the loss (MSE) and its gradients for a single sample
            error = y_pred - y_i
            gradient_m = x_i.T.dot(error)
            gradient_b = np.sum(error)
            
            # Update the parameters (m and b)
            m -= learning_rate * gradient_m
            b -= learning_rate * gradient_b
        
        # Calculate and store the average cost for the current epoch (optional)
        y_final_pred = X.dot(m) + b
        cost = np.mean((y_final_pred - y)**2)
        cost_history.append(cost)
        
    return m, b, cost_history

# 3. Run the SGD algorithm and make predictions
num_samples = 100
X, y = generate_data(num_samples)
m_final, b_final, cost_history = sgd_from_scratch(X, y, learning_rate=0.01, epochs=50)

print(f"Final parameters: m={m_final[0][0]:.4f}, b={b_final[0][0]:.4f}")

# Make predictions on the training data with the final parameters
X_new = np.array([[0], [2]])
y_pred_final = X_new.dot(m_final) + b_final

# 4. Plot the results
plt.figure(figsize=(12, 5))

# Plot the training data and the fitted line
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='b', label='Original Data')
plt.plot(X_new, y_pred_final, 'r-', label=f'SGD Fit: y={m_final[0][0]:.2f}x + {b_final[0][0]:.2f}')
plt.title('SGD Linear Regression from Scratch')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Plot the cost over epochs
plt.subplot(1, 2, 2)
plt.plot(range(len(cost_history)), cost_history, 'g-')
plt.title('Cost Function over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)

plt.tight_layout()
plt.show()


# %%
'''
For a binary classification problem, you can implement sensitivity (recall), specificity,
and an ROC curve using TensorFlow/Keras along with a confusion matrix. 
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Generate synthetic binary classification data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Build and train a simple neural network model
model = Sequential()
model.add(Dense(32, input_dim=20, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # Sigmoid activation for binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# 3. Predict probabilities and classes
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype("int32")

# 4. Calculate Sensitivity (Recall) and Specificity
# A. Get the confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# B. Calculate metrics from the confusion matrix
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print("Confusion Matrix:")
print(cm)
print(f"\nSensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

# C. Use scikit-learn's classification report for a summary
print("\nClassification Report (using scikit-learn):")
print(classification_report(y_test, y_pred))

# 5. Plot the ROC Curve
# A. Calculate ROC curve metrics using probabilities
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)

# B. Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# %%
