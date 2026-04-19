# %%
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %%
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# %%
data = pd.read_csv("train/train.csv")
dtest = pd.read_csv("test/test.csv")

# %% [markdown]
# ### Missing Values

# %%
vars_with_na = data.columns[data.isnull().any()].tolist()
vars_with_na_test = dtest.columns[dtest.isnull().any()].tolist()
print(f"Missing values in train: {len(vars_with_na)}")
print(f"Missing values in test: {len(vars_with_na_test)}")

# %% [markdown]
# ### Group the Data [train and test]

# %%
train_test_data = [data, dtest]

# %%
# list of numerical variables
for dataset in train_test_data:
    num_vars = [var for var in dataset.columns if dataset[var].dtypes != 'O']
    print('Number of numerical variables: ', len(num_vars))

# %% [markdown]
# ### Check for value greater than 1

# %%
num_vars = [var for var in data.columns if data[var].dtypes != 'O']
Xdatanum = data[num_vars].drop(["ID", "y"], axis=1)

# Ensure all columns are numeric
for var in Xdatanum.columns:
    Xdatanum[var] = pd.to_numeric(Xdatanum[var], errors='coerce')
    if not Xdatanum[var].isna().all():
        if Xdatanum[var].max() > 1:
            print(f"Column {var} has values greater than 1")

# %%
# list of categorical variables
cat_vars = [var for var in data.columns if data[var].dtypes == 'O']
print('Number of categorical variables: ', len(cat_vars))

# %% [markdown]
# ### Looking at individual plot

# %%
if "X0" in data.columns and "y" in data.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data["X0"], y="y", data=data)
    plt.title("X0 vs y")
    plt.show()

if "X4" in data.columns and "y" in data.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data["X4"], y="y", data=data)
    plt.title("X4 vs y")
    plt.show()

# %% [markdown]
# ### Suspicious data

# %%
suspiciousData = []
for col in data:
    if len(data[col].unique()) == 1:
        suspiciousData.append(col)
        
if suspiciousData:
    print(f"Dropping {len(suspiciousData)} suspicious columns: {suspiciousData}")
    data[suspiciousData].describe()

# %% [markdown]
# ### Drop suspicious features

# %%
for dataset in train_test_data:
    dataset.drop(suspiciousData, axis=1, inplace=True)

# %% [markdown]
# ### Type of data

# %%
dtype_df = data.dtypes.reset_index()
dtype_df.columns = ["Column", "Column Type"]
dtype_df["Column Type"] = dtype_df["Column Type"].astype(str)
print(dtype_df.groupby("Column Type").size().reset_index(name='Count'))

# %% [markdown]
# ## Target analysis

# %%
plt.figure(figsize=(10, 6))
sns.histplot(data["y"], bins=50, kde=True)
plt.title("Target Variable Distribution")
plt.xlabel("y")
plt.ylabel("Frequency")
plt.show()

# %%
print("Target variable statistics:")
print(data["y"].describe())

# %% [markdown]
# ## Categorical data processing

# %%
if len(cat_vars) > 0:
    print(f"Processing {len(cat_vars)} categorical variables")
    
    # Use frequency encoding instead of count encoder to avoid dependencies
    for var in cat_vars:
        # Create frequency encoding
        freq_encoding = data[var].value_counts().to_dict()
        data[f"{var}_freq"] = data[var].map(freq_encoding)
        dtest[f"{var}_freq"] = dtest[var].map(freq_encoding).fillna(0)
    
    # Drop original categorical columns
    data = data.drop(cat_vars, axis=1)
    dtest = dtest.drop(cat_vars, axis=1)
    
    print("Categorical variables encoded successfully")
else:
    print("No categorical variables to process")

# %% [markdown]
# # Feature Scaling

# %%
# Get frequency columns
freq_cols = [col for col in data.columns if col.endswith('_freq')]

if len(freq_cols) > 0:
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(data[freq_cols])
    data[freq_cols] = scaler.transform(data[freq_cols])
    dtest[freq_cols] = scaler.transform(dtest[freq_cols])
    print(f"Scaled {len(freq_cols)} frequency columns")

# %% [markdown]
# ## Prepare data for modeling

# %%
# Drop ID column
data = data.drop("ID", axis=1)

# Define X and y
X = data.drop("y", axis=1)
y = data["y"]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# %%
# Convert all columns to numeric
X = X.apply(pd.to_numeric, errors='coerce')
# Fill NaN values with column means
X = X.fillna(X.mean())
# Fill any remaining NaN with 0
X = X.fillna(0)

# %%
# Convert to numpy arrays
X = X.values
y = y.values

print(f"X array shape: {X.shape}")
print(f"y array shape: {y.shape}")

# %% [markdown]
# ## Train test split

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# %% [markdown]
# # Build and train the CNN model

# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# %%
# For 1D CNN (better for tabular data)
batch_size = 32
epochs = 50

# Reshape for 1D CNN: (samples, features, 1)
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f'X_train shape for CNN: {X_train_cnn.shape}')
print(f'X_test shape for CNN: {X_test_cnn.shape}')

# %%
# Define R2 metric using TensorFlow
def r2_metric(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

# %%
# Build 1D CNN model (better for tabular data)
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1), padding='same'),
    MaxPooling1D(pool_size=2),
    
    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    
    Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    
    Flatten(),
    
    Dense(256, activation='relu'),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    Dropout(0.2),
    
    Dense(64, activation='relu'),
    
    Dense(1, activation='linear')
])

# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mae', r2_metric]
)

model.summary()

# %%
# Callbacks for better training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

# %%
# Train the model
history = model.fit(
    X_train_cnn, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test_cnn, y_test),
    callbacks=callbacks
)

# %%
# Evaluate the model
score = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f'Test Loss (MSE): {score[0]:.4f}')
print(f'Test MAE: {score[1]:.4f}')
print(f'Test R2: {score[2]:.4f}')

# %% [markdown]
# # Visualize training history

# %%
# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot loss
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_title('Model Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()
axes[0].grid(True)

# Plot R2
axes[1].plot(history.history['r2_metric'], label='Train R2')
axes[1].plot(history.history['val_r2_metric'], label='Validation R2')
axes[1].set_title('Model R2 Score')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('R2 Score')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# # Visualise the predictions and residuals

# %%
# Make predictions
preds = model.predict(X_test_cnn)
preds = preds.flatten()

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values')
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Plot residuals
residuals = y_test - preds

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram of residuals
axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Residuals')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Residuals Distribution')
axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[0].grid(True, alpha=0.3)

# Residuals vs predictions
axes[1].scatter(preds, residuals, alpha=0.5)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predictions')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals vs Predictions')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Calculate and print metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
rmse = np.sqrt(mse)

print("="*50)
print("MODEL PERFORMANCE METRICS")
print("="*50)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
print("="*50)

# %% [markdown]
# # Make predictions on test set

# %%
# Prepare test data
test_data = dtest.drop("ID", axis=1).copy()

# Convert to numeric
test_data = test_data.apply(pd.to_numeric, errors='coerce')
test_data = test_data.fillna(test_data.mean())
test_data = test_data.fillna(0)

# Reshape for CNN
X_test_final = test_data.values
X_test_final = X_test_final.reshape(X_test_final.shape[0], X_test_final.shape[1], 1)

print(f"Test data shape: {X_test_final.shape}")

# %%
# Make predictions
predictions = model.predict(X_test_final)
predictions = predictions.flatten()

print(f"Predictions shape: {predictions.shape}")
print(f"Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
print(f"Predictions mean: {predictions.mean():.4f}")
print(f"Predictions std: {predictions.std():.4f}")

# %%
# Create submission file
submission = pd.DataFrame({
    "ID": dtest["ID"],
    "y": predictions
})

# Save submission
submission.to_csv('submission_5.csv', index=False)
print("\n" + "="*50)
print("SUBMISSION FILE CREATED")
print("="*50)
print(f"Submission shape: {submission.shape}")
print(f"Submission saved to: submission_5.csv")
print("\nFirst 5 rows of submission:")
print(submission.head())

# %%
# Optional: Save the model for later use
model.save('cnn_regression_model.h5')
print("\nModel saved as: cnn_regression_model.h5")
