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
print (len(vars_with_na), len(vars_with_na_test))

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

# FIX: Ensure all columns are numeric
for var in Xdatanum.columns:
    # Convert to numeric if possible
    Xdatanum[var] = pd.to_numeric(Xdatanum[var], errors='coerce')
    
    # Skip if column became all NaN
    if Xdatanum[var].isna().all():
        continue
        
    if Xdatanum[var].max() > 1:
        print(f"Column {var} has values greater than 1")

# %%
# list of categorical variables
cat_vars = [var for var in data.columns if data[var].dtypes == 'O']
print('Number of categorical variables: ', len(cat_vars))

# Check if there are any categorical variables before plotting
if len(cat_vars) > 0:
    for c in data[cat_vars]:
        value_counts = data[c].value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.title('Categorical feature {} - Cardinality {}'.format(c, len(np.unique(data[c]))))
        plt.xlabel('Feature value')
        plt.ylabel('Occurences')
        plt.bar(range(len(value_counts)), value_counts.values)
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation='vertical')
        plt.show()
else:
    print("No categorical variables found")

# %% [markdown]
# ### Looking at individual plot

# %%
# Make sure X0 and X4 exist in the data
if "X0" in data.columns:
    sns.boxplot(x=data["X0"], y="y", data=data)
if "X4" in data.columns:
    sns.scatterplot(x=data["X4"], y="y", data=data)

# %% [markdown]
# ### Suspicious data

# %%
suspiciousData = []
for col in data:
    if len(data[col].unique()) == 1:
        suspiciousData.append(col)
if suspiciousData:
    data[suspiciousData].describe()

# %% [markdown]
# ### Drop suspicious features

# %%
for dataset in train_test_data:
    dataset.drop(suspiciousData, axis=1, inplace=True)

# %% [markdown]
# ### Type of data
# FIX: Convert dtypes to string before grouping to avoid comparison issues

# %%
dtype_df = data.dtypes.reset_index()
dtype_df.columns = ["Column", "Column Type"]
# Convert dtype objects to strings to avoid comparison issues
dtype_df["Column Type"] = dtype_df["Column Type"].astype(str)
print(dtype_df.groupby("Column Type").size().reset_index(name='Count'))

# %% [markdown]
# ## Target analysis

# %%
sns.distplot(data["y"])
plt.show()

# %%
print(data["y"].describe())

# %% [markdown]
# ## Categorical data

# %%
if len(cat_vars) > 0:
    categoricalData = data[cat_vars]
    print(categoricalData.info())
    
    for var in cat_vars:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=var, y="y", data=data)
        plt.show()

# %% [markdown]
# # Count encoder

# %%
if len(cat_vars) > 0:
    import category_encoders as ce
    
    # Create the encoder itself
    Count_enc = ce.CountEncoder(cols=cat_vars)
    # Fit the encoder using the categorical features 
    Count_enc.fit(data[cat_vars], data["y"])
    
    data = data.join(Count_enc.transform(data[cat_vars]).add_suffix('_count'))
    dtest = dtest.join(Count_enc.transform(dtest[cat_vars]).add_suffix('_count'))
    
    # Drop original categorical columns
    data = data.drop(data[cat_vars], axis=1)
    dtest = dtest.drop(dtest[cat_vars], axis=1)

# %% [markdown]
# # Feature Scaling

# %%
# Get count columns if they exist
cat_count = [col for col in data.columns if col.endswith('_count')]

if len(cat_count) > 0:
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(data[cat_count])
    data[cat_count] = scaler.transform(data[cat_count])
    dtest[cat_count] = scaler.transform(dtest[cat_count])
    print(scaler.data_max_)

# %% [markdown]
# ## Define X and y

# %%
data = data.drop("ID", axis=1)

# %%
X = data.drop("y", axis=1)
y = data["y"]
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# %%
# Convert all columns to numeric, coercing errors to NaN
X = X.apply(pd.to_numeric, errors='coerce')
# Fill NaN values with 0
X = X.fillna(0)

# %%
X = X.values
y = y.values

# %% [markdown]
# ## Train test split

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# %% [markdown]
# # Build and train the CNN model

# %%
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import backend as K

# %%
batch_size = 15
epochs = 10

# Calculate the correct dimensions
n_features = X.shape[1]
# Find dimensions that work well
img_rows = int(np.sqrt(n_features))
img_cols = n_features // img_rows

# Adjust to ensure we don't lose features
while img_rows * img_cols < n_features:
    img_cols += 1

print(f"Original features: {n_features}")
print(f"Reshaping to: {img_rows} x {img_cols} = {img_rows * img_cols}")

# Pad features if necessary
n_pad = img_rows * img_cols - n_features
if n_pad > 0:
    print(f"Padding {n_pad} zeros to features")
    X_train = np.pad(X_train, ((0, 0), (0, n_pad)), mode='constant')
    X_test = np.pad(X_test, ((0, 0), (0, n_pad)), mode='constant')

# %%
# Reshape for CNN
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# %%
# Define R2 metric
def r2_keras(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

# %%
# Display sample image
plt.imshow(X_train[0].reshape(img_rows, img_cols))
plt.title(f"Sample input image ({img_rows}x{img_cols})")
plt.colorbar()
plt.show()

# %%
# Build the model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=[r2_keras])

model.summary()

# %%
# Train the model
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score}')

# %% [markdown]
# # Visualise the predictions and residuals

# %%
preds = model.predict(X_test)
preds = preds[:,0]
plt.scatter(y_test, preds)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values')
plt.show()

# %%
residuals = y_test - preds
sns.distplot(residuals)
plt.title('Residuals Distribution')
plt.show()

# %%
from sklearn.metrics import r2_score
r2 = r2_score(y_test, preds)
print(f'R2 Score: {r2}')

# %% [markdown]
# # Testing

# %%
# Prepare test data
test_data = dtest.drop("ID", axis=1).copy()

# Convert to numeric and handle NaN
test_data = test_data.apply(pd.to_numeric, errors='coerce')
test_data = test_data.fillna(0)

X_test_final = test_data.values
print(f"Test data shape: {X_test_final.shape}")

# Pad if necessary
if n_pad > 0:
    X_test_final = np.pad(X_test_final, ((0, 0), (0, n_pad)), mode='constant')

# %%
# Reshape test data
if K.image_data_format() == 'channels_first':
    X_test_final = X_test_final.reshape(X_test_final.shape[0], 1, img_rows, img_cols)
else:
    X_test_final = X_test_final.reshape(X_test_final.shape[0], img_rows, img_cols, 1)

X_test_final = X_test_final.astype('float32')

print('Test samples shape:', X_test_final.shape)

# %%
# Make predictions
prediction = model.predict(X_test_final)
prediction = prediction[:,0]

# %%
# Create submission file
submission = pd.DataFrame({
    "ID": dtest["ID"],
    "y": prediction
})

submission.to_csv('submission_5.csv', index=False)
print("Submission saved to submission_5.csv")
print(f"Submission shape: {submission.shape}")
