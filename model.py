import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress info/warning logs

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    plt.figure(figsize=(6, 5))
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    plt.scatter(test_data,  test_labels,  c="g", label="Testing data")
    plt.scatter(test_data,  predictions,  c="r", label="Predictions")
    plt.legend(shadow=True)
    plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
    plt.title('Model Results', family='Arial', fontsize=14)
    plt.xlabel('X axis values', family='Arial', fontsize=11)
    plt.ylabel('Y axis values', family='Arial', fontsize=11)
    plt.savefig('model_results.png', dpi=120)


def mae(y_test, y_pred):
    return tf.keras.losses.mae(y_test, y_pred)


def mse(y_test, y_pred):
    return tf.keras.losses.mse(y_test, y_pred)


print(tf.__version__)

X = np.arange(-100, 100, 4).reshape(-1, 1)
y = np.arange(-90,  110, 4).reshape(-1, 1)

X_train, y_train = X[:40], y[:40]
X_test,  y_test  = X[40:], y[40:]

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['mae']
)

model.fit(X_train, y_train, epochs=10, verbose=0)

y_preds = model.predict(X_test)

# plot_predictions( train_data=X_train, train_labels=y_train, test_data=X_test,   test_labels=y_test, predictions=y_preds)

mae_1 = np.round(float(mae(y_test.squeeze(), y_preds.squeeze())), 2)
mse_1 = np.round(float(mse(y_test.squeeze(), y_preds.squeeze())), 2)
print(f'\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')

with open('metrics.txt', 'w') as outfile:
    outfile.write(f'\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')
