import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks

# Load the dataset
df = pd.read_excel('Lotto_sequence.xlsx')
data = df.values.astype(np.int32)

# Prepare the input and output data
X = data[:, :-1]
Y = data[:, -1]

# Normalize the input data
X_norm = X / 45.0

# Reshape the input data
X_reshaped = X_norm.reshape((X_norm.shape[0], X_norm.shape[1], 1))

# Convert the output data to one-hot encoding
Y_onehot = tf.keras.utils.to_categorical(Y, num_classes=46)

# Define the model
model = tf.keras.Sequential([
    layers.LSTM(128, input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])),
    layers.Dense(46, activation='softmax'),
    layers.Dense(46, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model
model.fit(X_reshaped, Y_onehot, epochs=100, batch_size=32)

# Save the model
model.save('new_lotto_model.h5')
