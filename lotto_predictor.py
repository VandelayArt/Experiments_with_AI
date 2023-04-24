# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks

# Load the dataset
df = pd.read_excel('Lotto_sequence.xlsx')
data = df.values.astype(np.int32)

# %%
# Prepare the input and output data
X = data[:, :-1]
Y = data[:, -1]

# Normalize the input data
X_norm = X / 45.0

# Reshape the input data
X_reshaped = X_norm.reshape((X_norm.shape[0], X_norm.shape[1], 1))

# Convert the output data to one-hot encoding
Y_onehot = tf.keras.utils.to_categorical(Y, num_classes=46)

# %%

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


# %%
# Save the model
model.save('new_lotto_model.h5')


# %%
# Load the trained model
model = tf.keras.models.load_model("new_lotto_model.h5")


# %%
# Generate a single winning sequence
seed_sequence = X[np.random.choice(len(X))]
generated_sequence = list(seed_sequence)

# %%
generated_sequence = []

for j in range(6):
    seed_sequence = X[np.random.choice(len(X))]
    next_number = np.argmax(model.predict(seed_sequence.reshape(1, 6, 1))[0])
    while next_number + 1 in generated_sequence:
        next_number = np.argmax(model.predict(seed_sequence.reshape(1, 6, 1))[0])
    generated_sequence.append(next_number + 1)
    seed_sequence = np.append(seed_sequence[1:], [next_number])

print('Generated sequence:', generated_sequence)



# %%
# Generate the bonus number
bonus_number = np.random.randint(1, 46)
while bonus_number in generated_sequence:
    bonus_number = np.random.randint(1, 46)
generated_sequence.append(bonus_number)


# %%
# Print the generated sequence
print(generated_sequence)
