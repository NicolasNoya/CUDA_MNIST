#############################################
# Train a simple model on the MNIST dataset #
#############################################

#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

train_data, test_data = tf.keras.datasets.mnist.load_data(path="mnist.npz")
x_train = train_data[0].reshape(-1,784).astype(float) / 255
y_train = train_data[1]

# source : https://keras.io/guides/training_with_built_in_methods/

def build_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(32, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(32, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    return model

model = build_model()

loss = keras.losses.sparse_categorical_crossentropy

model.summary()

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#%%

# Train the model
model.fit(x_train, y_train, epochs=1, batch_size=600)
# %%
#########################################
# Create the weights file in C++ format #
#########################################

#%%
# print the weights
weights = model.get_weights()
print(f"Number of weights tensors: {len(weights[])}")

#%%
weight_dict = {}
for i, w in enumerate(weights):
    typ = "bias" if i % 2 else "dense"
    print(f"Layer {i} - Shape: {w.shape} - {typ}")
    weight_dict[f"layer_{typ}_{i}"] = w


#%%
for k, v in weight_dict.items():
    print(f"{k} - {v.shape}")


#%%
for i, w in enumerate(weights):
    with open("weights.cpp", "w") as cpp_file:
        cpp_file.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
        for layer_name in weight_dict.keys():
                weight_data = np.array(weight_dict[layer_name])
                flat_weights = weight_data.flatten()

                # Generate a C++ array
                cpp_file.write(f"// {layer_name}\n")
                cpp_file.write(f"const float {layer_name}[] = {{")
                cpp_file.write(", ".join(map(str, flat_weights)))
                cpp_file.write("};\n\n")

        cpp_file.write("#endif // WEIGHTS_H\n")
print("Done")