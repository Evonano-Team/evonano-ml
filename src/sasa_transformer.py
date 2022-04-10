import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras import layers
import training_data_generator as tg
import parameter_specification as params

num_feature = params.num_feature
window_size = params.window_size

test_mbtr_dir = params.test_mbtr_dir 
train_mbtr_dir = params.train_mbtr_dir

test_sasa_dir = params.test_sasa_dir
train_sasa_dir = params.train_sasa_dir

x_train, _, y_train = tg.mbtr_mbtr_ds_generator(train_mbtr_dir, window_size = window_size, sasa_dir = train_sasa_dir, shuffle = False)
x_test, _, y_test = tg.mbtr_mbtr_ds_generator(test_mbtr_dir, window_size = window_size, sasa_dir = test_sasa_dir, shuffle = False)

def plot_sasa(x_test, y_test, NN_model, num_feature):
    plt.figure(figsize=(25, 10))
    plt.plot(range(y_test.shape[0]), y_test)
    plt.plot(range(y_test.shape[0]), [NN_model.predict(x_test[i].reshape(-1, window_size, num_feature))[0][0] for i in range(x_test.shape[0])])
    plt.xlabel("Training iterations")
    plt.ylabel("SASA Values")
    plt.legend(["Real Value", "Predicted Value"])
    plt.title('SASA Model Performance')
    plt.savefig('plots/sasa_predictions.png')


print("Here", x_train.shape)
print("Here", y_train.shape)


class transformer():
  def __init__(self, num_feature,
    head_size=256, num_heads=12, ff_dim=4, num_transformer_blocks=4, mlp_units=[128],
    window_size = 40, mlp_dropout=0.4, dropout=0.25):
    super().__init__()
    self.head_size = head_size
    self.num_heads = num_heads
    self.ff_dim = ff_dim
    self.num_transformer_blocks= num_transformer_blocks
    self.mlp_units = mlp_units
    self.num_feature = num_feature
    self.window_size= window_size
    self.mlp_dropout = mlp_dropout
    self.dropout = dropout
    self.model = self.build_model()

  def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout = 0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

  def build_model(self):
    inputs = keras.Input(shape=(self.window_size, self.num_feature))
    x = inputs
    for _ in range(self.num_transformer_blocks):
        x = self.transformer_encoder(x, self.head_size, self.num_heads, self.ff_dim, self.dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in self.mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(self.mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)

NN_model = transformer(num_feature = num_feature,
    head_size=256, num_heads=12, ff_dim=4, num_transformer_blocks=4, mlp_units=[128],
    window_size = window_size, mlp_dropout=0.4, dropout=0.25).model

NN_model.compile(
    loss="mse",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["mae"],
)

callbacks = [keras.callbacks.EarlyStopping(patience = 40, restore_best_weights=True)]
h = NN_model.fit(
    x_train,
    y_train,
    epochs = 100,
    validation_split = 0.2,
    batch_size = 32,
    callbacks = callbacks,
    verbose = 1
)
NN_model.save("/home/cloud-user/evonano-ml/models/sasa_transformer_model/model")
print("Test Performance:", NN_model.evaluate(x_test, y_test))

plt.plot(np.arange(len(h.history['loss'])), h.history['loss'], color = 'g', label = 'training loss')
plt.plot(np.arange(len(h.history['loss'])), h.history['val_loss'], color = 'r', label = 'validation loss')
plt.xlabel("Training iterations")
plt.ylabel("Values")
plt.legend()
plt.title('SASA Model Performance')
plt.savefig('plots/sasa_model_performance.png')

plot_sasa(x_test, y_test, NN_model, num_feature)