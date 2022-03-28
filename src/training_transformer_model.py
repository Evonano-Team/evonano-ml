import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras import layers
import training_data_generator as tg


timestep_size = 300
num_feature = 72
window_size = 40

test_mbtr_dir = '/home/cloud-user/evonano-ml/data/processed/test_set/features'
train_mbtr_dir = '/home/cloud-user/evonano-ml/data/processed/training_set/features'

x_train, y_train = tg.mbtr_mbtr_ds_generator(train_mbtr_dir, window_size = window_size, timestep_size = timestep_size)
x_test, y_test = tg.mbtr_mbtr_ds_generator(test_mbtr_dir, window_size = window_size, timestep_size = timestep_size)

#test_sasa_dir = '/home/cloud-user/evonano-ml/data/processed/test_set/label'
#train_sasa_dir = '/home/cloud-user/evonano-ml/data/processed/training_set/label'

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
    outputs = layers.Dense(self.num_feature)(x)
    return keras.Model(inputs, outputs)

model = transformer(num_feature = num_feature,
    head_size=256, num_heads=12, ff_dim=4, num_transformer_blocks=4, mlp_units=[128],
    window_size = window_size, mlp_dropout=0.4, dropout=0.25).model

model.compile(
    loss="mse",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["mae"],
)

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
h = model.fit(
    x_train,
    y_train,
    epochs = 100,
    validation_split=0.2,
    batch_size = 32,
    callbacks = callbacks,
    verbose = 1
)


model.save("/home/cloud-user/evonano-ml/models/transformer_model")

plt.plot(np.arange(42), h.history['loss'], color = 'g', label = 'training loss')
plt.plot(np.arange(42), h.history['val_loss'], color = 'r', label = 'validation loss')
plt.xlabel("Training iterations")
plt.ylabel("Values")
plt.legend()
plt.title('Transformer Performance')
plt.savefig('plots/transformer_performance.png')


titles = ['MBTR GEM11',
'MBTR NCL11',
'MBTR OQL12v2',
'MBTR S1_51_3']
plt.figure(figsize=(10, 10))
for i in range(4):
  plt.tight_layout()
  plt.subplot('22'+str(i+1))
  plt.scatter(np.arange(num_feature), y_test[260 * i + 10], s= 3, label = 'label')
  plt.scatter(np.arange(num_feature), np.reshape(model.predict(np.expand_dims(x_test[260 * i + 10], axis = 0)), (num_feature,)), s=3, label = 'pred')
  plt.xlabel("MBTR Feature Index", fontsize ='medium')
  plt.ylabel("Column value", fontsize ='medium', rotation = 90)
  plt.title(titles[i])
  plt.legend()
plt.savefig('plots/transformer_evaluation.png')

