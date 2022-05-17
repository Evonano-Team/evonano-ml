import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint
import training_data_generator as tg
import parameter_specification as params

num_feature = params.num_feature
window_size = params.window_size

test_mbtr_dir = params.test_mbtr_dir 
train_mbtr_dir = params.train_mbtr_dir

test_sasa_dir = params.test_sasa_dir
train_sasa_dir = params.train_sasa_dir

x_train, _ = tg.mbtr_ds_generator(train_mbtr_dir)
x_test, _ = tg.mbtr_ds_generator(test_mbtr_dir)
y_train = tg.sasa_ds_generator(train_sasa_dir)
y_test = tg.sasa_ds_generator(test_sasa_dir)

def plot_sasa(x_test, y_test, NN_model, num_feature, name):
    c1 = 'slateblue'
    c2 = 'darkgray'
    plt.figure(figsize=(25, 10))
    plt.plot(range(y_test.shape[0]), y_test, color = c1)
    plt.plot(range(y_test.shape[0]), [NN_model.predict(x_test[i].reshape(-1, num_feature))[0][0] for i in range(x_test.shape[0])], color = c2)
    plt.xlabel("Training iterations")
    plt.ylabel("SASA Values")
    plt.legend(["Real Value", "Predicted Value"])
    plt.title('SASA Model Performance')
    plt.savefig(name)


print("Here", x_train.shape)
print("Here", y_train.shape)

NN_model = Sequential()
NN_model.add(Dense(256, kernel_initializer='normal', input_dim = num_feature, activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))
NN_model.summary()
NN_model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics=["mae"])

checkpoint_name = '../models/sasa_calculation_model/weights/best_weights.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [] # [checkpoint]

h = NN_model.fit(x_train, y_train, epochs = 50, batch_size= 32, validation_split = 0.3, verbose = 1, callbacks= callbacks_list)

plt.plot(np.arange(len(h.history['loss'])), h.history['loss'], color = 'g', label = 'training loss')
plt.plot(np.arange(len(h.history['loss'])), h.history['val_loss'], color = 'r', label = 'validation loss')
plt.xlabel("Training iterations")
plt.ylabel("Values")
plt.legend()
plt.title('SASA Model Performance')
plt.savefig('plots/sasa_model_50 epoch_performance.png')


#NN_model.load_weights(checkpoint_name)
#NN_model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics=["mae"])
NN_model.save("/home/cloud-user/evonano-ml/models/sasa_calculation_model/model")
print("Test Performance:", NN_model.evaluate(x_test, y_test))

plot_sasa(x_test, y_test, NN_model, num_feature, name = 'plots/sasa__50 epoch_predictions.png')

#plot_sasa(x_test[300:1201], y_test[300:1201], NN_model, num_feature, name = 'plots/sasa_sample_predictions.png')