from xgboost import XGBRegressor
from joblib import dump, load
import numpy as np
import os
from sklearn.metrics import mean_absolute_error

class ensemble_model():
  def __init__(self, num_feature):
    self.x_models = []
    self.num_feature = num_feature
    for i in range(self.num_feature):
      self.x_models.append(XGBRegressor(objective='reg:squarederror', n_jobs=-1, verbosity=1))
  def train(self, x, y):
    for i in range(self.num_feature):
      print(i, " Started")
      x_t = [x[j,:,i] for j in range(x.shape[0])]
      y_t = [y[j][i] for j in range(y.shape[0])]
      self.x_models[i].fit(x_t, y_t)
      print(i, " Done")
  def predict(self, test_sample):
    pred = []
    for i in range(self.num_feature):
      x_t = test_sample[:, i]
      prediction = self.x_models[i].predict(np.expand_dims(x_t, axis = 0))
      pred.append(prediction[0])
    return np.array(pred).reshape(1, -1)
  def evaluate(self, x, y):
    pred_error = []
    for i in range(x.shape[0]):
        prediction = self.predict(x[i])
        pred_error.append(mean_absolute_error(np.expand_dims(y[i], axis = 0), prediction))
    avg_mae = np.mean(pred_error)
    return avg_mae
  def save(self, directory):
    try: 
        os.mkdir(directory) 
    except OSError as error:
        pass
    if directory[-1] != '/':
        directory = directory + '/'
    for i in range(self.num_feature):
      dump(self.x_models[i], directory + 'xgbmodel_' + str(i).zfill(2) + '.joblib')
  def load_model(self, directory):
    if directory[-1] != '/':
        directory = directory + '/'
    self.x_models = []
    for i in range(self.num_feature):
      x_model = load(directory + 'xgbmodel_' + str(i).zfill(2) + '.joblib')
      self.x_models.append(x_model)