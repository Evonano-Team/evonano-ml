from xgboost import XGBRegressor
from joblib import dump, load
import numpy as np
import os
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

class ensemble_model():
  """
  A class that manages all functionality of the ensemble model.

  This class handles functions such as training, predicting, evaluating,
  etc. presenting the object of a class as a single model.  

  Attributes
  ----------
  x_models : list of XGBRegressor.model objects
      List of models in order representing each feature; initialized or trained
  num_feature : int
      Number of features in the training data

  Methods
  -------
  train(x, y):
      Loops through and Trains all the models
  predict(test_sample):
      Predicts the outcome, through the trained models
  evaluate(x, y):
      Calculates the Mean Absolute Error for given data
  save(directory):
      Saves the models into specified dirctory
  load_model(directory):
      Loads the models from specified directory
  """
  
  def __init__(self, num_feature):
    """
    Constructs all the necessary attributes for the model object.
    
    This function creates one XGBoost model for the specified number
    of features each and initializes a list of the models.
    
    Parameters
    ----------
    x_models : list of XGBRegressor.model objects
        List of models in order representing each feature
    num_feature : int
        Number of features in the training data
    """
    self.x_models = []
    self.num_feature = num_feature
    for i in range(self.num_feature):
      self.x_models.append(XGBRegressor(objective='reg:squarederror', n_jobs=-1, verbosity=1))
      
  def train(self, x, y):
    """
    Loops through and Trains all the models
    
    Reads each model from the x_models attribute, takes only the data for
    feature corresponding to the model, and trains the model.
    
    Parameters
    ----------
    x : array of float; shape (number of samples, window_size, num_feature)
        Windowed training data.
    y : array of float; shape (number of samples, num_feature)
        Training Label Vector
    """
    for i in tqdm(range(self.num_feature)):
      x_t = [x[j,:,i] for j in range(x.shape[0])]
      y_t = [y[j][i] for j in range(y.shape[0])]
      self.x_models[i].fit(x_t, y_t)
      
  def predict(self, test_sample):
    """
    Predicts the outcome, through the trained models
    
    Reads each model from the x_models attribute, takes only the data for
    feature corresponding to the model, and predicts the outcome for that 
    feature. The whole vector is collected in a list.
    
    Parameters
    ----------
    test_sample : array of float; shape (number of samples, window_size, num_feature)
        Windowed data for prediction
        
    Returns
    -------
    Array of float; shape (1, num_feature)
        The whole vector of prediction for the passed sample
    """
    pred = []
    for i in range(self.num_feature):
      x_t = test_sample[:, i]
      prediction = self.x_models[i].predict(np.expand_dims(x_t, axis = 0))
      pred.append(prediction[0])
    return np.array(pred).reshape(1, -1)
  
  def evaluate(self, x, y):
    """
    Provides the avearge of mean absolute error values for all the models for passed data
    
    This function reads each model from the x_models attribute, predicts the
    outcome, measures MAE for each model and calculates the average.
    
    Parameters
    ----------
    x : array of float; shape (number of samples, window_size, num_feature)
        Windowed data for evaluation
    y : array of float; shape (number of samples, num_feature)
        Label Vector for MAE calculation
        
    Returns
    -------
    float
        The avaerage MAE for the passed data
    """
    pred_error = []
    for i in range(x.shape[0]):
        prediction = self.predict(x[i])
        pred_error.append(mean_absolute_error(np.expand_dims(y[i], axis = 0), prediction))
    avg_mae = np.mean(pred_error)
    return avg_mae
  
  def save(self, directory):
    """
    Saves the models from the x_models attribute into specified dirctory.
    
    Parameters
    ----------
    directory : str
        absolute or relative directory for saving the models
    """
    try: 
        os.mkdir(directory) 
    except OSError as error:
        pass
    if directory[-1] != '/':
        directory = directory + '/'
    for i in range(self.num_feature):
      dump(self.x_models[i], directory + 'xgbmodel_' + str(i).zfill(2) + '.joblib')
      
  def load_model(self, directory):
    """
    Loads the models from specified dirctory and updates the x_models attribute.
    
    Parameters
    ----------
    directory : str
        absolute or relative directory for saving the models
    """
    if directory[-1] != '/':
        directory = directory + '/'
    self.x_models = []
    for i in range(self.num_feature):
      x_model = load(directory + 'xgbmodel_' + str(i).zfill(2) + '.joblib')
      self.x_models.append(x_model)
