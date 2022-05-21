from tensorflow import keras
from ensemble_model_class import ensemble_model
from parameter_specification import num_feature

def load_model(model_name, num_feature, directory):
    """
    Loads the sepcified model and returns the object representing that model.
    
    This function is a utility function that loads any of the models used in
    this project. The function reads the model from the specified directory 
    through methods applicable to the type of model.
    
    Parameters
    ----------
    model_name : str
        ``transfomer`` or``ensemble_model``, for simulation part or
        ``sasa_model`` for SASA calculation
    num_feature : int
        Number of features in the training data
    directory : str
        absolute or relative directory for saving the models
      
    Returns
    -------
    object
        The object of the corresponding model's class
    """
    if model_name == 'transformer' or model_name == 'sasa_model':
        model = keras.models.load_model(directory)
        return model
    elif model_name == 'ensemble':
        model = ensemble_model(num_feature)
        model.load_model(directory)
        return model

    
