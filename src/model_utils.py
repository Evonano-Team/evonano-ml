from tensorflow import keras
from ensemble_model_class import ensemble_model
from parameter_specification import num_feature

def load_model(model_name, num_feature, directory):
    if model_name == 'transformer' or model_name == 'sasa_model':
        model = keras.models.load_model(directory)
        return model
    elif model_name == 'ensemble':
        model = ensemble_model(num_feature)
        model.load_model(directory)
        return model

    