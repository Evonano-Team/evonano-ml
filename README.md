# evonano-ml
Predicting SASA through Descriptors

#### ensemble_model_class.py: 

- Function: Creates a class for ensemble model that provides methods to handle different operations for the stacked model  
- Input: NA 
- Output: Creates an instance of the class 

#### model_utils.py: 
- Function: Provides different utility functions for loading all the models just by specifying the names (‘transformer’, ‘ensemble’, ‘sasa_model’) and directory 
- Input: Model Name, Saved Directory 
- Output: Saved Model 

#### parameter_specification.py: 
- Function: Defines global set of parameters to be used by all the models, also checks the correctness of the data in the specified directory 
- Input: None 
- Output: Model Parameters 

#### train_sasa_calculation_model.py: 
- Function: Defines the sasa model, trains it for 500 epochs, loads the best weight, saves the model, creates performance plots, and provides evaluation scores 
- Input: train/test data, number of features, window size 
- Output: Trained model, model performance plots 

#### training_transformer_model.py: 
- Function: Defines a class with utility methods for the transformer model, trains it for 100 epochs, saves the model, creates performance plots, and provides evaluation scores 
- Input: train/test data, number of features, window size 
- Output: Trained model, model performance plots 

#### training_ensemble_xgb.py: 

- Function: Creates an instance of the ensemble model, trains it, and saves the model, creates performance plots, and provides evaluation scores 
- Input: train/test data, number of features, window size 
- Output: Trained model, model performance plots 

#### training_data_generator.py: 
- Function: Reads MBTR, or SASA files and creates training/test pair based on specified attributes, handles difference of time steps in different sets of data 
- Input: Directory of data, Window Size, Shuffle 
- Output: Train/Test data pairs 
