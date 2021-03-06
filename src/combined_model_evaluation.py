import numpy as np
import training_data_generator as tg
import parameter_specification as params
from model_utils import load_model
from tqdm import tqdm

num_feature = params.num_feature
window_size = params.window_size

test_mbtr_dir = params.test_mbtr_dir 
test_sasa_dir = params.test_sasa_dir

# Reading the test data to observe inference outcome
x_test, _ = tg.mbtr_ds_generator(test_mbtr_dir)
y_test = tg.sasa_ds_generator(test_sasa_dir)

print(x_test.shape, y_test.shape)

# Loading both the simulation, and the SASA calculation models
xgb_model = load_model('ensemble', num_feature, '../models/ensemble/')
sasa_model = load_model('sasa_model', num_feature, '../models/sasa_calculation_model/model')


def combined_inference(data, window_size = 40, sim_steps = 260):
    """
    Calculates the SASA value after looping through the Simulation Model.

    This function integrates the whole pipeline and provides inference results
    after multiple steps of simulation. The first model is used in a loop to
    iteratively move closer to the desired step, then the second model calculates
    the SASA value.

    Parameters
    ----------
    data : array of float; shape (window_size, num_feature)
        The initial batch of data from the test set for simulation.
    window_size : int
        Number of samples representing consequtive timesteps to be provided as input.
    sim_steps : int
        Desired number of steps ahead in time for the target SASA value.

    Returns
    -------
    int
        The target SASA value 
    """
  for i in tqdm(range(sim_steps)):
    next_mbtr = xgb_model.predict(data[-window_size:]) 
    data = np.concatenate((data, next_mbtr), axis = 0)
  sasa = sasa_model.predict(np.expand_dims(data[-1], axis = 0))
  return sasa

# Titles of the test set designs
titles = [
'GEM11',
'GEM41',
'NCL11',
'NHQ51',
'OQL11_3',
'OQL13v2_3',
'PAN11v2_3b',
'PAN14v2_3',
'PAN31_3',
'S1_11R2_3',
'S1_11R4_3',
'S1_15_3'
]
timestep = 300
error_pc = 0
sim_steps = timestep - window_size - 1
for i in range(12):
    """
    Looping through all the test set design, to evaluate the inference for each of them.
    
    This loop is used to iterate over each test sample, and take the initial batch of
    data from each of them for the combined inference. For comparison, the predictions
    are displayed beside the ground-truth labels. The ``error_pc`` is a measure of the
    average devation from the actual label for the whole test set.
    """
    pointer = (i * timestep)
    pred_sasa = combined_inference(x_test[pointer : pointer + window_size], window_size = window_size, sim_steps = sim_steps)
    print('{} > {}; {}'.format(pointer, pointer + window_size - 1, (i + 1) * timestep - 1))
    print("Predicted SASA for {}: {:.2f}\t {:.2f}\t| Actual SASA: {:.2f}\t{:.2f}".format(titles[i], pred_sasa[0][0], pred_sasa[0][0]/y_test[pointer], y_test[(i + 1) * timestep - 1], y_test[(i + 1) * timestep - 1]/ y_test[pointer]))
    error_pc = error_pc + np.abs(pred_sasa[0][0] - y_test[(i + 1) * timestep - 1])
print(error_pc/12)
