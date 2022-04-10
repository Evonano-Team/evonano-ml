import numpy as np
import training_data_generator as tg
import parameter_specification as params
from model_utils import load_model

num_feature = params.num_feature
window_size = params.window_size

test_mbtr_dir = params.test_mbtr_dir 
test_sasa_dir = params.test_sasa_dir

x_test, _, y_test = tg.mbtr_mbtr_ds_generator(test_mbtr_dir, window_size = window_size, sasa_dir = test_sasa_dir, shuffle = False)

print(x_test.shape, 'ytest shape')
xgb_model = load_model('ensemble', num_feature, '../models/ensemble/')
sasa_model = load_model('sasa_model', num_feature, '../models/sasa_calculation_model/model')


def combined_inference(data, window_size = 40, sim_steps = 260):
  for i in range(sim_steps):
    next_mbtr = xgb_model.predict(data[-window_size:]) 
    data = np.concatenate((data, next_mbtr), axis = 0)
  sasa = sasa_model.predict(np.expand_dims(data[-1], axis = 0))
  return sasa

titles = [
'CY511',
'GEM11',
'NCL11',
'OQL12v2',
'PAN11',
'S1_10_3',
'S1_51_3',
'ZIL15',
'PegZv1-5_1',
'PegZv2-5_1',
'DOX11Pz6',
'Wyc2.5nm'
]
timesteps = [300, 300, 300, 300, 300, 300, 300, 300, 120, 120, 200, 200]
for i in range(12):
    if i == 0:
        z = 0
    else:
        z = sum([timesteps[j] - window_size for j in range(1, 12) if j < i]) + timesteps[i - 1] - window_size + 1
    print(z)
    sim_steps =  timesteps[i] - window_size
    pred_sasa = combined_inference(x_test[z], window_size = 40, sim_steps = sim_steps)
    print("Predicted SASA for {}: {}\t {:.2f}\t| Actual SASA: {}\t{:.2f}".format(titles[i], pred_sasa[0][0], pred_sasa[0][0]/y_test[z], y_test[z + sim_steps], y_test[z + sim_steps]/ y_test[z]))
