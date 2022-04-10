import numpy as np
import matplotlib.pyplot as plt
import training_data_generator as tg
import parameter_specification as params
from ensemble_model_class import ensemble_model

num_feature = params.num_feature
window_size = params.window_size

test_mbtr_dir = params.test_mbtr_dir 
train_mbtr_dir = params.train_mbtr_dir

x_train, y_train = tg.mbtr_mbtr_ds_generator(train_mbtr_dir, window_size = window_size, shuffle = False)
x_test, y_test = tg.mbtr_mbtr_ds_generator(test_mbtr_dir, window_size = window_size, shuffle = False)



def xgb_function(model, train = True):  
  if train:
    model.train(x_train, y_train)
    model.save('/home/cloud-user/evonano-ml/models/ensemble/')
  else:
    model.load_model('/home/cloud-user/evonano-ml/models/ensemble/')

model = ensemble_model(num_feature)
xgb_function(model, True)

print("Test set MAE:", model.evaluate(x_test, y_test))

titles = [
'MBTR CY511',
'MBTR GEM11',
'MBTR NCL11',
'MBTR OQL12v2',
'MBTR PAN11',
'MBTR S1_10_3',
'MBTR S1_51_3',
'MBTR ZIL15',
'MBTR PegZv1-5_1',
'MBTR PegZv2-5_1',
'MBTR DOX11Pz6',
'MBTR Wyc2.5nm'
]
timesteps = [300, 300, 300, 300, 300, 300, 300, 300, 120, 120, 200, 200]
plt.figure(figsize=(26, 12))
for i in range(12):
  plt.tight_layout()
  plt.subplot(4, 3, (i+1))
  plt.scatter(np.arange(num_feature), y_test[(timesteps[i] - 40) * i + 10], s= 3, label = 'label')
  plt.scatter(np.arange(num_feature), np.reshape(model.predict(x_test[(timesteps[i] - 40) * i + 10]), (num_feature,)), s=3, label = 'pred')
  plt.xlabel("MBTR Feature Index", fontsize ='medium')
  plt.ylabel("Column value", fontsize ='medium', rotation = 90)
  plt.title(titles[i])
plt.legend()
plt.savefig('plots/ensemble_evaluation.png')


i = 2
c1 = 'royalblue'
c2 = 'tomato'
fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('ghostwhite')
width = 0.35
ax.bar(np.arange(num_feature) - (width/2), y_test[260 * i + 10], width = width, align="center", label = 'Ground-truth Values', color = c1)
ax.bar(np.arange(num_feature) + (width/2), np.reshape(model.predict(x_test[260 * i + 10]), (num_feature,)), width = width, align="center", label = 'Predicted Values', color = c2)
ax.set_xticks(np.arange(num_feature), fontsize ='x-small', rotation = 90)

ax.set_xlabel("MBTR Feature Index", fontsize ='medium')
ax.set_ylabel("Column value", fontsize ='medium', rotation = 90)
plt.title("Sample MBTR Prediction for " + titles[i])
ax.legend()
plt.savefig('plots/ensemble_barplot_evaluation.png')