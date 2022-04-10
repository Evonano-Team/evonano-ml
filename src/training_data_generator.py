import numpy as np
import pandas as pd
import os

def mbtr_ds_generator(directory):
  ptr = 0
  timestep_size = []
  if directory[-1] != '/':
    directory = directory + '/'
  for mbtr_file in sorted(os.listdir(directory)):
    x_mbtr = pd.read_csv(directory + str(mbtr_file))
    timestep_size.append(x_mbtr.shape[0])
    if ptr == 0:  
      x = np.array(x_mbtr.values.tolist())
      ptr = 1
    else:
      x = np.concatenate((x, np.array(x_mbtr.values.tolist())), axis = 0)
    print("MBTR: ", mbtr_file, "Shape:", x.shape)

  print("MBTR Shape:", x.shape)
  print("timesteps: ", len(timestep_size))
  return x, timestep_size

def sasa_ds_generator(directory):
  ptr = 0
  if directory[-1] != '/':
    directory = directory + '/'
  for sasa_file in sorted(os.listdir(directory)):
    print("SASA: ", sasa_file)
    x_sasa = pd.read_csv(directory + str(sasa_file), sep=';')
    if ptr == 0:
          x = np.array(x_sasa['TOTAL'])
          ptr = 1
    else:
          x = np.concatenate((x, np.array(x_sasa['TOTAL'])), axis = 0)
  print("SASA Shape:", x.shape)
  return x

def mbtr_mbtr_ds_generator(directory, window_size, sasa_dir = None, shuffle = False):
  dataset_x = []
  dataset_y = []
  y_sasa = []
  i = 0
  j = 0
  mbtr, timestep_size = mbtr_ds_generator(directory)
  if sasa_dir is not None:
    sasa = sasa_ds_generator(sasa_dir)
    if shuffle == True:
      idx = np.arange(mbtr.shape[0])
      np.random.shuffle(idx)
      mbtr = mbtr[idx]
      sasa = sasa[idx]
    assert mbtr.shape[0] == sasa.shape[0], "MBTR and SASA have mismatching shapes"
  elif shuffle == True:
    np.random.shuffle(mbtr)


  for i in range(len(timestep_size)):
    if i == 0:
        z = 0
    else:
        z = sum(timestep_size[:i])
    for j in range(z, sum(timestep_size[ : i + 1]) - window_size):
      dataset_x.append(mbtr[j : j + window_size])
      dataset_y.append(mbtr[j + window_size])
      if not sasa_dir is None:
        y_sasa.append(sasa[j + window_size])
 
  '''
  while True:
    if i + 1 == mbtr.shape[0]:
      j = j + 1
      break
    if (i % timestep_size[j]) >= (window_size - 1):
      if (i + 1) % timestep_size[j] == 0:
        i = i + 1
        
        continue
      dataset_x.append(mbtr[i - window_size + 1 : i + 1])
      dataset_y.append(mbtr[i + 1])
      if not sasa_dir is None:
        y_sasa.append(sasa[i + 1])

    i = i + 1
  '''
  if not sasa_dir is None:
    return np.array(dataset_x), np.array(dataset_y), np.array(y_sasa)
  return np.array(dataset_x), np.array(dataset_y)