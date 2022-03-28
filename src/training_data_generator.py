import numpy as np
import pandas as pd
import os

def mbtr_ds_generator(directory, timestep_size = 300):
  ptr = 0
  if directory[-1] != '/':
    directory = directory + '/'
  for mbtr_file in sorted(os.listdir(directory)):
    if ptr == 0:
      x_mbtr = pd.read_csv(directory + str(mbtr_file))
      x = np.array(x_mbtr.values.tolist())
      ptr = 1
    else:
      x_mbtr = pd.read_csv(directory + str(mbtr_file))
      x = np.concatenate((x, np.array(x_mbtr.values.tolist())), axis = 0)
    print("MBTR: ", mbtr_file, "Shape:", x.shape)

  print("MBTR Shape:", x.shape)
  return x

def sasa_ds_generator(directory, timestep_size = 300):
  ptr = 0
  if directory[-1] != '/':
    directory = directory + '/'
  for sasa_file in sorted(os.listdir(directory)):
    if ptr == 0:
          print("SASA: ", sasa_file)
          x_sasa = pd.read_csv(directory + str(sasa_file), sep=';')
          x = np.array(x_sasa['TOTAL'])
          ptr = 1
    else:
          print("SASA: ", sasa_file)
          x_sasa = pd.read_csv(directory + str(sasa_file), sep=';')
          x = np.concatenate((x, np.array(x_sasa['TOTAL'])), axis = 0)
  print("SASA Shape:", x.shape)
  return x

def mbtr_mbtr_ds_generator(directory, window_size, timestep_size = 300, sasa_dir = None, shuffle = False):
  dataset_x = []
  dataset_y = []
  y_sasa = []
  i = 0
  mbtr = mbtr_ds_generator(directory, timestep_size = timestep_size)
  if sasa_dir is not None:
    sasa = sasa_ds_generator(sasa_dir, timestep_size = timestep_size)
    if shuffle == True:
      idx = np.arange(mbtr.shape[0])
      np.random.shuffle(idx)
      mbtr = mbtr[idx]
      sasa_dir = sasa_dir[idx]
    assert mbtr.shape[0] == sasa.shape[0], "MBTR and SASA have mismatching shapes"
  elif shuffle == True:
    np.random.shuffle(mbtr)
  while True:
    if i  + 1 == mbtr.shape[0]:
      break
    if (i % timestep_size) >= (window_size - 1):
      if (i + 1) % timestep_size == 0:
        i = i + 1
        continue
      dataset_x.append(mbtr[i - window_size + 1 : i + 1])
      dataset_y.append(mbtr[i + 1])
      if not sasa_dir is None:
        y_sasa.append(sasa_dir[i + 1])

    i = i + 1
  if not sasa_dir is None:
    return np.array(dataset_x), np.array(dataset_y), np.array(y_sasa)
  return np.array(dataset_x), np.array(dataset_y)