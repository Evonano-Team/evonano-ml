import os
import re

def check_mbtr_sasa_pairing(mbtr_dir, sasa_dir, train = True):
  for mbtr_file, sasa_file in list(zip(sorted(os.listdir(mbtr_dir)), sorted(os.listdir(sasa_dir)))):
      
      if re.sub("_sasa", "", sasa_file) != re.sub("mbtr_data_ngrid2_", "", mbtr_file) and re.sub("_sasa", "", sasa_file[2:]) != re.sub("t_mbtr_data_ngrid2_", "", mbtr_file):
        if train:
          raise ValueError("There is an inconsistency in the MBTR and SASA directory of training")
        else:
          raise ValueError("There is an inconsistency in the MBTR and SASA directory of testing")
  if train:
    print("Training Directory check Successfully Completed.")
    print("Number of Features: {}".format(num_feature))
    print("Window Size: {}".format(window_size))
  else:
    print("Test Directory check Successfully Completed.")


num_feature = 72
window_size = 40

test_mbtr_dir = '/home/cloud-user/evonano-ml/data/processed/test_set/features'
train_mbtr_dir = '/home/cloud-user/evonano-ml/data/processed/training_set/features'

test_sasa_dir = '/home/cloud-user/evonano-ml/data/processed/test_set/label'
train_sasa_dir = '/home/cloud-user/evonano-ml/data/processed/training_set/label'

check_mbtr_sasa_pairing(train_mbtr_dir, train_sasa_dir, train = True)
check_mbtr_sasa_pairing(test_mbtr_dir, test_sasa_dir, train = False)
