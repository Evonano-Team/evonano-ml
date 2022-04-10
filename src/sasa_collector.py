import os
import shutil


directory = '/media/evonano04/PDB_SYNTHETIC/'

for sfile in os.listdir(directory):
    sasa_dir = directory + sfile + '/' + 'SASA.csv'
    new_sasa_dir = '/home/cloud-user/evonano-backup/evonano-project/new_sasa/' + 't_' + sfile + '_sasa.csv'
    shutil.copy(sasa_dir, new_sasa_dir)