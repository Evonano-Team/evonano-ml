import os
import shutil

"""
This is a utility script to copy the csv files of sasa for newer designs
as they are produced to the base directory containing previous files.

A prefix is used to differentiate between data gathered for varying timesteps.
"""
directory = './PDB_SYNTHETIC/'

for sfile in os.listdir(directory):
    sasa_dir = directory + sfile + '/' + 'SASA.csv'
    new_sasa_dir = './new_sasa/' + 't_' + sfile + '_sasa.csv'
    shutil.copy(sasa_dir, new_sasa_dir)
