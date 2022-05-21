import pandas as pd
import os.path
import matplotlib.pyplot as plt
import csv
import numpy as np
import re

if __name__ == '__main__':
    """
    Plots the ranges of SASA values in the test set realtive to each other.
    
    This is an utility function that loads the SASA values in the test set
    and takes the maximum and minimum SASA values for each configuration. 
    Then, a specified number are plotted in a scatter plot as (xi, yi) where,
    xi is the minimum and yi is the maximum SASA values.
    
    The plot is saved in a the provided directory.
    """
    sasa_ranges = []
    for file in os.listdir('../data/processed/test_set/label/'):
        df = pd.read_csv('../data/processed/test_set/label/' + file, delimiter=';')
        sasa_max = df['TOTAL'].max()
        sasa_min = df['TOTAL'].min()
        sasa_ranges.append((sasa_min, sasa_max, re.sub('_sasa.csv', '', file)))
    
    # Sorted the list of configurations based on minimum value for grouping together
    sasa_ranges.sort(key = lambda x: x[0])
    for item in sasa_ranges:
        print(item)
        plt.scatter(item[0], item[1])
    plt.xlabel('min')
    plt.ylabel('max')
    for i in range(9):
        n = 0 - i
        plt.annotate(sasa_ranges[n][2], (sasa_ranges[n][0] + 500, sasa_ranges[n][1] - 500))
    plt.savefig('plots/sasa_range_test.png')
