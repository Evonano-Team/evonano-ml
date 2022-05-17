import pandas as pd
import os.path
import matplotlib.pyplot as plt
import csv
import numpy as np
import re

if __name__ == '__main__':
    sasa_ranges = []
    for file in os.listdir('/home/cloud-user/evonano-ml/data/processed/test_set/label/'):
        df = pd.read_csv('/home/cloud-user/evonano-ml/data/processed/test_set/label/' + file, delimiter=';')
        sasa_max = df['TOTAL'].max()
        sasa_min = df['TOTAL'].min()
        sasa_ranges.append((sasa_min, sasa_max, re.sub('_sasa.csv', '', file)))
    sasa_ranges.sort(key=lambda x: x[0])
    for item in sasa_ranges:
        print(item)
        plt.scatter(item[0], item[1])
    plt.xlabel('min')
    plt.ylabel('max')
    for i in range(9):
        n = 0 - i
        plt.annotate(sasa_ranges[n][2], (sasa_ranges[n][0] + 500, sasa_ranges[n][1] - 500))
    plt.savefig('plots/sasa_range_test.png')