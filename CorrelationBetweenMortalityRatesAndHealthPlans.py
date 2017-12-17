import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pyspark
import numpy as np
from pyspark.mllib.stat import Statistics
from pylab import *
from scipy.stats.stats import pearsonr

if __name__ == "__main__":

    #Simple Correlation
	
	sc = pyspark.SparkContext()
	csv = np.genfromtxt('C:\\SBU\\Fall2017\\BigData\\Project\\BDProject\\final_test_selfHarm.csv', delimiter=",",skip_header=3)

    second = csv[:, 10].astype(float)
    print(second[:3])
    third = csv[:, 17].astype(float)
    print(third[:3])
    
	a = []
    b = []

    for i in range(len(second)):
        if (math.isnan(second[i]) or math.isnan(third[i])):
            continue
        else:
            a.append(second[i])
            b.append(third[i])

    a = np.asarray(a)
    b = np.asarray(b)
    seriesX = sc.parallelize(a)
    seriesY = sc.parallelize(b)
    corrval = Statistics.corr(seriesX, seriesY, method="pearson")
    
	print(corrval)
