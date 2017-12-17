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

from scipy import stats
app = Flask(__name__)


if __name__ == "__main__":

    sc = pyspark.SparkContext()
    sql = pyspark.sql.SQLContext(sc)

    csv = np.genfromtxt('C:\\SBU\\Fall2017\\BigData\\Project\\BDProject\\TopicCardio_plan.csv', delimiter=",",skip_header=1)

    #Correlation of Topics vs Mortality Rates
    corrVec = []
	
	#Run a loop to compute correlations between all topics
    for cntr in range(1600,2003):
        second = csv[:, cntr].astype(float)
        third = csv[:, 2011].astype(float) #Mortality Rate
        a = []
        b = []
	
		#To eliminate nan and inf values
        for i in range(len(second)):
            if (math.isnan(second[i]) or math.isnan(third[i])):
                continue
            else:
                a.append(second[i])
                b.append(third[i])

        a = np.asarray(a)
        b = np.asarray(b)
        
		#Parallelize the input arrays
		seriesX = sc.parallelize(a)
        seriesY = sc.parallelize(b)
		
		#Compute the correlation
        corrval = Statistics.corr(seriesX, seriesY, method="pearson")
        
		#Append to the final list to get top 10 positively and negatively correlated topics
		corrVec.append((cntr,corrval))

    print(corrVec)
    