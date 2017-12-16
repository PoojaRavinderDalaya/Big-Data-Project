import os
import numpy as np
import hashlib
import math
from flask import Flask

from datetime import datetime
from pyspark import SparkContext, SQLContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.evaluation import RegressionMetrics


import copy
import pandas as pd
from pylab import rcParams
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
import csv


app = Flask(__name__)

# Load and parse the data
def parsePoint(line):
    valueList = line.split(',')
    county = valueList[0]
    values = [float(x) for x in valueList[1:]]
    return (county, values)

def plotGraph(x):
    plt.plot(x[2])
    plt.plot(x[3], color='orange')
    plt.title('AR Error: %f' % (x[1]) +' for county '+x[0])
    plt.ylabel('Number of accidents due to cancer')
    plt.xlabel('From 2009 to 2014 ----->')
    plt.show()

# AR model for Time Series Analysis

def AR_TS_multipleYears(x):
    X = x[1]
    train, test = X[0:26], X[26:]
    # Training the model using data from 1980 to 2008
    model = AR(train)
    model_fit = model.fit()
    # predict from 2009 to 2014
    predictions = model_fit.predict(start=len(train), end=len(train) + len(test) + 3, dynamic=False)
    #error_for_each_county = mean_squared_error(test, predictions)
    # print(error)
    return (x[0],test,predictions)

def AR_TS_2014(x):
    X = x[1]
    train, test = X[:-1], X[-1:]
    # Training the model using data from 1980 to 2013
    model = AR(train)
    model_fit = model.fit()
    # predicting 2014
    predictions = model_fit.predict(start=len(train),end=len(train),dynamic=False)
    error_for_each_county = math.pow(test-predictions,2)
    return (x[0],error_for_each_county,test,predictions)

def predict_AR_TS_2015(x):
    X = x[1]
    train = X
    # Training the model using data from 1980 to 2013
    model = AR(train)
    model_fit = model.fit()
    # predicting 2014
    prediction = model_fit.predict(start=len(train), end=len(train), dynamic=False)
    prevYear = X[-1]
    difference = abs(prevYear-prediction[0])
    return (difference,x[0])


if __name__ == '__main__':
    sc = SparkContext()

    #data = sc.textFile(app.root_path+"/Files/test_chronic_final.csv")
    data = sc.textFile(app.root_path+"/Files/test_Self_harm_final.csv")
    header = data.first()
    data = data.filter(lambda x: x != header)

    ###################################################
    # Finding Mean Square Error for the year 2014 (Chronic Respiratory Diseases)
    ##################################################

    # rdd = data.map(parsePoint).map(lambda x: AR_TS_2014(x))


    # valuesAndPreds = rdd.map(lambda x: (float(x[2]), float(x[3])))
    # values = rdd.collect()
    # print(values)
    # # Instantiate metrics object
    # metrics = RegressionMetrics(valuesAndPreds)
    #
    # # Squared Error
    # print("MSE = %s" % metrics.meanSquaredError)


    ###################################################
    # Predicting for 2015 to get counties with highest and least mortality rate (Self_harm)
    ##################################################
    rdd = data.map(parsePoint).map(lambda x: predict_AR_TS_2015(x)).sortByKey()
    predError = rdd.collect()

    # Top 2 counties
    top2 = predError[0:2]
    #[(0.00066169357814871432, 'Watauga County'), (0.00066466254200747699, 'Cheshire County')]
    print(top2)

    # Bottom 2 counties
    bottom2 = predError[-2:]
    #[(2.9020858892163801, 'Apache County'), (4.0458817756345624, 'Robeson County')]
    print(bottom2)

    ###################################################
    # Predicting for the next 4 years 2015-2018 (Self_harm)
    ##################################################

    rdd = data.map(parsePoint).map(lambda x: AR_TS_multipleYears(x))
    values = rdd.collect()


    yearList = ['2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018']

    counties=[]
    for x in values:
        for e,c in top2:
            if x[0] == c:
                counties.append(x)
        for e,c in bottom2:
            if x[0] == c:
                counties.append(x)
    print(counties)

    for x in counties:
        row = []
        with open(app.root_path + "/TSFiles/Self_harm_AR_"+x[0]+".csv", 'w') as out:
            row.append(["Year", "Y", "Ypred"])
            for i in range(len(yearList)):
                try:
                    row.append([yearList[i], x[1][i], x[2][i]])
                except IndexError:
                    row.append([yearList[i], "", x[2][i]])
            csv_out = csv.writer(out)
            csv_out.writerows(row)
