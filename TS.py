import os
import numpy as np
import hashlib
import math
from flask import Flask

from datetime import datetime
from pyspark import SparkContext, SQLContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel


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

def AR_TS(x):
    X = x[1]
    train, test = X[0:26], X[26:]
    model = AR(train)
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
    error = mean_squared_error(test, predictions)
    print(error)
    return (x[0],error,test,predictions)


def ARIMA_model(x, p, d, q):
    X = x[1]
    train, test = X[0:26], X[26:]
    model = ARIMA(train, order=(p,d,q))
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
    # for i in range(len(predictions)):
    #     print('predicted=%f, expected=%f' % (predictions[i], test[i]))
    error = mean_squared_error(test, predictions)
    print(error)
    return (x[0], error, test, predictions)



if __name__ == '__main__':
    sc = SparkContext()

    data = sc.textFile(app.root_path+"/CSVs/test_cancer_final1.csv")
    header = data.first()
    data = data.filter(lambda x: x != header)

    #rdd = data.map(parsePoint).map(lambda x: AR_TS(x))

    rdd = data.map(parsePoint).map(lambda x: ARIMA_model(x,1,0,0))

    values = rdd.collect()
    print(values)
    yearList = ['2006','2007','2008','2009','2010','2011','2012','2013','2014']

    for x in values:
        row = []
        with open(app.root_path + "/TSFiles/cancer_ARIMA_"+x[0]+".csv", 'w') as out:
            row.append(["Year", "Y", "Ypred"])
            for i in range(len(yearList)):
                row.append([yearList[i], x[2][i], x[3][i]])
            csv_out = csv.writer(out)
            csv_out.writerows(row)
    #
