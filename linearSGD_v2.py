import os
import numpy as np
import hashlib
import math
from flask import Flask

from datetime import datetime
from pyspark import SparkContext, SQLContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel


app = Flask(__name__)

# Load and parse the data
def parsePoint(line,startIndex,offset):
    valueList = line.split(',')
    county = valueList[0]
    values = [float(x) for x in valueList[1:]]
    return (county, LabeledPoint(values[startIndex+offset], values[startIndex:startIndex+offset]))



if __name__ == '__main__':
    sc = SparkContext()

    data = sc.textFile(app.root_path+"/CSVs/test_cancer_final.csv")
    header = data.first()
    data = data.filter(lambda x: x != header)

    # considering 5 years values
    numberOfyrs = 5

    #predict
    parsedTrainData2011 = data.map( lambda x: parsePoint(x, 26, numberOfyrs)).map(lambda x: x[1])

    # Build the model
    model = LinearRegressionWithSGD.train(parsedTrainData2011, iterations=300, step=0.0000081)

    #Evaluate the model on training data
    valuesAndPreds = parsedTrainData2011.map(lambda p: (p.label, model.predict(p.features)))
    print(valuesAndPreds.collect())
    MSE_2011 = valuesAndPreds.map(lambda vp: (vp[0] - vp[1]) ** 2)
    print("Mean Squared error 2011 for all counties")
    #
    # # Save and load model
    # model.save(sc, app.root_path+"/Models/pythonLinearRegressionWithSGDModel_cancer")
    # sameModel = LinearRegressionModel.load(sc, app.root_path+"/Models/pythonLinearRegressionWithSGDModel_cancer")

    # Evaluate the model on test data
    # valuesAndPredsTest= parsedTestData.map(lambda p: (p.label, model.predict(p.features)))
    # print(valuesAndPredsTest.collect())
    # MSE_Test = valuesAndPredsTest.map(lambda vp: (vp[0] - vp[1]) ** 2).reduce(lambda x, y: x + y) / valuesAndPredsTest.count()
    # print("Mean Squared Error Testing Data  = " + str(MSE_Test))


