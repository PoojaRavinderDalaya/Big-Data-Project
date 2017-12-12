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
def parsePoint(line):
    #count
    valueList = line.split(',')
    county = valueList[0]
    values = [float(x) for x in valueList[1:]]
    return (county, LabeledPoint(values[-1], values[1:-1]))



if __name__ == '__main__':
    sc = SparkContext()

    data = sc.textFile(app.root_path+"/CSVs/test_cancer_final.csv")
    header = data.first()
    data = data.filter(lambda x: x != header)

    parsedData = data.map(parsePoint).map(lambda x: x[1])

    # Build the model
    model = LinearRegressionWithSGD.train(parsedData, iterations=100, step=10)

    # Evaluate the model on training data
    valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
    print(valuesAndPreds.collect())
    MSE = valuesAndPreds \
              .map(lambda vp: (vp[0] - vp[1]) ** 2) \
              .reduce(lambda x, y: x + y) / valuesAndPreds.count()
    print("Mean Squared Error = " + str(MSE))

    # Save and load model
    model.save(sc, app.root_path+"/Models/pythonLinearRegressionWithSGDModel_cancer")
    sameModel = LinearRegressionModel.load(sc, app.root_path+"/Models/pythonLinearRegressionWithSGDModel_cancer")