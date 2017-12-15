import os
import numpy as np
import hashlib
import math
from flask import Flask
import matplotlib.pyplot as plt
import csv

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

#write output to csv
def toCSVLine(data):
  return ','.join(str(d) for d in data)

if __name__ == '__main__':
    sc = SparkContext()

    data = sc.textFile(app.root_path+"/../data/test_cancer_final.csv")
    header = data.first()
    data = data.filter(lambda x: x != header)

    parsedTrainData = data.map(lambda x: parsePoint(x, 0, 25)).map(lambda x: x[1])

    ################################################
    # Build the model - LinearRegressionWithSGD
    #################################################
    model = LinearRegressionWithSGD.train(parsedTrainData, iterations=300, step=0.0000081)

    ##################################################
    # Evaluate the model on training data
    ##################################################
    valuesAndPreds = parsedTrainData.map(lambda p: (p.label, model.predict(p.features)))
    MSE = valuesAndPreds \
              .map(lambda vp: (vp[0] - vp[1]) ** 2) \
              .reduce(lambda x, y: x + y) / valuesAndPreds.count()
    print("Mean Squared Training Error = " + str(MSE))

    # Save and load model
    #model.save(sc, app.root_path+"/Models/pythonLinearRegressionWithSGDModel_cancer")
    #sameModel = LinearRegressionModel.load(sc, app.root_path+"/Models/pythonLinearRegressionWithSGDModel_cancer")

    ###################################################
    #running for test data
    ##################################################
    for i in range(1,10):
        parsedTestData = data.map(lambda x: parsePoint(x, i, 25)).map(lambda x: x[1])
        valuesAndPredsTest = parsedTestData.map(lambda p: (p.label, model.predict(p.features)))
        lines = valuesAndPredsTest.map(toCSVLine)
        j = 2005
        lines.saveAsTextFile(app.root_path+"/../data/cancer_prediction_"+str(i+j)+".csv")
        MSE_test = valuesAndPredsTest \
                  .map(lambda vp: (vp[0] - vp[1]) ** 2) \
                  .reduce(lambda x, y: x + y) / valuesAndPredsTest.count()
        print("Mean Squared Testing Error for year"+ str(i+j)+"= " + str(MSE_test))
    sc.stop()

    ###############################################
    #read the values for first two counties
    ##############################################
    for i in range(2006,2014):
        with open(app.root_path+"/../data/cancer_prediction_"+str(i)+".csv/part-00000","r") as file:
            for j, x in enumerate(file):
                if(j==1):
                    with open(app.root_path+"/../data/LR_AberVille.csv", 'a+') as target:
                        target.write(str(i)+","+x)
                elif(j==2):
                    with open(app.root_path+"/../data/LR_Acardia.csv", 'a+') as target:
                        target.write(str(i)+","+x)
                elif(j>2):
                    break

    ##############################################
    #plotting the graph for the two counties
    ##############################################
    for i in {"AberVille","Acardia"}:
        with open(app.root_path + "/../data/LR_"+str(i)+".csv", 'r') as file:
            plots = csv.reader(file, delimiter=',')
            year =[]
            y=[]
            y_pred=[]
            for row in plots:
                year.append(int(row[0]))
                y.append(float(row[1]))
                y_pred.append(float(row[2]))
        plt.plot(year, y, label="Actual value")
        plt.plot(year, y_pred, label="Predicted value")
        plt.ylim(ymin=150)
        plt.savefig(app.root_path + "/../result/LR_"+str(i)+".png")