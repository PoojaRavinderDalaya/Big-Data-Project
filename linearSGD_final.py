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
def parsePoint(line,startIndex,offset,flag):
    valueList = line.split(',')
    county = valueList[0]
    values = [float(x) for x in valueList[1:]]
    if(flag=="test" or flag=="train"):
        return (county, LabeledPoint(values[startIndex+offset], values[startIndex:startIndex+offset]))
    else:
        return (county,values[startIndex:startIndex + offset])

#write output to csv
def toCSVLine(data):
  return ','.join(str(d) for d in data)

def getDifference(line):
    return (line[0],abs(float(line[1][0][0])-float(line[1][1])))

def append(x):
    x[1][0].append(str(x[1][1]))
    return (x[0],x[1][0])

if __name__ == '__main__':
    sc = SparkContext()

    data = sc.textFile(app.root_path+"/../data/test_chronic_final.csv")
    header = data.first()
    data = data.filter(lambda x: x != header)

    parsedTrainData = data.map(lambda x: parsePoint(x, 0, 25, "train")).map(lambda x: x[1])

    ################################################
    # Build the model - LinearRegressionWithSGD
    #################################################
    model = LinearRegressionWithSGD.train(parsedTrainData, iterations=200, step=0.000185)

    ##################################################
    # Evaluate the model on training data
    ##################################################
    valuesAndPreds = parsedTrainData.map(lambda p: (p.label, model.predict(p.features)))

    MSE = valuesAndPreds \
              .map(lambda vp: (vp[0] - vp[1]) ** 2) \
              .reduce(lambda x, y: x + y) / valuesAndPreds.count()
    print("Mean Squared Training Error = " + str(MSE))

    # Save and load model
    #model.save(sc, app.root_path+"/Models/pythonLinearRegressionWithSGDModel_chronic")
    #sameModel = LinearRegressionModel.load(sc, app.root_path+"/Models/pythonLinearRegressionWithSGDModel_chronic")

    ###################################################
    #running for test data
    ##################################################
    for i in range(1,10):
        parsedTestData = data.map(lambda x: parsePoint(x, i, 25, "test"))#.map(lambda x: x[1])
        valuesAndPredsTest = parsedTestData.map(lambda p: (p[0],p[1].label, model.predict(p[1].features)))
        lines = valuesAndPredsTest.map(toCSVLine)
        j = 2005
        lines.saveAsTextFile(app.root_path+"/../data/chronic_prediction_"+str(i+j)+".csv")

        MSE_test = valuesAndPredsTest \
                      .map(lambda vp: (vp[1] - vp[2]) ** 2) \
                      .reduce(lambda x, y: x + y) / valuesAndPredsTest.count()
        print("Mean Squared Testing Error for year "+ str(i+j)+"= " + str(MSE_test))

    #####################################################
    # Predicting values for which actual are not given
    # and then taking top 2 counties and bottom 2 counties for
    #####################################################
    parsedPredictData = data.map(lambda x: parsePoint(x, 10, 25, "predict"))#.map(lambda x: x[1])
    prevData = data.map(lambda x: parsePoint(x, 34, 1, "predict"))
    PredLine = parsedPredictData.map(lambda p: (p[0],model.predict(p[1])))
    PredLine.saveAsTextFile(app.root_path + "/../data/chronic_prediction_2015.csv")

    diff = prevData.join(PredLine).map(lambda x:getDifference(x))\
        .sortBy(lambda x:x[1]).collect()

    # Put 2015 predicted data into base data RDD to predict more
    data_new = data.map(lambda x :x.split(",")).map(lambda x : (x[0],x[1:]))\
          .join(PredLine).map(lambda x: append(x))

    # Predict 2016 through 2018
    for i in range(2016,2019):
        new_pred = data_new.map(lambda p: (p[0],model.predict(p[1][-25:])))
        data_new = data_new.join(new_pred).map(lambda x: append(x))

    # Top 2 counties
    top2 = diff[0:2]
    print(top2)
    #('San Miguel County', 0.013135833354951387), ('Barrow County', 0.03151909917625062)]

    # Bottom 2 counties
    bottom2 = diff[-2:]
    print(bottom2)
    #[('San Juan County', 8.459624544626102), ('Bristol Bay Borough', 11.231922825686702)]

    ##################################################
    # Make csv for the best and the worst county
    ##################################################
    counties = []
    #counties = ['San Miguel County', 'Bristol Bay Borough']
    counties.append(top2[1][0])
    counties.append(bottom2[0][0])
    print(counties)

    dataPrint = data_new.filter(lambda x: x[0] in counties).collect()
    for x in dataPrint:
        row = []
        with open(app.root_path + "/../data/Diarrhea_LR_" + str(x[0]) + ".csv", 'w') as out:
            for i in range(39):
                j = 1980
                row.append([j + i, x[1][i]])
            csv_out = csv.writer(out)
            csv_out.writerows(row)



    counties = ["Barrow County","San Juan County"]
    for i in range(2006, 2015):
        for ii in ["00000","00001"]:
            with open(app.root_path + "/../data/chronic_prediction_" + str(i) + ".csv/part-"+ii, "r") as file:
                for j, x in enumerate(file):
                    #print(x)
                    y = x[:-1].split(',')
                    print(y)
                    if (y[0] in counties):
                        with open(app.root_path + "/../data/LR_"+y[0].replace(" ","_")+".csv", 'a+') as target:
                            if(i<2015):
                                target.write(str(i) + "," + y[1] + "," + y[2] + "\n")
                            else:
                                target.write(str(i) + ", ," + y[1] + "\n")
'''
    sc.stop()
    

    ##############################################
    #plotting the graph for the two counties
    ##############################################
    for i in counties:
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
    '''