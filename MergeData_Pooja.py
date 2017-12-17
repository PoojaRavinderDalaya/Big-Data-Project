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
import csv

app = Flask(__name__)

def mergeFiles():
    path = app.root_path  #use your path
    #C:\SBU\Fall2017\BigData\Project\US_HealthMap
    df1 = pd.read_csv(app.root_path + "\\Edu.csv")
    df2 = pd.read_csv(app.root_path + "\\finalMortalityRates.csv")
    df1['FIPS'] = df1['FIPS'].astype(str)
    df2['FIPS'] = df2['FIPS'].astype(str)
    df_merge = pd.merge_ordered(df1, df2, on=['FIPS'], how='left')
    df_merge.to_csv(path + "\\Pooja.csv",index=False)
    print(df_merge.columns)

if __name__ == "__main__":

    mergeFiles()
    file1reader = csv.reader(open(app.root_path + "\\merged_final.csv"), delimiter=",", quotechar='|')
    file2reader = csv.reader(open(app.root_path + "\\plan.csv"), delimiter=",", quotechar='|')
    header1 = next(file1reader)  # header
    header2 = next(file2reader)  # header
    if header1[1]==header2[3]:
        print ("yes")
    else: print(header1[1],'-',header2[3],'nooooo')
    print(header1)
    print(header2)
    cnt = 0
    ans = []
    myFile = open('pooja.csv', 'w')
    with myFile:
        writer = csv.writer(myFile, delimiter=',', quotechar='|')
        for i in file1reader:
            print("check", i)
            for j in file2reader:
                print("check",j)
                if(i[1].lower() == j[3].lower()):
                    print("Entered")
                    ans.append(i+j)
                    newlist=i+j
                    print ('newwwwww',newlist)
                    writer.writerows(newlist)
                else:
                    print("Mistake")
                    continue
    print(ans[:5])
    print("Valllllllllllll")
    for i in file1reader:
        print('valllll',i)