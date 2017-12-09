from flask import Flask
from flask import render_template , request, redirect, url_for, send_from_directory, jsonify
import json
import pandas as pd
import csv
import numpy as np
import re
import math
import glob
from sparkts.timeseriesrdd import time_series_rdd_from_observations

app = Flask(__name__)

#splitting Mortality.csv based on cause_id
def split_csv():
    df = pd.read_csv(app.root_path + '/MortalityRates.csv', encoding='iso-8859-1')
    for i, g in df.groupby('cause_id'):
        g.to_csv(app.root_path + '/Data/'+'{}.csv'.format(i), header=False, index_label=False)

# adding column names after splitting
def addColNames():
    path = app.root_path  # use your path
    allFiles = glob.glob(path + "/Data/*.csv")
    for file_ in allFiles:
        df = pd.read_csv(file_)
        cols = [1]
        df.drop(df.columns[cols], axis=1, inplace=True)
        df.columns = ["location_id", "location_name", "FIPS", "cause_id", "cause_name", "sex_id", "sex", "year_id", "mx",
                     "lower", "upper"]
        df.to_csv(file_,index=False)

def renameColumns():
    path = app.root_path  # use your path
    allFiles = glob.glob(path + "/*.csv")
    for file_ in allFiles:
        df = pd.read_csv(file_)
        cause = df['cause_name'].iloc[0]
        df = df.rename(columns={'mx': cause+'_mx', 'lower':cause+'_lower', 'upper': cause+'_upper'})
        df.to_csv(file_, index=False)

def renameColumns2():
    path = app.root_path  # use your path
    allFiles = glob.glob(path + "/Data/*.csv")
    for file_ in allFiles:
        df = pd.read_csv(file_)
        cause = df['cause_name'].iloc[0]
        print(cause)
        df = df.rename(columns={'cause_name': cause+'_cause_name', 'cause_id':cause+'_cause_id'})
        df.to_csv(file_, index=False)

def removeIndex():
    path = app.root_path  # use your path
    allFiles = glob.glob(path + "/*.csv")
    for file_ in allFiles:
        df = pd.read_csv(file_, index=False)
        df.to_csv(file_, index=False)

def mergeFiles():
    path = app.root_path  #use your path 
    df1 = pd.read_csv(app.root_path + "/merged728.csv")
    df2 = pd.read_csv(app.root_path + "/plan.csv")
    df_merge = pd.merge(df1, df2, on=['FIPS'], how='left')
    df_merge.to_csv(path + "/merged_final.csv",index=False)
    print(df_merge.columns)

    # for file in glob.glob(path + "/Data/*.csv"):
    #     df1 = pd.read_csv(app.root_path + "/cancer.csv")
    #     df2 = pd.read_csv(file)
    #     df_merge = pd.merge(df1, df2, on=['location_id', 'location_name', 'FIPS', 'sex_id', 'sex', 'year_id'], how='left')
    # print(df_merge.columns)
    #df_merge.to_csv(path + "/Data/merged.csv",index=False)


def displayRecord():
    #df = pd.read_csv(app.root_path + '/cancer.csv', encoding='iso-8859-1') #cause_id : 410
    #df = pd.read_csv(app.root_path + '/CardioVascularDisease.csv', encoding='iso-8859-1') #cause_id : 491
    df = pd.read_csv(app.root_path + '/cancer.csv', encoding='iso-8859-1') #cause_id : 508
    df.columns = ["location_id", "location_name", "FIPS", "cancer_cause_id", "cancer_cause_name", "sex_id", "sex", "year_id", "cancer_mx",
                  "cancer_lower", "cancer_upper"]
    df.to_csv(app.root_path + '/cancer.csv', index=False)
    df.head(5)

if __name__ == '__main__':
    #split_csv()
    #addColNames()
    #displayRecord()
    #removeIndex()
    #renameColumns2()
    mergeFiles()
    #displayRecord()
