import pandas as pd
import csv
if __name__ == "__main__":

    with open("C:\SBU\Fall2017\BigData\Project\BDProject\CorrelationOfTopicsAndMortalityRates.txt") as f:
        content = f.read().split("), (")
    #scores = content[0].split(",")
    content[0] = content[0][1:]
    content[-1] = content[-1][:-2]
    print(content)
    li = [0] * (len(content)+3)
    for i in content:
        x = i.split(", ")
        li[int(x[0])] = (x[0],float(x[1]))

    li = li[3:]
    li.sort(key=lambda x: x[1])
    #top = li[:11]
    top10 = [0] * 10

    #Find the positively correlated words
    for i in li[:11]:
        top10.append(i[0])
    neg10 = top10[10:]
    bottom10 = [0] * 10

    #Find the negatively correlated words
    for i in li[-11:]:
        bottom10.append(i[0])
    pos10 = bottom10[10:]
    print("Negatively Correlated:",neg10)
    print("Positively Correlated",pos10)

    #df1 = pd.read_csv("C:\\SBU\\Fall2017\\BigData\\Project\\BDProject\\1to3grams.csv",engine='python')
    #df1 = df1.loc[df1['value'].astype(str)=="535"]
    #df1.to_csv("C:\\SBU\\Fall2017\\BigData\\Project\\BDProject\\T535.csv")
