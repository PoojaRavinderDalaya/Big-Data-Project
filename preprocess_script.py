from flask import Flask
import pandas as pd

app = Flask(__name__)
def CSVForPivotYears():
    df = pd.read_csv(app.root_path + "/Files/test_ChronicRespiratory.csv")
    new_df = df.drop_duplicates(['location_name', 'year_id'])
    new_df_2 = pd.pivot_table(new_df,index='location_name', columns='year_id', values='Chronic respiratory diseases_mx')
    new_df_2.to_csv(app.root_path + "/Files/test_chronic_final.csv")

if __name__ =="__main__":
    CSVForPivotYears()