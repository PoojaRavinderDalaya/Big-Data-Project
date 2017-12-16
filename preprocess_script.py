from flask import Flask
import pandas as pd

app = Flask(__name__)
def CSVForTimeSeries():
    df = pd.read_csv(app.root_path + "\\..\\data\\merged_final_Sravya.csv")
    df = df.loc[df['sex'] == 'Both']
    requiredColumns = []
    requiredColumns.append('location_name')
    requiredColumns.append('year_id')
    #requiredColumns.append('FIPS')
    requiredColumns.append('cancer_mx')
    new_df = df[requiredColumns]

    print(new_df.columns)
    new_df.to_csv(app.root_path + '\\..\\data\\test_cancer.csv', index=False)


    df = pd.read_csv(app.root_path + "\\..\\data\\test_cancer.csv")
    new_df = df.drop_duplicates(['location_name', 'year_id'])
    new_df_2 = pd.pivot_table(new_df,index='year_id', columns='location_name', values='cancer_mx')
    new_df_2.to_csv(app.root_path + "\\..\\data\\test_cancer_final.csv")

if __name__ =="__main__":
    CSVForTimeSeries()