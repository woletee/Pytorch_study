import pandas as pd
data=pd.read_csv("energydata_complete.csv")
data=data.drop(columns=["date"])
cols=data.columns
num_cols=data._get_numeric_data().columns
list(set(cols)-set(num_cols))
