import pandas as pd
data=pd.read_csv("energydata_complete.csv")
data=data.drop(columns=["date"])
