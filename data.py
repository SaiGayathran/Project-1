import pandas as pd
import numpy as np
ud=pd.read_csv('dataset.csv')
ud['GpuMemorySize']=ud['GpuMemorySize']/1024
ud.drop_duplicates(inplace=True)
ud=ud[ud['FPS']!=0]
ud.dropna(inplace=True)
quality_map = {"low": 0.25,"medium": 0.50,"high": 0.75,"ultra": 1.00}
ud["GameSetting"] = ud["GameSetting"].map(quality_map)
y=ud['FPS'].to_frame()
x=ud.drop(columns=["FPS"])
x_data=x.to_numpy()
y_train=y.to_numpy().flatten()
#print(x_train.shape)
#print(y_train.shape)
print(x.head())
print(x.tail())