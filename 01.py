# 1.Demonstrate the following data preprocessing tasks using python libraries.
# a) Loading the dataset
# b) Identifying the dependent and independent variables
# c) Dealing with missing data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("suv_data.csv")
dataset.head()

x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
bool_series=pd.isnull(dataset["Gender"])
dataset[bool_series]
bool_series=pd.notnull(dataset["Gender"])
dataset[bool_series]
dataset[10:25]

new_data=dataset.dropna(axis=0,how="any")
new_data

dataset.replace(to_replace=np.nan,value=-99)
dataset["Gender"].fillna("No Gender",inplace=True)
dataset

print("Old data frame length: ", len(dataset))
print("New data frame length: ", len(dataset))
print("Number of rows with at least 1 NA value: ", len(dataset)-len(new_data))

new_df1=dataset.fillna(method="ffill")
new_df1

new_df3=dataset.dropna(how="all")
new_df3
