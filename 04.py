# 4. Write Python code to load the data from a CSV file and select the top 10 features using the
# chi-squared test. The selected features are to be printed on the console.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import chi2

df=pd.read_csv('loandata.csv')
df.head()

df=df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area', 'Loan_Status']]
df.head()

from sklearn.preprocessing import LabelEncoder
for col in df.columns:
	le=LabelEncoder()
	df[col]=le.fit_transform(df[col])
df.head()

x=df.iloc[:,0:6]
y=df.iloc[:,-1]

f_score=chi2(x,y)
f_score
p_value=pd.Series(f_score[1], index=x.columns)
p_value.sort_values(ascending=False,inplace=True)
p_value
p_value.plot(kind = 'bar')
plt.xlabel('Features', fontsize=20)
plt.ylabel('p_values', fontsize=20)
plt.title('chi squared test base on p value')
plt.show()
