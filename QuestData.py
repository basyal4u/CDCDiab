# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read the data file form lib. 'TCHOL_I' is for 15-16 and 'TCHOL_H' is 13-14
from functools import reduce 

df1 = pd.read_sas('BPQ_I.XPT')

df2 = pd.read_sas('CDQ_I.XPT')

df3 = pd.read_sas('DPQ_I.XPT')
df4 = pd.read_sas('ECQ_I.XPT')

df5 = pd.read_sas('HOQ_I.XPT')

df6 = pd.read_sas('MCQ_I.XPT')

df7 = pd.read_sas('SMQ_I.XPT')

df8 = pd.read_sas('SMQRTU_I.XPT')

df9= pd.read_sas('SMQSHS_I.XPT')
df10 = pd.read_sas('DIQ_I.XPT')

# Merging the file
dfs=[df1, df2, df3,df4,df5,df6,df7,df8,df9,df10]

df_final = reduce(lambda left,right: pd.merge(left,right,how='outer', left_on=["SEQN"], right_on = ["SEQN"]), dfs)

#df_diq_bpq = pd.merge(df1,df10,how='left', left_on=["SEQN"], right_on = ["SEQN"])



#df_final.iloc[:,[1,2,7,"DIQ010"]]

# Creating a set from the large chunk
df_new = df_final.loc[:,["SEQN","BPQ020","BPQ080","DIQ010"]]

# Data Filteration. Removed code, 3 and 9 and null value.
df_new = df_new[df_new.DIQ010 != 3]
df_new = df_new[df_new.DIQ010 != 9]
df_new = df_new[df_new.DIQ010.notnull()]

df_new = df_new[df_new.BPQ020.notnull()]
df_new = df_new[df_new.BPQ080.notnull()]

# Data split in Train and Test

df_new= df_new.loc[:,["BPQ020","BPQ080","DIQ010"]]

from sklearn.model_selection import train_test_split
attributes = ["BPQ020","BPQ080"]
train_x1, test_x1, train_y1, test_y1 = train_test_split(df_new[attributes], df_new["DIQ010"], test_size=0.25, random_state=123)

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(train_x1, train_y1)
print(model)
# make predictions
expected = test_y1
predicted = model.predict(test_x1)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
print("Accuracy:",metrics.accuracy_score(expected, predicted))





test = df_final.isnull().apply(sum,axis=0)

for i in df_final["BPD035"]:
    if 
    