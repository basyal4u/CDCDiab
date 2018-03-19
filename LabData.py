import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read the data file form lib. 'TCHOL_I' is for 15-16 and 'TCHOL_H' is 13-14
from functools import reduce 



df1 = pd.read_sas('ALB_CR_I.XPT')
df2 = pd.read_sas('BIOPRO_I.XPT')
df3 = pd.read_sas('CBC_I.XPT')
df4 = pd.read_sas('CHLMDA_I.XPT')
df5 = pd.read_sas('DIQ_I.XPT')
df6 = pd.read_sas('FASTQX_I.XPT')
df7 = pd.read_sas('FLDEP_I.XPT')
df8 = pd.read_sas('FLDEW_I.XPT')
df9 = pd.read_sas('GHB_I.XPT')
df10 = pd.read_sas('HDL_I.XPT')
df11 = pd.read_sas('HEPA_I.XPT')
df12 = pd.read_sas('HEPB_S_I.XPT')
df13 = pd.read_sas('HEPBD_I.XPT')
df14 = pd.read_sas('HEPC_I.XPT')
df15 = pd.read_sas('HEPE_I.XPT')
df16 = pd.read_sas('HIV_I.XPT')
df17 = pd.read_sas('HSV_I.XPT')
df18 = pd.read_sas('PBCD_I.XPT')
df19 = pd.read_sas('TCHOL_I.XPT')
df20 = pd.read_sas('TRICH_I.XPT')
df21 = pd.read_sas('UCFLOW_I.XPT')
df22 = pd.read_sas('UCPREG_I.XPT')

df23 = pd.read_sas('DEMO_I.XPT')











dfs = [df1, df2, df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22]

dfs1=[df3,df10,df19,df5]

df_final = reduce(lambda left,right: pd.merge(left,right,how='outer', left_on=["SEQN"], right_on = ["SEQN"]), dfs1)

for variable in df_final.columns.values:
    num_missing_mean_median(df_final,variable)


df_final.columns.values

df_final.to_csv("final1.csv")



import numpy as np
import pandas as pd

def num_missing_mean_median(df, variable, prefix="", mean=True):
    indicator = ""
    if prefix=="":
        indicator = variable+ "_" + "missing"
    else:
        indicator = prefix + "_"+ "missing"
    df[indicator] = np.where(df[variable].isnull(),1,0)
    replaceValue = 0
    if mean== True:
        replaceValue = df[variable].mean()
    else:
        replaceValue = df[variable].median()
    df[variable].fillna(replaceValue, inplace= True)
    return df

# you can use this method or use the dummy coding method defined in dummy_coding
def cat_missing_as_category(df, variable, prefix=""):
    indicator = ""
    if prefix=="":
        indicator = variable+ "_" + "missing"
    else:
        indicator = prefix + "_"+ "missing"
    df[indicator] = np.where(df[variable].isnull(),1,0)
    return df

def cat_missing_mode(df, variable, prefix=""):
    indicator = ""
    if prefix=="":
        indicator = variable+ "_" + "missing"
    else:
        indicator = prefix + "_"+ "missing"
    df[indicator] = np.where(df[variable].isnull(),1,0)
    df[variable].fillna(df[variable].mode().mode[0], inplace=True)
    return df


dataset3 = pd.merge(dataset, dataset2,  how='left', left_on=["SEQN"], right_on = ["SEQN"])

X = dataset.iloc[:, :1].values
y = dataset.iloc[:, 1:3].values

dataset.to_csv('out1.csv')

# Counting the null values

'''null_counts = dataset.isnull().sum()
null_counts[null_counts > 0].sort_values(ascending=False)

null_counts = dataset1.isnull().sum()
null_counts
file = open('TCHOL_I.xpt', encoding = "utf8")
null_counts = dataset1.isnull().sum()
null_counts
file = open('DIQ_I.xpt', encoding = "utf8")'''


# Removing the missing data

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis =0)
imputer = imputer.fit(y[:, 0:2])
y[:, 0:2]= imputer.transform(y[:, 0:2])

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# for merging the file
def readf(filename):
    lines=file(filename).readlines()
    for i in lines:
        data=i.split()
    return data

fa=readf('a1.txt')
fb=readf('b1.txt')

lines=[]
for i in fa:
    s=fa[i]+fb[3]
    s+='\n'
    lines.append(s)

with open ('c.txt','w') as f:
    f.writelines(lines)
    

'''with open('file3.txt', 'w') as file3:
    with open('TCHOL_I.xpt', 'r') as file1:
        with open('TCHOL_H.xpt', 'r') as file2:
            for line1, line2 in zip(file1, file2):
                print(line1.strip(), line2.strip(), file=file3)'''
                

