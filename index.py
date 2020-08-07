import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset 
dataset = pd.read_csv('./Data.csv')
x = dataset.iloc[:, :-1].values # grab values in all rows and all but the last columns
y = dataset.iloc[:, -1].values # grab values in all rows for the last column

print(x)
print(y) 
print("".join(['-' for i in range(40)]))

# Handle Missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # replace missing values as defined in np with 'mean' strategy
imputer.fit(x[:, 1:3])  # compute the missing values for all rows for 1st and 2nd calumn
x[:, 1:3] = imputer.transform(x[:, 1:3])  # replace all rows for 1st and second column with imputers version
print(x)
print("".join(['-' for i in range(40)]))


# Transform and Encode Categorical Data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# use oneHotEncoder to trasform 0th column, keep the others unchanged
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x)) # transform the x and convert it to np array

print(x)
print("".join(['-' for i in range(40)]))


# Transform and Encode The Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)
print("".join(['-' for i in range(40)]))
