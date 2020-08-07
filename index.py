import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset 
dataset = pd.read_csv('./Data.csv')
x = dataset.iloc[:, :-1].values # grab values in all rows and all but the last columns
y = dataset.iloc[:, -1].values # grab values in all rows for the last column
# print(x)
# print(y) 
# print("".join(['-' for i in range(40)]))
print(type(x))

# Handle Missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # replace missing values as defined in np with 'mean' strategy
imputer.fit(x[:, 1:3])  # compute the missing values for all rows for 1st and 2nd calumn
x[:, 1:3] = imputer.transform(x[:, 1:3])  # replace all rows for 1st and second column with imputers version
# print(x)
# print("".join(['-' for i in range(40)]))


# Transform and Encode Categorical Data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# use oneHotEncoder to trasform 0th column, keep the others unchanged
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x)) # transform the x and convert it to np array
# print(x)
# print("".join(['-' for i in range(40)]))


# Transform and Encode The Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# print(y)
# print("".join(['-' for i in range(40)]))


# Split dataset into Training and Testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 1) # split 80/20 and seed random with 1
print(X_train)
print("".join(['-' for i in range(40)]))


# Sacle Features using Standarization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_test)
print("".join(['-' for i in range(40)]))
print(X_train)
