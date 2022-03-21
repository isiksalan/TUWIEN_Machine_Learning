import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
#the ones under here are for heat-map
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import time

dataset= pd.read_csv("weatherAUS.csv")
print(dataset)
print(dataset.describe())
print(dataset.head(10))
print(dataset.columns)


print(dataset.info())
# unique values in whole columns
print("UNIQUE VALUES IN COLUMNS")
print(dataset.nunique())

# missing values in columns in train_data
print("COUNT OF MISSING VALUES IN COLUMNS")
print(dataset.isnull().sum())

#percentage of missing values in columns in train data
print("PERCENTAGE OF MISSING VALUES IN COLUMNS")
print(dataset.isnull().mean())
#print the categorical'object type' columns and numerical'float type' columns separatly
categorical, numerical = [], []

for i in dataset.columns:

    if dataset[i].dtype == 'object':
        categorical.append(i)
    else:
        numerical.append(i)

print(categorical)
print(numerical)
print(dataset.shape)
#dataset=dataset.drop(['Date','Cloud9am','Sunshine' , 'Cloud3pm' , 'Evaporation'],axis=1)
dataset=dataset.drop(['Date'],axis=1)
#dataset=dataset.drop(['Date','Temp9am','Temp3pm','Pressure9am'],axis=1)
dataset=dataset.dropna(how='any')
print(dataset.shape)
categorical, numerical = [], []

for i in dataset.columns:

    if dataset[i].dtype == 'object':
        categorical.append(i)
    else:
        numerical.append(i)

print(categorical)
print(numerical)

#separate categorical and numerical columns
#separate categorical and numerical columns
#categorical=dataset[['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']]
#numerical=dataset[[ 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']]


categorical=dataset[['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']]
numerical=dataset[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']]


#categorical=dataset[['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']]
#numerical=dataset[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',]]



#applying label encoder to categorical data
label_encoder = LabelEncoder()
for i in categorical:
    dataset[i] = label_encoder.fit_transform(dataset[i])

print(dataset.info())

plt.figure(figsize=(18,12))
sns.heatmap(dataset.corr(), annot=True)
plt.xticks(rotation=90)
plt.show()

y=dataset['RainTomorrow']
X=dataset.drop(['RainTomorrow'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

scaler = StandardScaler()

scaler.fit(X_train)
X_train_s = pd.DataFrame(scaler.fit_transform(X_train),columns = X_train.columns)

scaler.fit(X_test)
X_test_s = pd.DataFrame(scaler.fit_transform(X_test),columns = X_test.columns)

