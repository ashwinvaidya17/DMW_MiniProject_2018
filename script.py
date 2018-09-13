import pandas as pd
import numpy as np
from DecisionTreeModel import DecisionTreeModel
from SVC import SVC
from NN import NN

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

dataset = pd.read_csv("Churn_Modelling.csv", sep=',')

column_selector = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                   'IsActiveMember', 'EstimatedSalary', 'Exited']

dataset = dataset[column_selector]

#remove outlier
def remove_outlier(feature):
    first_q = np.percentile(dataset[feature],25)
    third_q = np.percentile(dataset[feature],75)
    IQR = third_q - first_q
    IQR *= 1.5
    minimum = first_q - IQR
    maximum = third_q + IQR

    mean = dataset[feature].median()

    dataset.loc[dataset[feature] < minimum, feature] = mean
    dataset.loc[dataset[feature] > maximum, feature] = mean

outliers = ["CreditScore", "Age", "NumOfProducts"]

for i in range(len(outliers)):
    remove_outlier(outliers[i])

#Scale data
column_normalize_selector = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
scalar = MinMaxScaler()
dataset[column_normalize_selector] = scalar.fit_transform(dataset[column_normalize_selector])


# convert categorical to one hot
le = LabelEncoder()
dataset['Gender'] = le.fit_transform(dataset['Gender'])
dataset['Geography'] = le.fit_transform(dataset['Geography'])

print(dataset.head())

X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)

# Run three models

model_DecisionTree = DecisionTreeModel(X_train, X_test, y_train, y_test)
model_DecisionTree.test()
model_DecisionTree.visualize()

model_LinearSVC = SVC(X_train, X_test, y_train, y_test)
model_LinearSVC.test()
model_LinearSVC.visualize()

model_NN = NN(X_train, X_test, y_train, y_test)
model_NN.test()
