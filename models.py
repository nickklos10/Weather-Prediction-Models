import warnings
import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import jaccard_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics


def warn(*args, **kwargs):
    pass


warnings.warn = warn


filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv"
df = pd.read_csv(filepath)
print(df.head())

df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)


# Training data and testing data

df_sydney_processed.drop('Date', axis=1, inplace=True)
df_sydney_processed = df_sydney_processed.astype(float)
features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']
x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=10)

# To check the shapes of the splits
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Create the Linear Regression model
LinearReg = LinearRegression()
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Retrain the model with scaled data
LinearReg.fit(x_train_scaled, y_train)

# Make predictions on the scaled test data
predictions = LinearReg.predict(x_test_scaled)

# Calculate the metrics again
LinearRegression_MAE = mean_absolute_error(y_test, predictions)
LinearRegression_MSE = mean_squared_error(y_test, predictions)
LinearRegression_R2 = r2_score(y_test, predictions)


# Create a dictionary with the metrics
report_data = {
    'Metric': ['MAE', 'MSE', 'R2'],
    'Value': [LinearRegression_MAE, LinearRegression_MSE, LinearRegression_R2]
}
Report = pd.DataFrame(report_data)

print(Report)


# Create the KNN model

KNN = KNeighborsClassifier(n_neighbors=4)
KNN.fit(x_train, y_train)

predictions = KNN.predict(x_test)

KNN_Accuracy_Score = accuracy_score(y_test, predictions)
KNN_JaccardIndex = jaccard_score(y_test, predictions)
KNN_F1_Score = f1_score(y_test, predictions)

print("KNN Accuracy Score:", KNN_Accuracy_Score)
print("KNN Jaccard Index:", KNN_JaccardIndex)
print("KNN F1 Score:", KNN_F1_Score)

# Create the tree model

Tree = DecisionTreeClassifier()
Tree.fit(x_train, y_train)

predictions = Tree.predict(x_test)

Tree_Accuracy_Score = accuracy_score(y_test, predictions)
Tree_JaccardIndex = jaccard_score(y_test, predictions)
Tree_F1_Score = f1_score(y_test, predictions)

print("Tree Accuracy Score:", Tree_Accuracy_Score)
print("Tree Jaccard Index:", Tree_JaccardIndex)
print("Tree F1 Score:", Tree_F1_Score)


# Create the logistic regression model

x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=1)

# Create the Logistic Regression model with solver set to 'liblinear'
LR = LogisticRegression(solver='liblinear')
LR.fit(x_train, y_train)

predictions = LR.predict(x_test)

predict_proba = LR.predict_proba(x_test)

from sklearn.metrics import accuracy_score, jaccard_score, f1_score, log_loss

# Calculate Accuracy Score
LR_Accuracy_Score = accuracy_score(y_test, predictions)

# Calculate Jaccard Index
LR_JaccardIndex = jaccard_score(y_test, predictions)

# Calculate F1 Score
LR_F1_Score = f1_score(y_test, predictions)

# Calculate Log Loss
LR_Log_Loss = log_loss(y_test, predict_proba)

# Optionally, print the values to see the results
print("LR Accuracy Score:", LR_Accuracy_Score)
print("LR Jaccard Index:", LR_JaccardIndex)
print("LR F1 Score:", LR_F1_Score)
print("LR Log Loss:", LR_Log_Loss)

# Create the SVM model
SVM = SVC(class_weight='balanced')

SVM.fit(x_train, y_train)

predictions = SVM.predict(x_test)

SVM_Accuracy_Score = accuracy_score(y_test, predictions)
SVM_JaccardIndex = jaccard_score(y_test, predictions)
SVM_F1_Score = f1_score(y_test, predictions)

print("SVM Accuracy Score:", SVM_Accuracy_Score)
print("SVM Jaccard Index:", SVM_JaccardIndex)
print("SVM F1 Score:", SVM_F1_Score)


# Create a dictionary with the metrics
report_data = {
    'Model': ['Logistic Regression', 'KNN', 'Decision Tree', 'SVM'],
    'Accuracy': [LR_Accuracy_Score, KNN_Accuracy_Score, Tree_Accuracy_Score, SVM_Accuracy_Score],
    'Jaccard Index': [LR_JaccardIndex, KNN_JaccardIndex, Tree_JaccardIndex, SVM_JaccardIndex],
    'F1 Score': [LR_F1_Score, KNN_F1_Score, Tree_F1_Score, SVM_F1_Score],
    'Log Loss': [LR_Log_Loss, None, None, None]  # Log Loss is only for Logistic Regression
}

Report = pd.DataFrame(report_data)
print(Report)
