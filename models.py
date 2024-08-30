import warnings
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import joblib  # For saving models

def warn(*args, **kwargs):
    pass

warnings.warn = warn

# Load and preprocess the data
filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv"
df = pd.read_csv(filepath)

df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)

df_sydney_processed.drop('Date', axis=1, inplace=True)
df_sydney_processed = df_sydney_processed.astype(float)
features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=10)

# 1. Linear Regression Model
LinearReg = LinearRegression()
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test_scaled)
LinearReg.fit(x_train_scaled, y_train)
linear_predictions = LinearReg.predict(x_test_scaled)

# Save the linear model
joblib.dump(LinearReg, 'linear_model.pkl')

# 2. KNN Model
KNN = KNeighborsClassifier(n_neighbors=4)
KNN.fit(x_train, y_train)
knn_predictions = KNN.predict(x_test)

# Save the KNN model
joblib.dump(KNN, 'knn_model.pkl')

# 3. Decision Tree Model
Tree = DecisionTreeClassifier()
Tree.fit(x_train, y_train)
tree_predictions = Tree.predict(x_test)

# Save the Decision Tree model
joblib.dump(Tree, 'tree_model.pkl')

# 4. Logistic Regression Model
x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=1)
LR = LogisticRegression(solver='liblinear')
LR.fit(x_train, y_train)
lr_predictions = LR.predict(x_test)
lr_predict_proba = LR.predict_proba(x_test)

# Save the Logistic Regression model
joblib.dump(LR, 'logistic_model.pkl')

# 5. SVM Model
SVM = SVC(class_weight='balanced', probability=True)
SVM.fit(x_train, y_train)
svm_predictions = SVM.predict(x_test)

# Save the SVM model
joblib.dump(SVM, 'svm_model.pkl')

# Save predictions for evaluation
np.savez('predictions.npz', 
         linear=linear_predictions, 
         knn=knn_predictions, 
         tree=tree_predictions, 
         lr=lr_predictions, 
         lr_proba=lr_predict_proba,
         svm=svm_predictions)

# Save test data for evaluation
np.savez('test_data.npz', x_test=x_test, y_test=y_test)
