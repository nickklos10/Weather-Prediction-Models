import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, log_loss

# Load predictions and test data
predictions = np.load('predictions.npz')
test_data = np.load('test_data.npz')
y_test = test_data['y_test']

# 1. Evaluate Linear Regression
linear_predictions = predictions['linear']
LinearRegression_MAE = mean_absolute_error(y_test, linear_predictions)
LinearRegression_MSE = mean_squared_error(y_test, linear_predictions)
LinearRegression_R2 = r2_score(y_test, linear_predictions)

# 2. Evaluate KNN
knn_predictions = predictions['knn']
KNN_Accuracy_Score = accuracy_score(y_test, knn_predictions)
KNN_JaccardIndex = jaccard_score(y_test, knn_predictions)
KNN_F1_Score = f1_score(y_test, knn_predictions)

# 3. Evaluate Decision Tree
tree_predictions = predictions['tree']
Tree_Accuracy_Score = accuracy_score(y_test, tree_predictions)
Tree_JaccardIndex = jaccard_score(y_test, tree_predictions)
Tree_F1_Score = f1_score(y_test, tree_predictions)

# 4. Evaluate Logistic Regression
lr_predictions = predictions['lr']
lr_predict_proba = predictions['lr_proba']
LR_Accuracy_Score = accuracy_score(y_test, lr_predictions)
LR_JaccardIndex = jaccard_score(y_test, lr_predictions)
LR_F1_Score = f1_score(y_test, lr_predictions)
LR_Log_Loss = log_loss(y_test, lr_predict_proba)

# 5. Evaluate SVM
svm_predictions = predictions['svm']
SVM_Accuracy_Score = accuracy_score(y_test, svm_predictions)
SVM_JaccardIndex = jaccard_score(y_test, svm_predictions)
SVM_F1_Score = f1_score(y_test, svm_predictions)

# Create a report
report_data = {
    'Model': ['Linear Regression', 'Logistic Regression', 'KNN', 'Decision Tree', 'SVM'],
    'Accuracy': [None, LR_Accuracy_Score, KNN_Accuracy_Score, Tree_Accuracy_Score, SVM_Accuracy_Score],
    'Jaccard Index': [None, LR_JaccardIndex, KNN_JaccardIndex, Tree_JaccardIndex, SVM_JaccardIndex],
    'F1 Score': [None, LR_F1_Score, KNN_F1_Score, Tree_F1_Score, SVM_F1_Score],
    'Log Loss': [None, LR_Log_Loss, None, None, None]  # Log Loss is only for Logistic Regression
}

Report = pd.DataFrame(report_data)
print(Report)
