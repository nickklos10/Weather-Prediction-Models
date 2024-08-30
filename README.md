# Weather-Prediction-Models

This project implements and compares several machine learning models to predict whether it will rain tomorrow based on historical weather data. The models used include Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, and Support Vector Machine (SVM). The project evaluates each model's performance using various metrics such as Accuracy, Jaccard Index, F1 Score, and Log Loss.

## Features
- **Data Preprocessing:** Handling of categorical variables and missing data.
- **Model Training:** Implementation of Logistic Regression, KNN, Decision Tree, and SVM models.
- **Model Evaluation:** Evaluation of models using metrics such as Accuracy, Jaccard Index, F1 Score, and Log Loss (specific to Logistic Regression).
- **Comparison:** Comparison of model performances in a tabular format.

## Technologies Used
- **Python**: Core programming language for model implementation.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-Learn**: For implementing machine learning models and evaluation metrics.
- **NumPy**: For numerical operations.
- **Imbalanced-learn**: For handling imbalanced datasets using techniques like SMOTE.

## Project Structure
- `weather_data.csv`: The dataset used for training and testing models.
- `models.py`: Script containing the implementation of all machine learning models.
- `evaluation.py`: Script for evaluating and comparing the performance of models.
- `requirements.txt`: File for installing necessary libraries.
- `README.md`: This file.

## How to Run
1. **Clone the repository:**
   ```
   git clone https://github.com/yourusername/Weather-Prediction-Models.git
   cd Weather-Prediction-Models
   ```

2. **Install the required libraries:**
   ```
   pip install -r requirements.txt
   ```
3. **Run the model training and evaluation:**
   ```
   python models.py
   python evaluation.py
   ```

## Results

The project provides a comparison of different models' performance based on the Accuracy, Jaccard Index, F1 Score, and Log Loss (specific to Logistic Regression). The report is generated in a tabular format for easy comparison.


