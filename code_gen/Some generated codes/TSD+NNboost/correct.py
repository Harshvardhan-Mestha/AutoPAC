import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit

def preprocess_data(data):
    # Handling missing values with median imputation
    data = data.fillna(data.median())
    
    # Encoding categorical features if any
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    categorical_cols = categorical_cols.append(pd.Index(['era']))  # Adding 'era' column
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Splitting the data respecting era boundaries
    ts_split = TimeSeriesSplit(n_splits=5)  # Assuming 'era' is a time-like sequential column
    for train_index, test_index in ts_split.split(data):
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]
        yield train_data, test_data

# Assuming 'train' is already loaded
for train_data, test_data in preprocess_data(train):
    print("Processed training data and test data")

def engineer_features(data):
    # Adding rolling features for feature engineering
    window_sizes = [5, 10, 20]
    for window in window_sizes:
        data[f'rolling_mean_{window}'] = data['Close_n_val'].rolling(window=window).mean()
        data[f'rolling_std_{window}'] = data['Close_n_val'].rolling(window=window).std()
    
    return data.dropna()

# Process and enhance each train and test set (function from the preprocessing phase)
for train_data, test_data in preprocess_data(train):
    train_features = engineer_features(train_data)
    test_features = engineer_features(test_data)
    print("Engineered features for training and testing")

from pymfe import mfe

from pymfe import mfe

def extract_metafeatures(data, target_variable):
    # Remove rows with missing target values
    data = data.loc[target_variable.notna(), :]
    target_variable = target_variable.loc[target_variable.notna()]

    # Check if data and target are not empty
    if not data.empty and not target_variable.empty:
        # Check if the number of rows matches
        if data.shape[0] == target_variable.shape[0]:
            mfe_obj = mfe.MFE(groups=["statistical"])
            mfe_obj.fit(data.values, target_variable.values)
            stats_metafeatures = mfe_obj.extract()
        else:
            stats_metafeatures = {}
            print("Number of rows in data and target variable do not match.")
    else:
        stats_metafeatures = {}
        print("Input data or target variable is empty. Returning an empty dictionary.")

    return stats_metafeatures

# Extract metafeatures from training data
metafeatures = extract_metafeatures(train_features, train['target_10_val'])
print("Extracted metafeatures:", metafeatures)

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss

def train_and_evaluate(train_data, test_data):
    # Models setup
    models = {
        "Logistic Regression": LogisticRegression(),
        "XGBoost": XGBClassifier()
    }
    
    # Evaluate each model
    results = {}
    for name, model in models.items():
        model.fit(train_data.drop('target_10_val', axis=1), train_data['target_10_val'])
        predictions = model.predict(test_data.drop('target_10_val', axis=1))
        accuracy = accuracy_score(test_data['target_10_val'], predictions)
        logloss = log_loss(test_data['target_10_val'], model.predict_proba(test_data.drop('target_10_val', axis=1)))
        results[name] = {'accuracy': accuracy, 'log_loss': logloss}
    
    return results

for train_features, test_features in preprocess_data(train):
    evaluation_results = train_and_evaluate(train_features, test_features)
    print("Evaluation results:", evaluation_results)

def report_results(results):
    for model_name, metrics in results.items():
        print(f"Model: {model_name}")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value}")

for evaluation_results in preprocess_data(train):
    report_results(evaluation_results)