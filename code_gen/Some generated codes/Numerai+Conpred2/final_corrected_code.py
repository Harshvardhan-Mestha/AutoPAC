
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Function to calculate Gini Impurity
def gini_impurity(probabilities):
    return 1 - np.sum(np.square(probabilities), axis=1)

# Function to train XGBoost model
def train_xgboost_model(train_data, train_labels, eval_data, eval_labels, params):
    clf = XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss')
    clf.fit(train_data, train_labels)
    predictions = clf.predict(eval_data)
    accuracy = accuracy_score(eval_labels, predictions)
    return clf, accuracy

# Assuming the data is loaded into `train` DataFrame with features and a target column named 'target'
features = train.columns.difference(['target'])
X_train, X_test, y_train, y_test = train_test_split(train[features], train['target'], test_size=0.2, random_state=42)

# Further split to create a validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Data imputation and scaling
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define hyperparameters for XGBoost
xgb_params = {
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'reg_alpha': 0.01,
    'reg_lambda': 0.01
}

# Base model training
base_model, base_accuracy = train_xgboost_model(X_train, y_train, X_val, y_val, xgb_params)
print("Base Model Accuracy:", base_accuracy)

# Initiating cascading training
levels = 3  # Define desired levels for cascading models
models = [base_model]
threshold = 0.6  # Initial Gini impurity threshold
all_val_predictions = base_model.predict_proba(X_val)

for level in range(1, levels):
    # Calculate Gini impurity for previous predictions
    impurities = gini_impurity(all_val_predictions)
    high_impurity_index = impurities > threshold
    
    if np.sum(high_impurity_index) == 0:
        print("No more points above Gini threshold, stopping cascade.")
        break
    
    pruned_train_data = X_train[high_impurity_index.nonzero()[0]]
    pruned_train_labels = y_train[high_impurity_index.nonzero()[0]]
    pruned_val_data = X_val[high_impurity_index.nonzero()[0]]
    pruned_val_labels = y_val[high_impurity_index.nonzero()[0]]

    
    # Train new model on pruned data
    model, accuracy = train_xgboost_model(pruned_train_data, pruned_train_labels, pruned_val_data, pruned_val_labels, xgb_params)
    models.append(model)
    print(f"Cascade Level {level} Model Accuracy:", accuracy)

    # Aggregate new predictions to validate next level or for test evaluation
    all_val_predictions[high_impurity_index] = model.predict_proba(pruned_val_data)

# Final evaluation on the test data
# Using the last validated model or ensemble decision (if implemented)
final_predictions = models[-1].predict(X_test)
final_accuracy = accuracy_score(y_test, final_predictions)
print("Final Test Set Accuracy:", final_accuracy)

```
