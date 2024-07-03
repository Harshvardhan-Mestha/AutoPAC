
```python
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import numpy as np

# Assume 'train' is preloaded DataFrame containing the training data

# Handle missing values
def preprocess_data(data):
    imputer = KNNImputer(n_neighbors=5)
    data_imputed = imputer.fit_transform(data.select_dtypes(include=[np.number]))
    data[data.select_dtypes(include=[np.number]).columns] = data_imputed

    # Feature scaling
    scaler = QuantileTransformer(output_distribution='normal')
    scaled_features = scaler.fit_transform(data_imputed)
    return pd.DataFrame(scaled_features, columns=data.columns)

train, test = train_test_split(train, test_size=0.2, shuffle=False)

train_preprocessed = preprocess_data(train)

### 2. Model Selection and Hyperparameter Optimization


import optuna
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def objective(trial):
    model_name = trial.suggest_categorical('model_name', ['logistic_regression', 'catboost', 'xgboost'])
    
    # Hyperparameters for Logistic Regression
    if model_name == 'logistic_regression':
        solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
        C = trial.suggest_loguniform('C', 1e-3, 10)
        model = LogisticRegression(solver=solver, C=C)
    
    # Hyperparameters for CatBoost
    elif model_name == 'catboost':
        depth = trial.suggest_int('depth', 4, 10)
        iterations = trial.suggest_int('iterations', 50, 300)
        learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
        model = CatBoostClassifier(depth=depth, iterations=iterations, learning_rate=learning_rate, verbose=False)
    
    # Hyperparameters for XGBoost
    elif model_name == 'xgboost':
        max_depth = trial.suggest_int('max_depth', 3, 9)
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
        model = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=0.01, verbosity=0)
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_idx, val_idx in tscv.split(train_preprocessed):
        X_train, X_val = train_preprocessed.iloc[train_idx], train_preprocessed.iloc[val_idx]
        y_train, y_val = train.target.iloc[train_idx], train.target.iloc[val_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)
        scores.append(log_loss(y_val, preds))
    
    return np.mean(scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)

best_model_params = study.best_params


### 3. Evaluation and Ensemble Methods


# Assign the best model from Optuna's study
if study.best_params['model_name'] == 'catboost':
    best_model = CatBoostClassifier(**{k: study.best_params[k] for k in study.best_params if k != 'model_name'})
elif study.best_params['model_name'] == 'xgboost':
    best_model = XGBClassifier(**{k: study.best_params[k] for k in study.best_params if k != 'model_name'})
else:
    best_model = LogisticRegression(**{k: study.best_params[k] for k in study.best_params if k != 'model_name'})

best_model.fit(train_preprocessed, train.target) 

# Evaluate with financial metrics here, such as Sharpe ratio (implement correlation and std evaluation with predictions)
predictions = best_model.predict_proba(test)  # Assume test data are already preprocessed

### 4. Final Presentation and Prediction


def submit_predictions(predictions, filename='submission.csv'):
    if isinstance(predictions, np.ndarray):
        # If predictions is a NumPy array with multiple columns
        predictions = predictions[:, 0]  # Select the first column
    elif isinstance(predictions, pd.DataFrame):
        # If predictions is a DataFrame with multiple columns
        predictions = predictions.iloc[:, 0]  # Select the first column
    
    # Assume 'test_data' contains the test data with an 'era' column
    submission = pd.DataFrame({
        'era': test['era'],
        'prediction': predictions
    })
    submission.to_csv(filename, index=False)

submit_predictions(predictions)
```

