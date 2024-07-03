import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

# Assume the DataFrame 'train' is pre-loaded with historical stock data
# train = pd.read_csv('stock_data.csv') # This line is commented since data is assumed to be pre-loaded

# Specify the numeric and categorical features
numeric_features = ['Open_n_val', 'High_n_val', 'Low_n_val', 'Close_n_val', 'Volume_n_val', 'SMA_10_val', 'SMA_20_val', 'CMO_14_val', 'High_n-Low_n_val', 'Open_n-Close_n_val', 'SMA_20-SMA_10_val', 'Close_n_slope_3_val', 'Close_n_slope_5_val', 'Close_n_slope_10_val', 'Open_n_changelen_val', 'High_n_changelen_val', 'Low_n_changelen_val', 'Close_n_changelen_val', 'High_n-Low_n_changelen_val', 'Open_n-Close_n_changelen_val', 'SMA_20-SMA_10_changelen_val', 'Close_n_slope_3_changelen_val', 'Close_n_slope_5_changelen_val', 'Close_n_slope_10_changelen_val', 'row_num']
categorical_features = ['era']

# Assuming 'target_10_val', 'target_5_val', and '0.75' are your target variables
target_variables = ['target_10_val', 'target_5_val']

# Create a DataPreprocessor class
class DataPreprocessor:
    def __init__(self):
        self.numeric_transformer = StandardScaler()
        self.categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, numeric_features),
                ('cat', self.categorical_transformer, categorical_features)
            ])
        
    def fit_transform(self, X):
        return self.preprocessor.fit_transform(X)

    def transform(self, X):
        return self.preprocessor.transform(X)

preprocessor = DataPreprocessor()
X = preprocessor.fit_transform(train.drop(target_variables, axis=1))
y = train[target_variables]


### 3. Train/Test Split respecting temporal nature

# Reset the index of the DataFrame
train = train.reset_index(drop=True)

# Split the data
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index].toarray(), X[test_index].toarray()
    y_train, y_test = y[train_index].toarray(), y[test_index].toarray()

    ### 4. Model Training and Evaluation Function
    def train_model(model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_prob)
        f1score = f1_score(y_test, y_pred, average='weighted')
        print(f"Accuracy: {accuracy}, Log Loss: {loss}, F1-Score: {f1score}")
        return model, accuracy, loss, f1score

    # Rest of the code...

# Set up a neural network model
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

models = {
    "Neural Network": nn_model,
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier()
}

results = {}
for name, model in models.items():
    print(f"Training {name}")
    trained_model, acc, loss, f1 = train_model(model, X_train, X_test, y_train, y_test)
    results[name] = {"Model": trained_model, "Accuracy": acc, "Log Loss": loss, "F1 Score": f1}


### 5. Reporting Results


# Display the results as DataFrame for better visualization
results_df = pd.DataFrame(results).transpose()
print(results_df[['Accuracy', 'Log Loss', 'F1 Score']])