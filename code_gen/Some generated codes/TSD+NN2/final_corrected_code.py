import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# Load dataset (assuming the data is already in a variable 'train')
columns = ['Open_n_val', 'High_n_val', 'Low_n_val', 'Close_n_val', 'Volume_n_val',
           'SMA_10_val', 'SMA_20_val', 'CMO_14_val', 'High_n-Low_n_val',
           'Open_n-Close_n_val', 'SMA_20-SMA_10_val', 'Close_n_slope_3_val',
           'Close_n_slope_5_val', 'Close_n_slope_10_val', 'Open_n_changelen_val',
           'High_n_changelen_val', 'Low_n_changelen_val', 'Close_n_changelen_val',
           'High_n-Low_n_changelen_val', 'Open_n-Close_n_changelen_val',
           'SMA_20-SMA_10_changelen_val', 'Close_n_slope_3_changelen_val',
           'Close_n_slope_5_changelen_val', 'Close_n_slope_10_changelen_val',
           'row_num', 'era', 'target_10_val', 'target_5_val']

data = train.copy()
data.columns = columns

# Define preprocessing function
def preprocess_data(data, label_column):
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    # Encode categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    onehot_encoder = OneHotEncoder()
    encoded_data = onehot_encoder.fit_transform(data[categorical_cols])
    data = data.drop(columns=categorical_cols)
    data = pd.concat([data, pd.DataFrame(encoded_data.toarray(), index=data.index)], axis=1)
    
    # Split data into features and target
    X = data.drop(columns=[label_column])
    y = data[label_column]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

# Train-test split
X, y = preprocess_data(data, 'target_10_val')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Define and train models
# Example: Gradient Boosting
gb_params = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 500],
    'max_depth': [3, 5]
}
model_gb = GridSearchCV(xgb.XGBClassifier(), gb_params, cv=3)
model_gb.fit(X_train, y_train)

# Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.out(x)

def train_nn_model(X_train, y_train, input_dim, num_classes):
    model = SimpleNN(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # Convert to tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    train_tensor = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True)
    
    # Training loop
    for epoch in range(10):  # Assume 10 epochs for simplicity
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
    return model

# Example Neural network training
num_classes = len(np.unique(y_train))
nn_model = train_nn_model(X_train, y_train, X_train.shape[1], num_classes)

# Model evaluation
def evaluate_model(model, X_val, y_val):
    if isinstance(model, SimpleNN):
        X_val_tensor = torch.from_numpy(X_val).float()
        y_val_tensor = torch.from_numpy(y_val).long()
        with torch.no_grad():
            y_pred_logits = model(X_val_tensor)
            y_pred_proba = nn.functional.softmax(y_pred_logits, dim=1).numpy()
            y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred_proba = model.predict_proba(X_val)
        y_pred = np.argmax(y_pred_proba, axis=1)
    acc = accuracy_score(y_val, y_pred)
    logloss = log_loss(y_val, y_pred_proba)
    return acc, logloss

acc_gb, logloss_gb = evaluate_model(model_gb, X_val, y_val)
print(f"Gradient Boosting - Accuracy: {acc_gb:.4f}, Log loss: {logloss_gb:.4f}")

acc_nn, logloss_nn = evaluate_model(nn_model, X_val, y_val)
print(f"Neural Network - Accuracy: {acc_nn:.4f}, Log loss: {logloss_nn:.4f}")