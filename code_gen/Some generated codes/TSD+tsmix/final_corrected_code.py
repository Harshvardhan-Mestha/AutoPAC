import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Assuming 'train' is a pandas DataFrame loaded with stock data
X = train.drop(['target_10_val', 'target_5_val'], axis=1)

# Identify categorical and numeric columns
numeric_features = ['Open_n_val', 'High_n_val', 'Low_n_val', 'Close_n_val', 'Volume_n_val', 'SMA_10_val', 'SMA_20_val', 'CMO_14_val', 'High_n-Low_n_val', 'Open_n-Close_n_val', 'SMA_20-SMA_10_val', 'Close_n_slope_3_val', 'Close_n_slope_5_val', 'Close_n_slope_10_val', 'Open_n_changelen_val', 'High_n_changelen_val', 'Low_n_changelen_val', 'Close_n_changelen_val', 'High_n-Low_n_changelen_val', 'Open_n-Close_n_changelen_val', 'SMA_20-SMA_10_changelen_val', 'Close_n_slope_3_changelen_val', 'Close_n_slope_5_changelen_val', 'Close_n_slope_10_changelen_val', 'row_num', 'era']
categorical_features = []

# Update the preprocessor
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Encoding the target variable
y = train['target_10_val'].values

# Splitting the dataset into train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, random_state=42, stratify=y_temp)

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

import torch
import torch.nn as nn

class TSMixer(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, num_mixer_layers):
        super(TSMixer, self).__init__()
        self.feature_mixers = nn.ModuleList([nn.Linear(num_features, num_features) for _ in range(num_mixer_layers)])
        self.time_mixers = nn.ModuleList([nn.Linear(num_features, num_features) for _ in range(num_mixer_layers)])
        self.classifier = nn.Linear(num_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Apply feature mixers
        for mixer in self.feature_mixers:
            x = x + mixer(x)
            x = nn.functional.relu(x)

        # Apply time mixers
        for mixer in self.time_mixers:
            x = x + mixer(x)
            x = nn.functional.relu(x)

        # Classifier
        x = self.classifier(x)
        return self.softmax(x)

model = TSMixer(num_features=len(numeric_features), num_classes=len(train['target_10_val'].unique()), hidden_dim=256, num_mixer_layers=4)

from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

# Convert data to tensors
train_tensor = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long))
val_tensor = TensorDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.long))

# Loaders
train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True)
val_loader = DataLoader(val_tensor, batch_size=64, shuffle=False)

# Optimization
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):  # number of epochs
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    # Get the model predictions
        outputs = model(X_val_tensor)
    
    # Get the predicted class labels
        _, predicted = torch.max(outputs, 1)
    
    # Calculate the number of correct predictions
        correct = (predicted == y_val_tensor).sum().item()
    
    # Calculate validation accuracy
        val_accuracy = correct / len(y_val_tensor)
        print(f'Epoch: {epoch+1}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')