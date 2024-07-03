import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from scikeras.wrappers import KerasClassifier

class TSMixerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_features, num_targets, hidden_dim=256, num_mixer_layers=4, dropout=0.2):
        # Initialize the architecture parameters
        self.num_features = num_features
        self.num_targets = num_targets
        self.hidden_dim = hidden_dim
        self.num_mixer_layers = num_mixer_layers
        self.dropout = dropout
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_dim, activation='relu', input_shape=(self.num_features,)))
        for _ in range(self.num_mixer_layers):
            model.add(BatchNormalization())
            model.add(Dense(self.hidden_dim, activation='relu'))
            model.add(Dropout(self.dropout))
        model.add(Dense(self.num_targets, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X, y, epochs=50, batch_size=32, verbose=0):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(np.argmax(y, axis=1), y_pred)

# Load and preprocess data
# Assuming 'train' is your DataFrame and has proper column names
X = train.drop(columns=['target_10_val','era'])
y = pd.get_dummies(train['target_10_val'])  # Assume 'target' is the column to predict

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline with preprocessing and classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KerasClassifier(build_fn=lambda: TSMixerClassifier(num_features=X_train.shape[1], num_targets=y_train.shape[1])))
])

# Fit the model
pipeline.fit(X_train.values, y_train.values)

# Predict and evaluate
y_pred = pipeline.predict(X_test.values)
accuracy = accuracy_score(np.argmax(y_test.values, axis=1), y_pred)
f1 = f1_score(np.argmax(y_test.values, axis=1), y_pred, average='macro')
roc_auc = roc_auc_score(y_test.values, pipeline.named_steps['classifier'].model.predict_proba(X_test.values), multi_class='ovr')
print(f'Accuracy: {accuracy}, F1-Score: {f1}, ROC-AUC: {roc_auc}')