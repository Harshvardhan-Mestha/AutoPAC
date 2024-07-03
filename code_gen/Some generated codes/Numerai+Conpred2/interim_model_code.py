
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier

# Assuming 'train' DataFrame is already loaded
# Prepare data
features = train.columns.difference(['target'])
X = train[features].to_numpy()
y = train['target'].to_numpy()

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Imputation
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Dimensionality reduction with PCA to adhere to the feature limit
pca = PCA(n_components=100)
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)
X_test = pca.transform(X_test)

# Function to calculate Gini Impurity
def gini_impurity(probabilities):
    return 1 - np.sum(np.square(probabilities))

# Custom function for training TabPFN model considering chunks
def train_tabpfn_models(data, labels, chunks, confidence_threshold):
    models = []
    chunk_size = len(data) // chunks
    for i in range(chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i != (chunks - 1) else len(data)
        # Here, you initialize your TabPFNClassifier
        clf = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
        # Fit your model
        selected_data = data[start:end][:1000]
        selected_labels = labels[start:end][:1000]
        clf.fit(selected_data, selected_labels)
        models.append(clf)
    return models

# Adjusting the number of models for conservative approach
num_models = 2  # Adjust based on chunking strategy if necessary
models = train_tabpfn_models(X_train, y_train, num_models, confidence_threshold=0.5)

# Utility function to decide predictions based on confidence level
def predict_conservatively(models, data, confidence_threshold):
    for model in models:
        y_pred_proba = model.predict_proba(data)
        confidence = gini_impurity(y_pred_proba)
        if confidence <= confidence_threshold:
            return np.argmax(y_pred_proba, axis=1), confidence
    return None, None  # Return None if no model is confident enough

# Apply predictions conservatively
predictions, confidences = [], []
for x in X_val:
    pred, conf = predict_conservatively(models, x.reshape(1, -1), confidence_threshold=0.5)
    predictions.append(pred)
    confidences.append(conf)

# Evaluate validation results
valid_preds = [p[0] if p is not None else None for p in predictions]  # Flatten list and handle Nones
accuracy = accuracy_score(y_val[~np.isnan(np.array(valid_preds))], np.array(valid_preds).astype(float)[~np.isnan(np.array(valid_preds))])
print("Validation Accuracy:", accuracy)

# Final evaluation on the test data using the last model, modified to use correct method signature if necessary
# Note: Assuming .predict now instead of .predict_proba due to missing context on implementation details
final_predictions = models[-1].predict(X_test)
final_accuracy = accuracy_score(y_test, final_predictions)
print("Final Test Set Accuracy:", final_accuracy)

