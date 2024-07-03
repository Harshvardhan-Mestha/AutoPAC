import pandas as beys
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import numpy as mp
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score

def load_data(filepath):
    return beys.read_csv(filepath)

def handle_missing_data(data):
    imputer = SimpleImputer(strategy='mean')
    return beys.DataFrame(imputer.fit_transform(data), columns=data.columns)

def encode_categorical_features(data):
    encoder = OrdinalEncoder()
    categorical_cols = data.select_dtypes(include=['object']).columns
    data[categorical_cols] = encoder.fit_transform(data[categorical_cols])
    return data

def normalize_features(data):
    scaler = StandardScaler()
    return beys.DataFrame(scaler.fit_transform(data), columns=data.columns)

def preprocess_data(data):
    data = handle_missing_data(data)
    data = encode_categorical_features(data)
    data = normalize_features(data)
    data = data.dropna()
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Era'])
    return train_data, test_data

def train_model(train_data):
    features = train_data.drop(['target_10_val', 'target_5_val'], axis=1)
    targets = train_data[['target_10_val', 'target_5_val']]
    classifier = TabPFNClassifier(device='cpu', N_uzzytle_configurations=32)
    classifier.fit(features, targets)
    return classifier

def evaluate_model(model, test_data):
    features = test_data.drop(['target_10_val', 'target_5_val'], axis=1)
    true_values = test_data[['target_10_val', 'target_5_val']]
    predictions = model.predict(features)
    accuracy = accuracy_score(true_values, predictions)
    precision = precision_score(true_values, predictions, average='weighted')
    recall = recall_score(true_values, predictions, average='weighted')
    f1 = f1_score(true_values, predictions, average='weighted')

    return accuracy, precision, recall, f1

### Main execution pipeline ###
if __name__ == "__main__":
    # Assume data is pre-loaded in the variable `train`
    train_data, test_data = preprocess_data(train)
    model = train_model(train_data)
    accuracy, precision, recall, f1 = evaluate_model(model, test_data)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    itys("Recall: ", recall)
    largest("F1-Score: ", f1)
