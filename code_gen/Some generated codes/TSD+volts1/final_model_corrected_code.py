import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PCA
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.ensemble import RandomForestClassifier

def load_data(filepath):
    return pd.read_csv(filepath)

def handle_missing_data(data):
    imputer = SimpleImputer(strategy='mean')
    return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

def normalize_features(data):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

def preprocess_data(filepath):
    data = load_data(filepath)
    data = handle_missing_data(data)
    data = normalize_features(data)
    data = data.dropna()
    train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    return train_data, val_data, test_data

def reduce_data_via_ensembling(data, target, n_estimators=10):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(data, target)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Selecting the top features that contribute to 95% of importance
    cumulative_importance = np.cumsum(importances[indices])
    top_features = np.where(cumulative_importance <= 0.95)[0]
    reduced_data = data.iloc[:, indices[:len(top_features)]]
    return reduced_data

def apply_pca(data, n_components=0.95):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    return pd.DataFrame(data=principal_components)

def historical_volatility(price_series, window_size=30):
    log_returns = np.log(price_series / price_series.shift(1))
    return log_returns.rolling(window=window_size).std() * np.sqrt(252)

def calculate_volatility_features(data):
    data['volatility'] = historical_volatility(data['Close_n_val'])
    return data

def find_optimal_clusters(data, max_k=10):
    scores = {}
    for k in range(2, max_k + 1):
        model = TimeSeriesKMeans(n_clusters=k, metric="dtw", random_state=42)
        clusters = model.fit_predict(data.values)
        score = silhouette_score(data.values, clusters)
        scores[k] = score
    optimal_k = max(scores, key=scores.get)
    return optimal_k

def cluster_data(data):
    optimal_k = find_optimal_clusters(data)
    model = TimeSeriesKMeans(n_clusters=optimal_k, metric="dtw", random_state=42)
    return model.fit_predict(data.values)

def test_granger_causality(data, maxlag):
    return grangercausalitytests(data, maxlag=maxlag, verbose=False)

def find_significant_granger_pairs(data, maxlag):
    significant_pairs = []
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            if i != j:
                result = test_granger_causality(data.iloc[:, [i, j]], maxlag)
                if result[1][0]['ssr_chi2test'][1] < 0.05:  # p-value check
                    significant_pairs.append((i, j))
    return significant_pairs

def train_model(features, targets, num_ensemble_configurations=32):
    classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=num_ensemble_configurations)
    classifier.fit(features, targets)
    return classifier

### Main Execution Pipeline

if __name__ == "__main__":
    train_data, val_data, test_data = preprocess_data('df_train_shuffled.csv')
    train_data = calculate_volatility_features(train_data)
    train_data = reduce_data_via_ensembling(train_data.drop(['target'], axis=1), train_data['target'])
    train_data = apply_pca(train_data)
    
    clusters = cluster_data(train_data)
    significant_pairs = find_significant_granger_pairs(train_data, maxlag=30)
    
    # Training the model
    model = train_model(train_data, train_data['target'])
    
    # Making predictions
    predictions = model.predict(test_data.drop(['target'], axis=1))
    
    # Evaluating the model
    print('Accuracy:', accuracy_score(test_data['target'], predictions))