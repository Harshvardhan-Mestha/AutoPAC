import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit

# Assume data loaded into DataFrame `train`
def preprocess_data(data):
    # Handle missing values
    data.fillna(data.mean(), inplace=True)  # Impute with mean for simplicity
    
    # Normalize the features
    scaler = StandardScaler()
    features = data.drop(['target_10_val', 'target_5_val','era'], axis=1)

    # Check if there are any features left after dropping the target columns
    if features.shape[1] > 0:
        features_scaled = scaler.fit_transform(features)
        data[features.columns] = features_scaled
    
    return data

train = preprocess_data(train)
print

def engineer_features(data):
    # Engineering lag features as an example
    window_sizes = [1, 3, 5, 10]  # You can choose different window sizes
    for window in window_sizes:
        data[f'Close_n_rolling_mean_{window}'] = data['Close_n_val'].rolling(window=window).mean()
        data[f'Close_n_rolling_std_{window}'] = data['Close_n_val'].rolling(window=window).std()
    
    data.dropna(inplace=True)
    return data

train = engineer_features(train)

from pymfe import mfe


def extract_metafeatures(data, target):
    if data.shape[0] > 0 and target.shape[0] > 0:
        mfe_obj = mfe.MFE(groups=["general", "statistical", "info-theory"])
        mfe_obj.fit(data.values, target.values)
        ft = mfe_obj.extract()
        metafeatures = dict(zip(ft[0], ft[1]))
    else:
        metafeatures = {}
        print("Input data or target variable is empty. Returning an empty dictionary.")
    return metafeatures

# Assuming 'target_10_val' and 'target_5_val' are your target variables
metafeatures_10 = extract_metafeatures(train.drop(['target_10_val', 'target_5_val','era'], axis=1), train['target_10_val'])
metafeatures_5 = extract_metafeatures(train.drop(['target_10_val', 'target_5_val','era'], axis=1), train['target_5_val'])
print(metafeatures_10)
print(metafeatures_5)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

def train_and_evaluate(data, target):
    timesplit = TimeSeriesSplit(n_splits=5)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42),
        'MLP': MLPClassifier(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, data, target, cv=timesplit, scoring='accuracy')
        results[name] = scores.mean()
    return results

# Split features and target
features = train.drop(['target_10_val', 'target_5_val','era'], axis=1)
target_10 = train['target_10_val']
target_5 = train['target_5_val']

results_10 = train_and_evaluate(features, target_10)
results_5 = train_and_evaluate(features, target_5)
print(results_10)
print(results_5)


### Step 5: Metafeature Analysis

def metafeature_correlation(metafeatures, results):
    # Example: Correlate model performance with number of features
    performance_correlation = {}
    for model, accuracy in results.items():
        # Simple correlation computation (placeholder, implement correctly)
        performance_correlation[model] = metafeatures['n_features'] * accuracy  # Simplified
    
    return performance_correlation

correlations_10 = metafeature_correlation(metafeatures_10, results_10)
correlations_5 = metafeature_correlation(metafeatures_5, results_5)
print(correlations_10)
print(correlations_5)


### Step 6: Reporting


def report_results(results_10, results_5, metafeatures_10, metafeatures_5, correlations_10, correlations_5):
    print("Model performances for target_10_val:", results_10)
    print("Important metafeatures for target_10_val:", metafeatures_10)
    print("Correlation between metafeatures and model performance for target_10_val:", correlations_10)
    print("\nModel performances for target_5_val:", results_5)
    print("Important metafeatures for target_5_val:", metafeatures_5)
    print("Correlation between metafeatures and model performance for target_5_val:", correlations_5)

report_results(results_10, results_5, metafeatures_10, metafeatures_5, correlations_10, correlations_5)