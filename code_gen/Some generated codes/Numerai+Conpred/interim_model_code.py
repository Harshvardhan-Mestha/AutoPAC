import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tabpfn import TabPFNClassifier

# Loading the Numerai dataset (assuming 'train' variable is already defined)
# Let's assume 'train' is a DataFrame with features and a target column named 'target'

# Step 1: Data Preparation
def prepare_data(df, test_size=0.2, seed=42):
    # Imputing missing values
    imputer = SimpleImputer(strategy='median')
    df_imputed = imputer.fit_transform(df.drop(columns=['target']))
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns[:-1])

    # Feature Scaling
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_imputed)

    # PCA for dimensionality reduction, optional if needed
    pca = PCA(n_components=0.95) # Explains 95% of variance
    features_pca = pca.fit_transform(features_scaled)

    X_train, X_val_test, y_train, y_val_test = train_test_split(
        features_pca, df['target'], test_size=test_size, stratify=df['target'], random_state=seed)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.5, stratify=y_val_test, random_state=seed)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Step 2: Base Model Training and Gini Impurity Calculation
def train_and_evaluate(X_train, X_val, y_train, y_val):
    model = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
    model.fit(X_train, y_train)
    
    # Predict probabilities and classes
    val_predictions, probabilities = model.predict(X_val, return_winning_probability=True)
    
    # Gini impurity of predictions
    def compute_gini_impurity(probabilities):
        return 1 - np.sum(np.square(probabilities), axis=1)
    
    impurities = compute_gini_impurity(probabilities)
    accuracy = accuracy_score(y_val, val_predictions)
    
    return model, accuracy, impurities, probabilities, val_predictions

# Step 3: Cascading Model Training
def cascade_training(X_train, X_val, y_train, y_val, num_levels=3):
    base_model, base_accuracy, base_impurities, base_probabilities, base_predictions = train_and_evaluate(
        X_train, X_val, y_train, y_val)
    print(f"Base Model Accuracy: {base_accuracy}")
    
    models = [base_model]
    accuracies = [base_accuracy]
    thresholds = np.percentile(base_impurities, [75])  # Assuming removing bottom 25% uncertain data
    
    for level in range(1, num_levels):
        keep_indices = base_impurities < thresholds[-1]
        if len(keep_indices) != len(X_train):
          keep_indices = np.repeat(keep_indices, len(X_train) // len(keep_indices) + 1)[:len(X_train)]

        X_train_pruned = X_train[keep_indices]
        y_train_pruned = y_train[keep_indices]
        model, accuracy, impurities, probabilities, predictions = train_and_evaluate(
            X_train_pruned, X_val, y_train_pruned, y_val)
        
        print(f"Level {level} Model Accuracy: {accuracy}")
        thresholds = np.concatenate((thresholds, np.percentile(impurities, [75])))
        models.append(model)
        accuracies.append(accuracy)

    return models, accuracies

# Step 4: Final Evaluation
def final_evaluation(models, X_test, y_test):
    final_model = models[-1]
    final_predictions, _ = final_model.predict(X_test, return_winning_probability=True)
    final_accuracy = accuracy_score(y_test, final_predictions)
    conf_matrix = confusion_matrix(y_test, final_predictions)
    print(f"Final Model Test Accuracy: {final_accuracy}")
    return final_accuracy, conf_matrix

# Main execution flow
X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(train)
models, accuracies = cascade_training(X_train, X_val, y_train, y_val)
final_accuracy, conf_matrix = final_evaluation(models, X_test, y_test)
