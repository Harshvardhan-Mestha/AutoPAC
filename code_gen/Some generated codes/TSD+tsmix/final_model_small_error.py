import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load the stock data
train = pd.read_csv('path_to_your_data.csv')

# Drop columns that may not be necessary or are non-numeric
X = train.drop(['target_10_val', 'target_5_val', 'row_num', 'era'], axis=1)

# Check that the number of features meets the requirement (<100)
if len(X.columns) > 100:
    raise ValueError("Number of features exceeds 100, which is not supported by TabPFNClassifier.")

# Ensure no categorical features are included, only numerical data is processed
categorical_features_check = X.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_features_check:
    raise TypeError(f"Non-numerical columns found: {categorical_features_check}. Only numerical data is supported.")

# Apply preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler())                  # Standardize data
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, X.columns)
])

# Encode the target variable (assuming binary classification)
y = train['target_10_val']
if y.nunique() > 10:
    raise ValueError("Number of classes exceeds 10, which is not supported by TabPFNClassifier.")

# Split the dataset into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Verify that the training dataset has less than 10,000 datapoints
if X_train.shape[0] > 10000:
    raise ValueError("Training set size exceeds 10,000 datapoints, which is not supported by TabPFNClassifier.")

# Preprocess data
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

# Incorporate TabPFNClassifier
from tabpfn import TabPFNClassifier

# Initialize the TabPFN classifier
classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=5)  # Reduced ensemble configurations due to dataset size

# Since TabPFN does in-context learning, it does not need a fit step
# Directly use predict method which internally handles fit and predict

# Predict on the validation set and calculate accuracy
y_val_preds = classifier.predict(X_val)
val_accuracy = accuracy_accuracy(y_val, y_val_preds)
print(f'Validation Accuracy: {val_accuracy}')

# Optionally, evaluate on the test set
y_test_preds = classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_all)
print(f'Test Accuracy: {test_accuracy}')