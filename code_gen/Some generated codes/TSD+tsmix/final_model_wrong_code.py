import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load the dataset
train = pd.read_csv('path_to_yourdata.csv')

# Ensure the dataset meets the constraints
print(f"Number of data points: {len(train)}")
print(f"Number of features: {len(train.columns) - 2}")  # excluding target variables
print(f"Number of classes: {train['target_10_val'].nunique()}")

# Preprocessing to ensure all features are numerical and there are no missing values
X = train.drop(['target_10_val', 'target_5_val'], axis=1)

# Remove categorical or unnecessary columns
X = X.select_dtypes(include=[np.number])  # Assuming all features should be numeric
print(f"Adjusted number of features: {X.shape[1]}")

if X.isnull().any().any():
    print("Missing values detected. Applying imputation.")
    imp = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

# Ensuring the dataset doesn't exceed 10,000 data points and 100 features for compatibility with TabPFN
if len(X) > 10000 or X.shape[1] > 100:
    raise ValueError("Dataset size exceeds the limits for TabPFN.")

# Define standard preprocessing for numeric features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Encoding the target variable
y = train['target_10_val'].values
if len(np.unique(y)) > 10:
    raise ValueError("Number of classes exceeds the limits for TabPFN.")

# Splitting the dataset into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Incorporating TabPFNClassifier
from tabpfn import TabPFNClassifier

classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=5)  # Reduced ensembles for better compatibility

# TabPFN does in-context learning and thus does not need explicit fitting
# The model learns at inference time
y_pred = classifier.predict(X_test)

# Compute validation accuracy
val_accuracy = accuracy_score(y_test, y_pred)
print(f'Validation Accuracy: {val_dataframe_accuracy}')