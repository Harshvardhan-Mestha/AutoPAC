import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

# Assuming 'train' is a pandas DataFrame loaded with stock data

X = train.drop(['target_10_val', 'target_5_val'], axis=1)
y = train['target_10_val'].values

# Identify numeric columns. Assuming all features except 'row_num' and 'era' are numerical as described.
numeric_features = [col for col in X.columns if col not in ('row_num', 'era')]

# Preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=24, svd_solver='covariance_eigh')

)  # Ensuring the number of features is < 100
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Splitting the dataset into train and test sets ensuring the data point limits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.18, stratify=y_train)

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

# Incorporating TabPFNClassifier
from tabpfn import TabPFNClassifier

# Initialize the TabPFN classifier
classifiers = [TabPFNClassifier(device='cpu') for i in range(5)]
ensemble_clf = VotingClassifier(estimators=[(f'clf{i}', clf) for i, clf in enumerate(classifiers)], voting='soft')

# Fit the model to the training data
ensemble_clf.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = ensemble_clf.predict(X_val)

# Calculate validation accuracy
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy}')

# Optional: Predict on the test set to evaluate the model
y_test_pred = ensemble_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy}')