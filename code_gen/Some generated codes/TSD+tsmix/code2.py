import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# Assuming data is loaded into a pandas DataFrame called `train`

# Define a function to create the adapted TSMixer model
def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Preprocessing pipeline for numerical and categorical data
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = train.select_dtypes(include=['object', 'bool']).columns

num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder())
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols),
    ('remainder', 'passthrough', remaining_cols)
])

# Splitting the data
X = train.drop(['target'], axis=1)
y = train['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Creating the adaptation of TSMixer model in a pipeline
ml_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # Wrapping Keras model with KerasClassifier for compatibility
    ('classifier', KerasClassifier(build_fn=create_model, input_shape=len(numerical_cols) + len(categorical_cols) + len(remaining_cols), num_classes=len(train['target'].unique())))
])

# Using GridSearchCV for hyperparameter tuning
param_grid = {
    'classifier__epochs': [10, 20, 30],
    'classifier__batch_size': [16, 32, 64]
}

grid_search = GridSearchCV(ml_model, param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Evaluating the model
y_pred = grid_search.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



##Wrong library import