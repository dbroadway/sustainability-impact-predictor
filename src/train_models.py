import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import os
import matplotlib.pyplot as plt

# Load the feature-engineered data
data_path = os.path.join('..', 'data', 'feature_engineered_data.csv')
df = pd.read_csv(data_path)

# Inspect NaN values
print("NaN values in the dataset:")
print(df.isna().sum())

# Prepare the features and target
X = df.drop(['CO2 emissions (non-biogenic) '], axis=1)
y = df['CO2 emissions (non-biogenic) ']

# Handle NaN values in the target variable
print(f"\nRows with NaN in target variable: {y.isna().sum()}")
print(f"Total rows before removing NaN: {len(y)}")
X = X[y.notna()]
y = y.dropna()
print(f"Total rows after removing NaN: {len(y)}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create or load preprocessor
preprocessor_path = os.path.join('..', 'models', 'preprocessor.joblib')
if os.path.exists(preprocessor_path):
    print("Loading existing preprocessor...")
    preprocessor = joblib.load(preprocessor_path)
else:
    print("Creating new preprocessor...")
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit the preprocessor and save it
    preprocessor.fit(X_train)
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")

# Preprocess the data
X_train_preprocessed = preprocessor.transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Function to create and evaluate a model
def create_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"{model_name} Cross-Validation Scores: {cv_scores}")
    print(f"{model_name} Mean CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
    
    # Predictions and evaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    print(f"\n{model_name} Performance:")
    print(f"Train R2: {r2_score(y_train, y_pred_train):.4f}")
    print(f"Test R2: {r2_score(y_test, y_pred_test):.4f}")
    print(f"Train MAE: {mean_absolute_error(y_train, y_pred_train):.2f}")
    print(f"Test MAE: {mean_absolute_error(y_test, y_pred_test):.2f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Get feature names (handling both DataFrames and numpy arrays)
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns
        else:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title(f"Top 20 Feature Importances - {model_name}")
        plt.bar(range(20), importances[indices][:20])
        plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=90)
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs('../models', exist_ok=True)
        
        # Save the plot
        plt.savefig(f"../models/{model_name.replace(' ', '_').lower()}_feature_importance.png")
        plt.close()
        
        # Print top 20 feature importances
        print(f"\nTop 20 Feature Importances for {model_name}:")
        for i in indices[:20]:
            print(f"{feature_names[i]}: {importances[i]:.4f}")
    
    return model

# Hyperparameter tuning
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

gb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

rf_model = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, n_jobs=-1)
gb_model = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=3, n_jobs=-1)

# Train and evaluate models
rf_model = create_and_evaluate_model(rf_model, X_train_preprocessed, X_test_preprocessed, y_train, y_test, "Random Forest")
gb_model = create_and_evaluate_model(gb_model, X_train_preprocessed, X_test_preprocessed, y_train, y_test, "Gradient Boosting")

# Print best parameters
print("\nBest Random Forest Parameters:", rf_model.best_params_)
print("Best Gradient Boosting Parameters:", gb_model.best_params_)

# Save the best model (based on test set performance)
best_model = rf_model if r2_score(y_test, rf_model.predict(X_test_preprocessed)) > r2_score(y_test, gb_model.predict(X_test_preprocessed)) else gb_model
model_path = os.path.join('..', 'models', 'best_model.joblib')
joblib.dump(best_model, model_path)
print(f"\nBest model saved to {model_path}")

# Error analysis
best_predictions = best_model.predict(X_test_preprocessed)
residuals = y_test - best_predictions

plt.figure(figsize=(10, 6))
plt.scatter(best_predictions, residuals)
plt.xlabel("Predicted CO2 Emissions")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig("../models/residual_plot.png")
plt.close()

print("\nError Analysis:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, best_predictions):.2f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, best_predictions)):.2f}")