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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_path):
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Data loaded successfully from {data_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {data_path}: {str(e)}")
        raise

def preprocess_data(df, target_column):
    logging.info("Starting data preprocessing")
    X = df.drop([target_column], axis=1)
    y = df[target_column]

    # Handle NaN values in the target variable
    nan_count = y.isna().sum()
    logging.info(f"Rows with NaN in target variable: {nan_count}")
    logging.info(f"Total rows before removing NaN: {len(y)}")
    
    X = X[y.notna()]
    y = y.dropna()
    
    logging.info(f"Total rows after removing NaN: {len(y)}")
    return X, y

def create_preprocessor(X_train, preprocessor_path):
    if os.path.exists(preprocessor_path):
        logging.info("Loading existing preprocessor...")
        return joblib.load(preprocessor_path)
    
    logging.info("Creating new preprocessor...")
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

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

    preprocessor.fit(X_train)
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    logging.info(f"Preprocessor saved to {preprocessor_path}")
    return preprocessor

def create_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    logging.info(f"Training and evaluating {model_name}")
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    logging.info(f"{model_name} Cross-Validation Scores: {cv_scores}")
    logging.info(f"{model_name} Mean CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
    
    # Predictions and evaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    logging.info(f"\n{model_name} Performance:")
    logging.info(f"Train R2: {r2_score(y_train, y_pred_train):.4f}")
    logging.info(f"Test R2: {r2_score(y_test, y_pred_test):.4f}")
    logging.info(f"Train MAE: {mean_absolute_error(y_train, y_pred_train):.2f}")
    logging.info(f"Test MAE: {mean_absolute_error(y_test, y_pred_test):.2f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        logging.info(f"Extracting feature importances for {model_name}")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns
        else:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Create a DataFrame with feature importances
        importance_df = pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': importances[indices]
        })
        
        # Save to CSV
        csv_path = f"../models/{model_name.replace(' ', '_').lower()}_feature_importance.csv"
        importance_df.to_csv(csv_path, index=False)
        logging.info(f"Feature importances saved to {csv_path}")
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title(f"Top 20 Feature Importances - {model_name}")
        plt.bar(range(20), importances[indices][:20])
        plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=90)
        plt.tight_layout()
        
        # Save the plot
        plot_path = f"../models/{model_name.replace(' ', '_').lower()}_feature_importance.png"
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Feature importance plot saved to {plot_path}")
    
    return model

def main():
    # Load the feature-engineered data
    data_path = os.path.join('..', 'data', 'feature_engineered_data.csv')
    df = load_data(data_path)

    # Prepare the features and target
    X, y = preprocess_data(df, 'CO2 emissions (non-biogenic) ')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create or load preprocessor
    preprocessor_path = os.path.join('..', 'models', 'preprocessor.joblib')
    preprocessor = create_preprocessor(X_train, preprocessor_path)

    # Preprocess the data
    X_train_preprocessed = preprocessor.transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

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
    logging.info(f"\nBest Random Forest Parameters: {rf_model.best_params_}")
    logging.info(f"Best Gradient Boosting Parameters: {gb_model.best_params_}")

    # Save the best model (based on test set performance)
    best_model = rf_model if r2_score(y_test, rf_model.predict(X_test_preprocessed)) > r2_score(y_test, gb_model.predict(X_test_preprocessed)) else gb_model
    model_path = os.path.join('..', 'models', 'best_model.joblib')
    joblib.dump(best_model, model_path)
    logging.info(f"\nBest model saved to {model_path}")

    # Error analysis
    best_predictions = best_model.predict(X_test_preprocessed)
    residuals = y_test - best_predictions

    plt.figure(figsize=(10, 6))
    plt.scatter(best_predictions, residuals)
    plt.xlabel("Predicted CO2 Emissions")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    residual_plot_path = "../models/residual_plot.png"
    plt.savefig(residual_plot_path)
    plt.close()
    logging.info(f"Residual plot saved to {residual_plot_path}")

    logging.info("\nError Analysis:")
    logging.info(f"Mean Absolute Error: {mean_absolute_error(y_test, best_predictions):.2f}")
    logging.info(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, best_predictions)):.2f}")

if __name__ == "__main__":
    main()