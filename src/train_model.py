import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
import argparse

# Define paths for project structure
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

for dir_path in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

def load_and_prepare_data(force_reload=False):
    raw_data_path = os.path.join(DATA_DIR, 'feature_engineered_data.csv')
    processed_data_path = os.path.join(DATA_DIR, 'processed_data.joblib')
    
    if not force_reload and os.path.exists(processed_data_path):
        print("Loading preprocessed data...")
        return joblib.load(processed_data_path)
    
    print("Processing raw data...")
    df = pd.read_csv(raw_data_path)
    
    # Identify the target variable (CO2 emissions)
    co2_columns = [col for col in df.columns if 'co2' in col.lower() or 'carbon dioxide' in col.lower()]
    if not co2_columns:
        raise ValueError("No CO2 emissions column found in the dataset")
    target_column = co2_columns[0]
    
    # Select features (excluding unique identifiers and the target)
    exclude_keywords = ['emissions', 'CO2', 'reported direct', 'GHG', 'Facility Id', 'FRS Id', 'Zip Code']
    features = [col for col in df.columns if not any(keyword.lower() in col.lower() for keyword in exclude_keywords) and col != target_column]
    
    X = df[features]
    y = df[target_column]
    
    # Handle categorical variables
    X = pd.get_dummies(X, columns=[col for col in X.select_dtypes(include=['object']).columns])
    
    joblib.dump((X, y), processed_data_path)
    return X, y

def preprocess_data(X, y, force_reprocess=False):
    preprocessed_data_path = os.path.join(DATA_DIR, 'preprocessed_data.joblib')
    
    if not force_reprocess and os.path.exists(preprocessed_data_path):
        print("Loading preprocessed data...")
        return joblib.load(preprocessed_data_path)
    
    print("Preprocessing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))
    X_scaled = pd.DataFrame(X_scaled, columns=X.select_dtypes(include=[np.number]).columns)
    
    X_preprocessed = pd.concat([X_scaled, X.select_dtypes(exclude=[np.number])], axis=1)
    
    # Log transform the target variable
    y = np.log1p(y)
    
    joblib.dump((X_preprocessed, y, scaler), preprocessed_data_path)
    return X_preprocessed, y, scaler

def train_and_evaluate_model(X, y, force_retrain=False):
    model_path = os.path.join(MODEL_DIR, 'best_model.joblib')
    results_path = os.path.join(RESULTS_DIR, 'model_results.joblib')
    
    if not force_retrain and os.path.exists(model_path) and os.path.exists(results_path):
        print("Loading trained model and results...")
        model = joblib.load(model_path)
        X_test, y_test, y_pred, mse, r2 = joblib.load(results_path)
    else:
        print("Training and evaluating model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        joblib.dump(model, model_path)
        joblib.dump((X_test, y_test, y_pred, mse, r2), results_path)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")
    
    return model, X_test, y_test, y_pred

def interpret_model(model, X, y_true, y_pred):
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(np.expm1(y_true), np.expm1(y_pred))
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual CO2 Emissions')
    plt.ylabel('Predicted CO2 Emissions')
    plt.title('Actual vs Predicted CO2 Emissions')
    plt.savefig(os.path.join(RESULTS_DIR, 'actual_vs_predicted.png'))
    plt.close()
    
    feature_importance.head(10).to_csv(os.path.join(RESULTS_DIR, 'top_10_features.csv'), index=False)
    print("Top 10 Important Features saved to CSV in results directory")

def main(args):
    X, y = load_and_prepare_data(args.force_reload)
    X_preprocessed, y, scaler = preprocess_data(X, y, args.force_reprocess)
    model, X_test, y_test, y_pred = train_and_evaluate_model(X_preprocessed, y, args.force_retrain)
    interpret_model(model, X_preprocessed, y_test, y_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CO2 Emissions Prediction Pipeline")
    parser.add_argument("--force_reload", action="store_true", help="Force reload and reprocess raw data")
    parser.add_argument("--force_reprocess", action="store_true", help="Force reprocessing of data")
    parser.add_argument("--force_retrain", action="store_true", help="Force retraining of the model")
    args = parser.parse_args()
    
    main(args)