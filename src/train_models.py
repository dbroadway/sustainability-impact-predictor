import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import matplotlib.pyplot as plt
import logging
import traceback
from scipy import stats
import sys
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, RegressorMixin, clone
import scipy.sparse


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Error handling
def exception_handler(exception_type, exception, traceback):
    print(f"{exception_type.__name__}: {exception}")

sys.excepthook = exception_handler

# Range-Based Regressor
class RangeBasedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator=GradientBoostingRegressor()):
        self.base_estimator = base_estimator
        self.models = {}
        self.ranges = None

    def fit(self, X, y):
        # Define ranges
        self.ranges = [
            (y.min(), np.percentile(y, 25)),
            (np.percentile(y, 25), np.percentile(y, 50)),
            (np.percentile(y, 50), np.percentile(y, 75)),
            (np.percentile(y, 75), y.max())
        ]

        # Train a model for each range
        for i, (lower, upper) in enumerate(self.ranges):
            mask = (y >= lower) & (y < upper)
            model = clone(self.base_estimator)
            self.models[i] = model.fit(X[mask], y[mask])

        return self

    def predict(self, X):
        predictions = np.zeros(X.shape[0])

        # Use the first feature (assuming it's 'Total reported direct emissions') to determine which model to use
        if scipy.sparse.issparse(X):
            emissions = X.tocsr()[:, 0].toarray().flatten()
        else:
            emissions = X[:, 0]

        for i, (lower, upper) in enumerate(self.ranges):
            mask = (emissions >= lower) & (emissions < upper)
            if np.any(mask):
                predictions[mask] = self.models[i].predict(X[mask])

        return predictions

# Weighted MAPE
def weighted_mape(y_true, y_pred, weights=None):
    if weights is None:
        weights = y_true / y_true.sum()
    return np.average(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8)), weights=weights) * 100

# Gradient Boosting with Prediction Intervals
class GradientBoostingWithPI(GradientBoostingRegressor):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0, 
                 max_features=None, random_state=None, max_leaf_nodes=None, 
                 warm_start=False, validation_fraction=0.1, 
                 n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0):
        super().__init__(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            subsample=subsample, max_features=max_features, random_state=random_state,
            max_leaf_nodes=max_leaf_nodes, warm_start=warm_start,
            validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change,
            tol=tol, ccp_alpha=ccp_alpha
        )

    def fit(self, X, y):
        super().fit(X, y)
        self.y_train_mean = y.mean()
        self.y_train_std = y.std()
        return self

    def predict_with_pi(self, X, alpha=0.05):
        y_pred = self.predict(X)

        predictions = []
        for estimator in self.estimators_:
            predictions.append(estimator[0].predict(X))

        lower = np.percentile(predictions, alpha/2 * 100, axis=0)
        upper = np.percentile(predictions, (1 - alpha/2) * 100, axis=0)

        return y_pred, lower, upper

def load_data(data_path):
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Data loaded successfully from {data_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {data_path}: {str(e)}")
        raise

def engineer_features(X, feature_names):
    print("Engineering features...")
    # Convert sparse matrix to DataFrame if necessary
    if scipy.sparse.issparse(X):
        X = pd.DataFrame.sparse.from_spmatrix(X, columns=feature_names)
    elif isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names)
    
    # List of potential features we want to use
    potential_features = [
        'Total reported direct emissions',
        'Electricity Generation',
        'Nitrous Oxide (N2O) emissions',
        'Methane (CH4) emissions',
        'HFC emissions'
    ]

    # Check which features are actually present in the dataset
    available_features = [feat for feat in potential_features if feat in X.columns]

    logging.info(f"Available features for engineering: {available_features}")

    # Create interaction terms for available features
    for i in range(len(available_features)):
        for j in range(i+1, len(available_features)):
            feat1 = available_features[i]
            feat2 = available_features[j]
            if feat1 in X.columns and feat2 in X.columns:
                interaction_name = f"{feat1}_X_{feat2}"
                X[interaction_name] = X[feat1] * X[feat2]
                logging.info(f"Created interaction term: {interaction_name}")

    logging.info("Feature engineering completed")
    return X

def preprocess_data(data, target_column=None):
    print("Preprocessing data...")
    logging.info("Starting data preprocessing")

    if isinstance(data, pd.DataFrame):
        # If data is a DataFrame, we assume it's raw data that needs preprocessing
        logging.info(f"Columns in the dataset: {data.columns.tolist()}")

        if target_column is None:
            raise ValueError("target_column must be specified when input is a DataFrame")

        X = data.drop([target_column], axis=1)
        y = data[target_column]

        # Handle NaN values in the target variable
        nan_count = y.isna().sum()
        logging.info(f"Rows with NaN in target variable: {nan_count}")
        logging.info(f"Total rows before removing NaN: {len(y)}")

        X = X[y.notna()]
        y = y.dropna()

        # Log transform the target variable
        y = np.log1p(y)
        logging.info("Applied log transformation to target variable")

        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        # Create preprocessing steps for numeric and categorical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Fit the preprocessor and transform the data
        X_processed = preprocessor.fit_transform(X)
        
        # Get feature names after preprocessing
        onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = onehot_encoder.get_feature_names_out(categorical_features)
        feature_names = list(numeric_features) + list(cat_feature_names)

    else:
        # If data is not a DataFrame, we assume it's already preprocessed
        X_processed = data
        y = target_column  # In this case, target_column is actually y
        preprocessor = None
        feature_names = None

    logging.info(f"Total rows after preprocessing: {len(y)}")
    return X_processed, y, preprocessor, feature_names


def create_preprocessor(X_train, preprocessor_path):
    print("Creating preprocessor...")
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

def create_range_based_models(X, y, model_class, param_grid):
    print("Creating range-based models...")
    # Define emission ranges
    low_threshold = y.quantile(0.33)
    high_threshold = y.quantile(0.67)

    # Split data into ranges
    X_low, y_low = X[y <= low_threshold], y[y <= low_threshold]
    X_medium, y_medium = X[(y > low_threshold) & (y <= high_threshold)], y[(y > low_threshold) & (y <= high_threshold)]
    X_high, y_high = X[y > high_threshold], y[y > high_threshold]

    # Create and train models for each range
    models = {}
    for name, X_range, y_range in [("low", X_low, y_low), ("medium", X_medium, y_medium), ("high", X_high, y_high)]:
        model = GridSearchCV(model_class(), param_grid, cv=3, n_jobs=-1)
        model.fit(X_range, y_range)
        models[name] = model
        logging.info(f"Trained {name} emissions model. Best params: {model.best_params_}")

    return models, low_threshold, high_threshold

def predict_with_range_models(models, X, low_threshold, high_threshold):
    predictions = np.zeros(X.shape[0])

    # Use the first feature (assuming it's 'Total reported direct emissions') to determine which model to use
    if scipy.sparse.issparse(X):
        emissions = X.tocsr()[:, 0].toarray().flatten()
    else:
        emissions = X[:, 0]

    for i in range(X.shape[0]):
        if emissions[i] <= low_threshold:
            predictions[i] = models['low'].predict(X[i].reshape(1, -1))[0]
        elif emissions[i] <= high_threshold:
            predictions[i] = models['medium'].predict(X[i].reshape(1, -1))[0]
        else:
            predictions[i] = models['high'].predict(X[i].reshape(1, -1))[0]

    return predictions

def prediction_intervals_rf(rf_model, X, percentile=95):
    predictions = []
    for estimator in rf_model.estimators_:
        predictions.append(estimator.predict(X))
    predictions = np.array(predictions)
    
    lower_bound = np.percentile(predictions, (100 - percentile) / 2., axis=0)
    upper_bound = np.percentile(predictions, 100 - (100 - percentile) / 2., axis=0)
    point_estimate = np.mean(predictions, axis=0)
    
    return point_estimate, lower_bound, upper_bound

def create_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, feature_names):
    print(f"Creating and evaluating {model_name}...")
    logging.info(f"Training and evaluating {model_name}")
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    logging.info(f"{model_name} Cross-Validation Scores: {cv_scores}")
    logging.info(f"{model_name} Mean CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
    
    # Predictions and evaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_pred_train = np.expm1(y_pred_train)
    y_pred_test = np.expm1(y_pred_test)
    y_train_original = np.expm1(y_train)
    y_test_original = np.expm1(y_test)
    
    logging.info(f"\n{model_name} Performance:")
    logging.info(f"Train R2: {r2_score(y_train_original, y_pred_train):.4f}")
    logging.info(f"Test R2: {r2_score(y_test_original, y_pred_test):.4f}")
    logging.info(f"Train MAE: {mean_absolute_error(y_train_original, y_pred_train):.2f}")
    logging.info(f"Test MAE: {mean_absolute_error(y_test_original, y_pred_test):.2f}")
    logging.info(f"Train WMAPE: {weighted_mape(y_train_original, y_pred_train):.2f}%")
    logging.info(f"Test WMAPE: {weighted_mape(y_test_original, y_pred_test):.2f}%")
   
    
    # Feature importance
    try:
        if hasattr(model.best_estimator_, 'feature_importances_'):
            logging.info(f"Extracting feature importances for {model_name}")
            importances = model.best_estimator_.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            logging.info(f"Shape of importances: {importances.shape}")
            
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
        else:
            logging.warning(f"{model_name} best estimator does not have feature_importances_ attribute")
    except Exception as e:
        logging.error(f"Error in feature importance extraction for {model_name}: {str(e)}")
        logging.error(traceback.format_exc())
    
    return model

def main():
    print("Starting main function...")
    # Load the feature-engineered data
    data_path = os.path.join('..', 'data', 'feature_engineered_data.csv')
    df = load_data(data_path)

    # Preprocess the data
    X, y, preprocessor, feature_names = preprocess_data(df, 'CO2 emissions (non-biogenic) ')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # Create range-based models
    rf_models, rf_low_threshold, rf_high_threshold = create_range_based_models(X_train, y_train, RandomForestRegressor, rf_params)
    gb_models, gb_low_threshold, gb_high_threshold = create_range_based_models(X_train, y_train, GradientBoostingRegressor, gb_params)

    # Evaluate range-based models
    rf_predictions = predict_with_range_models(rf_models, X_test, rf_low_threshold, rf_high_threshold)
    gb_predictions = predict_with_range_models(gb_models, X_test, gb_low_threshold, gb_high_threshold)

    # Inverse transform predictions and actual values
    y_test_original = np.expm1(y_test)
    rf_predictions = np.expm1(rf_predictions)
    gb_predictions = np.expm1(gb_predictions)

    logging.info("\nRange-based Random Forest Performance:")
    logging.info(f"Test R2: {r2_score(y_test_original, rf_predictions):.4f}")
    logging.info(f"Test MAE: {mean_absolute_error(y_test_original, rf_predictions):.2f}")

    logging.info("\nRange-based Gradient Boosting Performance:")
    logging.info(f"Test R2: {r2_score(y_test_original, gb_predictions):.4f}")
    logging.info(f"Test MAE: {mean_absolute_error(y_test_original, gb_predictions):.2f}")

    # Train and evaluate single models (for comparison)
    rf_model = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, n_jobs=-1)
    gb_model = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=3, n_jobs=-1)

    rf_model = create_and_evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random Forest", feature_names)
    gb_model = create_and_evaluate_model(gb_model, X_train, X_test, y_train, y_test, "Gradient Boosting", feature_names)

    # Print best parameters
    logging.info(f"\nBest Random Forest Parameters: {rf_model.best_params_}")
    logging.info(f"Best Gradient Boosting Parameters: {gb_model.best_params_}")

    # Prediction intervals for Random Forest
    point_estimate, lower_bound, upper_bound = prediction_intervals_rf(rf_model.best_estimator_, X_test)
    logging.info(f"Average prediction interval width: {np.mean(np.expm1(upper_bound) - np.expm1(lower_bound)):.2f}")

    # Save the best model (based on test set performance)
    rf_score = r2_score(y_test_original, np.expm1(rf_model.predict(X_test)))
    gb_score = r2_score(y_test_original, np.expm1(gb_model.predict(X_test)))

    best_model = rf_model if rf_score > gb_score else gb_model
    best_model_name = "Random Forest" if rf_score > gb_score else "Gradient Boosting"

    model_path = os.path.join('..', 'models', 'best_model.joblib')
    joblib.dump(best_model, model_path)
    logging.info(f"\nBest model ({best_model_name}) saved to {model_path}")

    # Print final performance of the best model
    best_predictions = np.expm1(best_model.predict(X_test))
    logging.info(f"\nBest Model ({best_model_name}) Final Performance:")
    logging.info(f"Test R2: {r2_score(y_test_original, best_predictions):.4f}")
    logging.info(f"Test MAE: {mean_absolute_error(y_test_original, best_predictions):.2f}")

    # Create residual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(best_predictions, y_test_original - best_predictions)
    plt.xlabel("Predicted CO2 Emissions")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot - Best Model ({best_model_name})")
    residual_plot_path = "../models/residual_plot.png"
    plt.savefig(residual_plot_path)
    plt.close()
    logging.info(f"Residual plot saved to {residual_plot_path}")

    logging.info("\n--- Evaluating New Models ---")

    # Range-Based Model
    logging.info("\nTraining and evaluating Range-Based Model...")
    range_based_model = RangeBasedRegressor(base_estimator=GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5))
    range_based_model.fit(X_train, y_train)
    y_pred_range = range_based_model.predict(X_test)

    logging.info("Range-Based Model Performance:")
    logging.info(f"R2 Score: {r2_score(y_test, y_pred_range):.4f}")
    logging.info(f"MAE: {mean_absolute_error(y_test, y_pred_range):.2f}")
    logging.info(f"Weighted MAPE: {weighted_mape(y_test, y_pred_range):.2f}%")

    # Feature Engineering
    logging.info("\nApplying Feature Engineering...")
    X_engineered = engineer_features(X, feature_names)
    X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(X_engineered, y, test_size=0.2, random_state=42)

    gb_model_eng = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    gb_model_eng.fit(X_train_eng, y_train_eng)
    y_pred_eng = gb_model_eng.predict(X_test_eng)

    logging.info("Gradient Boosting with Engineered Features:")
    logging.info(f"R2 Score: {r2_score(y_test_eng, y_pred_eng):.4f}")
    logging.info(f"MAE: {mean_absolute_error(y_test_eng, y_pred_eng):.2f}")
    logging.info(f"Weighted MAPE: {weighted_mape(y_test_eng, y_pred_eng):.2f}%")

    # Gradient Boosting with Prediction Intervals
    logging.info("\nTraining Gradient Boosting with Prediction Intervals...")
    model_with_pi = GradientBoostingWithPI(n_estimators=100, learning_rate=0.1, max_depth=5)
    model_with_pi.fit(X_train_eng, y_train_eng)
    y_pred_pi, lower, upper = model_with_pi.predict_with_pi(X_test_eng)

    logging.info("Gradient Boosting with Prediction Intervals:")
    logging.info(f"R2 Score: {r2_score(y_test_eng, y_pred_pi):.4f}")
    logging.info(f"MAE: {mean_absolute_error(y_test_eng, y_pred_pi):.2f}")
    logging.info(f"Weighted MAPE: {weighted_mape(y_test_eng, y_pred_pi):.2f}%")

    avg_interval_width = np.mean(upper - lower)
    logging.info(f"Average 95% Prediction Interval Width: {avg_interval_width:.2f}")

    # Compare new models with the best previous model
    new_models = {
        'Range-Based': (range_based_model, y_pred_range),
        'GB with Engineered Features': (gb_model_eng, y_pred_eng),
        'GB with Prediction Intervals': (model_with_pi, y_pred_pi)
    }

    best_new_model_name = None
    best_new_score = r2_score(y_test, best_model.predict(X_test))

    for name, (model, predictions) in new_models.items():
        score = r2_score(y_test, predictions)
        if score > best_new_score:
            best_new_score = score
            best_new_model_name = name
            best_new_model = model

    if best_new_model_name:
        logging.info(f"\nNew best model: {best_new_model_name}")
        new_model_path = os.path.join('..', 'models', 'best_model_updated.joblib')
        joblib.dump(best_new_model, new_model_path)
        logging.info(f"New best model saved to {new_model_path}")
    else:
        logging.info("\nPrevious model remains the best performer.")

    # Error Analysis
    logging.info("\nError Analysis:")
    # Calculate and log various error metrics
    mse = mean_squared_error(y_test_original, best_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, best_predictions)

    mape = np.mean(np.abs((y_test_original - best_predictions) / y_test_original) * 100)
    if np.isinf(mape):
        mape = np.mean(np.abs((y_test_original - best_predictions) / (y_test_original + 1e-8)) * 100)

    logging.info(f"Mean Squared Error: {mse:.2f}")
    logging.info(f"Root Mean Squared Error: {rmse:.2f}")
    logging.info(f"Mean Absolute Error: {mae:.2f}")
    logging.info(f"Mean Absolute Percentage Error: {mape:.2f}%")

    # Residual Analysis
    residuals = y_test_original - best_predictions

    # Residual statistics
    logging.info(f"\nResidual Statistics:")
    logging.info(f"Mean of Residuals: {np.mean(residuals):.2f}")
    logging.info(f"Standard Deviation of Residuals: {np.std(residuals):.2f}")

    # Error Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50)
    plt.xlabel("Residual Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals")
    error_dist_path = "../models/error_distribution.png"
    plt.savefig(error_dist_path)
    plt.close()
    logging.info(f"Error distribution plot saved to {error_dist_path}")

    # Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original, best_predictions)
    plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
    plt.xlabel("Actual CO2 Emissions")
    plt.ylabel("Predicted CO2 Emissions")
    plt.title(f"Actual vs Predicted - Best Model ({best_model_name})")
    actual_vs_predicted_path = "../models/actual_vs_predicted.png"
    plt.savefig(actual_vs_predicted_path)
    plt.close()
    logging.info(f"Actual vs Predicted plot saved to {actual_vs_predicted_path}")

    # Analyze largest errors
    abs_errors = np.abs(residuals)
    largest_errors = pd.Series(abs_errors).nlargest(10)
    logging.info("\nLargest Errors:")
    for i, error in largest_errors.items():
        actual = y_test_original[i] if i < len(y_test_original) else "N/A"
        predicted = best_predictions[i] if i < len(best_predictions) else "N/A"
        if isinstance(actual, (int, float)) and isinstance(predicted, (int, float)):
            logging.info(f"Index: {i}, Actual: {actual:.2f}, Predicted: {predicted:.2f}, Error: {error:.2f}")
        else:
            logging.info(f"Index: {i}, Actual: {actual}, Predicted: {predicted}, Error: {error:.2f}")

    # Performance across different ranges
    y_test_original_series = pd.Series(y_test_original)
    ranges = [y_test_original_series.min(), y_test_original_series.quantile(0.25), 
              y_test_original_series.quantile(0.5), y_test_original_series.quantile(0.75), 
              y_test_original_series.max()]
    logging.info("\nPerformance across different ranges:")
    for i in range(len(ranges) - 1):
        mask = (y_test_original_series >= ranges[i]) & (y_test_original_series < ranges[i+1])
        range_predictions = best_predictions[mask]
        range_actuals = y_test_original_series[mask]
        range_mse = mean_squared_error(range_actuals, range_predictions)
        range_r2 = r2_score(range_actuals, range_predictions)
        logging.info(f"Range {ranges[i]:.2f} to {ranges[i+1]:.2f}:")
        logging.info(f"  MSE: {range_mse:.2f}")
        logging.info(f"  R2: {range_r2:.4f}")

    logging.info("Model training and evaluation completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc())