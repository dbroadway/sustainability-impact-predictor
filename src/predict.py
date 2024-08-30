import os
import joblib
import pandas as pd
import numpy as np
import argparse

print("Current working directory:", os.getcwd())

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

preprocessor = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'preprocessor.joblib'))
best_model = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'best_model.joblib'))

print("Preprocessor structure:")
print(preprocessor)
print("\nPreprocessor named transformers:")
for name, transformer in preprocessor.named_transformers_.items():
    print(f"{name}:")
    print(transformer)
    if hasattr(transformer, 'named_steps'):
        for step_name, step in transformer.named_steps.items():
            print(f"  {step_name}:")
            print(f"    {step}")
            if hasattr(step, 'get_feature_names_out'):
                print(f"    Feature names: {step.get_feature_names_out()}")

column_mapping = {
    'state': 'State',
    'naics_code': 'Primary NAICS Code',
    'industry_type': 'Industry Type (sectors)',
    'latitude': 'Latitude',
    'longitude': 'Longitude'
}

def transform_input_data(data, preprocessor):
    print("Input data shape:", data.shape)
    print("Input data columns:", data.columns.tolist())
    
    # Rename columns according to the expected names
    data.rename(columns=column_mapping, inplace=True)
    
    print("Renamed columns:", data.columns.tolist())
    
    # Create a DataFrame with all expected columns, initialized with 0
    expected_columns = preprocessor.feature_names_in_
    full_data = pd.DataFrame(0, index=range(len(data)), columns=expected_columns)
    
    # Fill in the data we have
    for col in data.columns:
        if col in expected_columns:
            full_data[col] = data[col]
    
    print("Full data shape:", full_data.shape)
    print("Full data columns:", full_data.columns.tolist())
    print("Sample of full data:")
    print(full_data.head())
    
    # Apply the preprocessor
    transformed_data = preprocessor.transform(full_data)
    
    print("Transformed data shape:", transformed_data.shape)
    
    return transformed_data

def predict_emissions(data):
    print("Input data columns:", data.columns.tolist())

    # Transform input data using the preprocessor
    transformed_data = transform_input_data(data, preprocessor)

    print("Transformed data shape:", transformed_data.shape)

    # Make predictions
    predictions = best_model.predict(transformed_data)

    return predictions

def main(input_file, output_file):
    # Load input data
    input_data = pd.read_csv(input_file)
    print("Loaded input data shape:", input_data.shape)

    # Make predictions
    predictions = predict_emissions(input_data)

    # Add predictions to the input dataframe
    input_data['Predicted_CO2_Emissions'] = np.expm1(predictions)  # Assuming predictions are log-transformed

    # Save results
    input_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict CO2 emissions for new data.")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("output_file", help="Path to save the output CSV file")
    args = parser.parse_args()

    main(args.input_file, args.output_file)