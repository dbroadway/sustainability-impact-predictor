import joblib
import pandas as pd

# Load preprocessor and models
preprocessor = joblib.load('preprocessor.joblib')
rf_model = joblib.load('rf_model.joblib')
xgb_model = joblib.load('xgb_model.joblib')

def predict_emissions(data):
    # Preprocess the input data
    processed_data = preprocessor.transform(data)
    
    # Make predictions
    rf_prediction = rf_model.predict(processed_data)
    xgb_prediction = xgb_model.predict(processed_data)
    
    # Average the predictions
    final_prediction = (rf_prediction + xgb_prediction) / 2
    
    return final_prediction[0]

# Example usage
if __name__ == "__main__":
    # Sample input data (adjust based on your actual features)
    sample_data = pd.DataFrame({
        'state': ['CA'],
        'naics_code': [221112],
        'industry_type': ['Power Plants'],
        'latitude': [34.052235],
        'longitude': [-118.243683]
    })
    
    prediction = predict_emissions(sample_data)
    print(f"Predicted emissions: {prediction:.2f}")