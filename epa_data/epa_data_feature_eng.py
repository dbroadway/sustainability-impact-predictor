import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import boto3
import joblib
from io import BytesIO

print("Starting feature engineering script...")

# Set up S3 client
s3 = boto3.client('s3')

# S3 bucket and file information
bucket_name = 'sustainability-impact-predictor-data'
input_file_key = 'processed-data/merged_emissions_data.csv'
output_file_key = 'processed-data/feature_engineered_data.csv'

def load_data_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(BytesIO(obj['Body'].read()))

def save_data_to_s3(df, bucket, key):
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())

def engineer_features(df):
    # Create state-level features
    state_emissions = df.groupby('state_direct')['total_reported_direct_emissions'].mean().reset_index()
    state_emissions.columns = ['state_direct', 'state_avg_emissions']
    df = pd.merge(df, state_emissions, on='state_direct', how='left')
    
    # Create industry sector features
    industry_emissions = df.groupby('industry_type_(sectors)')['total_reported_direct_emissions'].mean().reset_index()
    industry_emissions.columns = ['industry_type_(sectors)', 'industry_avg_emissions']
    df = pd.merge(df, industry_emissions, on='industry_type_(sectors)', how='left')
    
    # Normalize emissions values
    df['log_emissions'] = np.log1p(df['total_reported_direct_emissions'])
    
    # Create binary feature for power plants (assuming power plants are in the "Electricity Generation" sector)
    df['is_power_plant'] = (df['industry_type_(sectors)'] == 'Electricity Generation').astype(int)
    
    # Select features for model training
    features = ['state_direct', 'industry_type_(sectors)', 'state_avg_emissions', 
                'industry_avg_emissions', 'log_emissions', 'is_power_plant', 
                'latitude_direct', 'longitude_direct', 'primary_naics_code_direct']
    
    return df[features + ['total_reported_direct_emissions']]

def create_preprocessor(df):
    numeric_features = ['state_avg_emissions', 'industry_avg_emissions', 'log_emissions', 
                        'latitude_direct', 'longitude_direct', 'primary_naics_code_direct']
    categorical_features = ['state_direct', 'industry_type_(sectors)']
    
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
    
    return preprocessor

# Main execution
try:
    # Load data
    print("Loading data from S3...")
    df = load_data_from_s3(bucket_name, input_file_key)
    
    # Perform feature engineering
    print("Engineering features...")
    df_engineered = engineer_features(df)
    
    # Create and fit preprocessor
    print("Creating and fitting preprocessor...")
    preprocessor = create_preprocessor(df_engineered)
    preprocessor.fit(df_engineered.drop('total_reported_direct_emissions', axis=1))
    
    # Save preprocessor
    print("Saving preprocessor...")
    joblib.dump(preprocessor, 'preprocessor.joblib')
    s3.upload_file('preprocessor.joblib', bucket_name, 'models/preprocessor.joblib')
    
    # Save engineered data
    print("Saving engineered data to S3...")
    save_data_to_s3(df_engineered, bucket_name, output_file_key)
    
    print("Feature engineering completed successfully.")

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    print("Script execution finished.")