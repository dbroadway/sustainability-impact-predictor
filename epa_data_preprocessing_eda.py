import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import boto3
from io import StringIO
import time
import joblib
import os
import scipy.sparse

print("Script started.")

# Set up S3 client
s3 = boto3.client('s3')

# S3 bucket and file information
bucket_name = 'sustainability-impact-predictor-data'
file_key = 'processed-data/merged_emissions_data.csv'
chunk_size = 1000  # Adjust this based on your memory constraints

def process_chunk(chunk, preprocessor):
    if preprocessor is None:
        # Function to determine if a column should be treated as numeric
        def is_numeric(col):
            try:
                pd.to_numeric(chunk[col].replace({'confidential': np.nan}), errors='raise')
                return True
            except ValueError:
                return False

        numeric_features = [col for col in chunk.columns if is_numeric(col)]
        categorical_features = [col for col in chunk.columns if col not in numeric_features]
        
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
        
        # Replace 'confidential' with NaN in numeric columns
        for col in numeric_features:
            chunk[col] = pd.to_numeric(chunk[col].replace({'confidential': np.nan}), errors='coerce')
        
        preprocessor.fit(chunk)
        joblib.dump(preprocessor, 'preprocessor.joblib')
    
    # Replace 'confidential' with NaN in numeric columns before transform
    numeric_features = preprocessor.transformers_[0][2]  # Get numeric feature names from preprocessor
    for col in numeric_features:
        chunk[col] = pd.to_numeric(chunk[col].replace({'confidential': np.nan}), errors='coerce')
    
    X_processed = preprocessor.transform(chunk)
    if scipy.sparse.issparse(X_processed):
        X_processed = X_processed.toarray()
    return X_processed, preprocessor

# Function to update EDA results
def update_eda_results(chunk, eda_results):
    if eda_results is None:
        eda_results = {
            'total_emissions': [],
            'states': {},
            'industries': {}
        }
    
    total_emissions = pd.to_numeric(chunk['total_reported_direct_emissions'], errors='coerce')
    eda_results['total_emissions'].extend(total_emissions.dropna().tolist())
    
    state_emissions = chunk.groupby('state_direct')['total_reported_direct_emissions'].sum()
    for state, emissions in state_emissions.items():
        if pd.notna(emissions):
            eda_results['states'][state] = eda_results['states'].get(state, 0) + emissions
    
    industry_emissions = chunk.groupby('industry_type_(sectors)')['total_reported_direct_emissions'].mean()
    for industry, emissions in industry_emissions.items():
        if pd.notna(emissions):
            eda_results['industries'][industry] = eda_results['industries'].get(industry, [])
            eda_results['industries'][industry].append(emissions)
    
    return eda_results

# Main processing loop
preprocessor = None
eda_results = None
processed_chunks = []

try:
    for chunk in pd.read_csv(StringIO(s3.get_object(Bucket=bucket_name, Key=file_key)['Body'].read().decode('utf-8')), chunksize=chunk_size):
        print(f"Processing chunk of size {len(chunk)}...")
        X_processed, preprocessor = process_chunk(chunk, preprocessor)
        processed_chunks.append(X_processed)
        eda_results = update_eda_results(chunk, eda_results)
        
        # Save intermediate results
        np.save(f'processed_chunk_{len(processed_chunks)}.npy', X_processed)
        joblib.dump(eda_results, 'eda_results.joblib')
        
        print(f"Chunk {len(processed_chunks)} processed and saved.")

    # Combine all processed chunks
    X_processed_full = np.concatenate(processed_chunks, axis=0)
    np.save('X_processed_full.npy', X_processed_full)

    print("All chunks processed. Performing final EDA...")

    # Perform final EDA
    plt.figure(figsize=(10, 6))
    sns.histplot(eda_results['total_emissions'], kde=True)
    plt.title('Distribution of Total Reported Direct Emissions')
    plt.xlabel('Emissions (metric tons CO2e)')
    plt.ylabel('Frequency')
    plt.savefig('total_emissions_distribution.png')
    plt.close()

    top_states = pd.Series(eda_results['states']).sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    top_states.plot(kind='bar')
    plt.title('Top 10 States by Total Emissions')
    plt.xlabel('State')
    plt.ylabel('Total Emissions (metric tons CO2e)')
    plt.xticks(rotation=45)
    plt.savefig('top_10_states.png')
    plt.close()

    top_industries = pd.Series({k: np.mean(v) for k, v in eda_results['industries'].items()}).sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    top_industries.plot(kind='bar')
    plt.title('Top 10 Industry Sectors by Average Emissions')
    plt.xlabel('Industry Sector')
    plt.ylabel('Average Emissions (metric tons CO2e)')
    plt.xticks(rotation=90)
    plt.savefig('top_10_industries.png')
    plt.close()

    print("Uploading results to S3...")
    s3.upload_file('preprocessor.joblib', bucket_name, 'models/preprocessor.joblib')
    s3.upload_file('X_processed_full.npy', bucket_name, 'processed-data/X_processed_full.npy')
    s3.upload_file('total_emissions_distribution.png', bucket_name, 'visualizations/total_emissions_distribution.png')
    s3.upload_file('top_10_states.png', bucket_name, 'visualizations/top_10_states.png')
    s3.upload_file('top_10_industries.png', bucket_name, 'visualizations/top_10_industries.png')

    print("Script completed successfully.")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    # You can add more error handling here if needed

finally:
    print("Script execution finished.")