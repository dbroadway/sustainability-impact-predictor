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

# Set up S3 client
s3 = boto3.client('s3')

# S3 bucket and file information
bucket_name = 'sustainability-impact-predictor-data'
file_key = 'processed-data/merged_emissions_data.csv'

# Read the CSV file from S3
obj = s3.get_object(Bucket=bucket_name, Key=file_key)
df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

# Identify numeric and categorical columns
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

# Create preprocessing pipelines
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

# Fit and transform the data
X_processed = preprocessor.fit_transform(df)

# Convert to DataFrame for easier analysis
feature_names = (numeric_features.tolist() + 
                 preprocessor.named_transformers_['cat']
                 .named_steps['onehot']
                 .get_feature_names_out(categorical_features).tolist())
X_processed_df = pd.DataFrame(X_processed.toarray(), columns=feature_names)

# Exploratory Data Analysis
def perform_eda(df, processed_df):
    # Distribution of total emissions
    plt.figure(figsize=(10, 6))
    sns.histplot(df['total_reported_direct_emissions'], kde=True)
    plt.title('Distribution of Total Reported Direct Emissions')
    plt.xlabel('Emissions (metric tons CO2e)')
    plt.ylabel('Frequency')
    plt.show()
    
    # Top 10 states by total emissions
    top_states = df.groupby('state_direct')['total_reported_direct_emissions'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    top_states.plot(kind='bar')
    plt.title('Top 10 States by Total Emissions')
    plt.xlabel('State')
    plt.ylabel('Total Emissions (metric tons CO2e)')
    plt.xticks(rotation=45)
    plt.show()
    
    # Correlation heatmap of numeric features
    corr = processed_df.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr, cmap='coolwarm', linewidths=0.5, annot=False)
    plt.title('Correlation Heatmap of Processed Features')
    plt.show()
    
    # Top 10 industry sectors by average emissions
    top_industries = df.groupby('industry_type_(sectors)')['total_reported_direct_emissions'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    top_industries.plot(kind='bar')
    plt.title('Top 10 Industry Sectors by Average Emissions')
    plt.xlabel('Industry Sector')
    plt.ylabel('Average Emissions (metric tons CO2e)')
    plt.xticks(rotation=90)
    plt.show()

# Perform EDA
perform_eda(df, X_processed_df)

print("Data preprocessing and initial EDA completed.")

# Save the preprocessor and processed data
import joblib

joblib.dump(preprocessor, 'preprocessor.joblib')
print("Preprocessor saved as 'preprocessor.joblib'")

# Save processed data to CSV
X_processed_df.to_csv('processed_emissions_data.csv', index=False)
print("Processed data saved as 'processed_emissions_data.csv'")

# Upload files to S3
s3.upload_file('preprocessor.joblib', bucket_name, 'models/preprocessor.joblib')
s3.upload_file('processed_emissions_data.csv', bucket_name, 'processed-data/processed_emissions_data.csv')
print("Files uploaded to S3.")