import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
from io import BytesIO
from botocore.exceptions import ClientError

# Set up S3 client using environment variables
s3 = boto3.client('s3')

# Your S3 bucket name
bucket_name = 'sustainability-impact-predictor-data'

# Function to read Excel from S3
def read_excel_from_s3(bucket, key):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        # Read all sheets, skip first 3 rows, use 4th row as header
        excel_file = pd.ExcelFile(BytesIO(obj['Body'].read()))
        sheets = {}
        for sheet_name in excel_file.sheet_names:
            sheets[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name, skiprows=3, header=0)
        return sheets
    except ClientError as e:
        print(f"Error reading file from S3: {e}")
        return None

# File path
file_path = 'epa-greenhouse-gas-data/ghgp_data_2022.xlsx'

print(f"Attempting to read file: {file_path}")

# Read the main dataset
sheets = read_excel_from_s3(bucket_name, file_path)

if sheets:
    print("Excel file successfully loaded. Available sheets:")
    for i, sheet_name in enumerate(sheets.keys()):
        print(f"{i+1}. {sheet_name}")
    
    sheet_choice = int(input("Enter the number of the sheet you want to analyze: ")) - 1
    sheet_name = list(sheets.keys())[sheet_choice]
    df = sheets[sheet_name]
    
    print(f"\nAnalyzing sheet: {sheet_name}")
    print("\nDataset Info:")
    print(df.info())
    
    print("\nColumn names:")
    for col in df.columns:
        print(col)
    
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    print("\nSummary statistics of numeric columns:")
    print(df.describe())
    
    print("\nMissing values in each column:")
    print(df.isnull().sum())

    # Check for columns that might represent emissions
    potential_emissions_columns = [col for col in df.columns if 'EMISSION' in str(col).upper() or 'GHG' in str(col).upper()]
    
    if potential_emissions_columns:
        for col in potential_emissions_columns:
            print(f"\nAnalyzing column: {col}")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel('Emissions')
            plt.ylabel('Frequency')
            plt.show()

    else:
        print("\nNo column containing 'EMISSION' or 'GHG' found in the dataset.")
        print("Available columns:", df.columns.tolist())
        
    # Additional exploration
    print("\nUnique values in categorical columns:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"{col}: {df[col].nunique()} unique values")
        if df[col].nunique() < 10:  # If there are few unique values, print them
            print(df[col].value_counts())
        print()

else:
    print("Failed to read the dataset. Please check the bucket name and file path.")

print("Data exploration complete.")