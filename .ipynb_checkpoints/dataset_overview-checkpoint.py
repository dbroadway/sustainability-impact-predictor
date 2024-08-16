import pandas as pd
import boto3
from io import BytesIO

# Set up S3 client using environment variables
s3 = boto3.client('s3')

# Your S3 bucket name
bucket_name = 'sustainability-impact-predictor-data'
file_path = 'epa-greenhouse-gas-data/ghgp_data_2022.xlsx'

def read_excel_from_s3(bucket, key):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.ExcelFile(BytesIO(obj['Body'].read()))
    except Exception as e:
        print(f"Error reading file from S3: {e}")
        return None

def analyze_sheet(sheet_name, df):
    print(f"\nAnalyzing sheet: {sheet_name}")
    print(f"Shape: {df.shape}")
    print("\nColumn names:")
    for col in df.columns:
        print(col)
    print("\nSample data:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\n" + "="*50)

# Read the Excel file
excel_file = read_excel_from_s3(bucket_name, file_path)

if excel_file:
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=3)
        analyze_sheet(sheet_name, df)
else:
    print("Failed to read the Excel file.")

print("Dataset overview complete.")