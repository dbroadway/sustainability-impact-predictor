import pandas as pd
import boto3
from io import BytesIO, StringIO

# Set up S3 client using environment variables
s3 = boto3.client('s3')

# Your S3 bucket name
bucket_name = 'sustainability-impact-predictor-data'
input_file_path = 'epa-greenhouse-gas-data/ghgp_data_2022.xlsx'
output_file_path = 'processed-data/merged_emissions_data.csv'

def read_excel_from_s3(bucket, key):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.ExcelFile(BytesIO(obj['Body'].read()))
    except Exception as e:
        print(f"Error reading file from S3: {e}")
        return None

def clean_column_names(df):
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    return df

def upload_to_s3(df, bucket, key):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
    print(f"File uploaded to S3: s3://{bucket}/{key}")

# Read the Excel file
excel_file = read_excel_from_s3(bucket_name, input_file_path)

if excel_file:
    # Read Direct Emitters and Suppliers sheets
    direct_emitters = pd.read_excel(excel_file, sheet_name='Direct Emitters', header=3)
    suppliers = pd.read_excel(excel_file, sheet_name='Suppliers', header=3)

    # Clean column names
    direct_emitters = clean_column_names(direct_emitters)
    suppliers = clean_column_names(suppliers)

    # Merge datasets on facility_id
    merged_data = pd.merge(direct_emitters, suppliers, on='facility_id', how='left', suffixes=('_direct', '_supplier'))

    # Display info about the merged dataset
    print("Merged Dataset Info:")
    print(merged_data.info())

    # Sample of the merged data
    print("\nSample of Merged Data:")
    print(merged_data.head())

    # Upload the merged dataset to S3
    upload_to_s3(merged_data, bucket_name, output_file_path)

else:
    print("Failed to read the Excel file.")

print("Data integration complete.")