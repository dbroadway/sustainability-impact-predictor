import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the data
data_path = './data/2022_data_summary_spreadsheets/ghgp_data_2022.xlsx'
df = pd.read_excel(data_path)

# Basic preprocessing
def preprocess_data(df):
    # Select relevant columns (adjust as needed based on your data)
    columns_to_keep = ['FACILITY_ID', 'STATE', 'PRIMARY_NAICS_CODE', 'INDUSTRY_TYPE_SECTORS', 
                       'LATITUDE', 'LONGITUDE', 'TOTAL_REPORTED_DIRECT_EMISSIONS']
    df = df[columns_to_keep]
    
    # Rename columns for consistency
    df.columns = ['facility_id', 'state', 'naics_code', 'industry_type', 
                  'latitude', 'longitude', 'total_emissions']
    
    # Handle missing values
    df = df.dropna(subset=['total_emissions'])
    
    # Create log emissions
    df['log_emissions'] = np.log1p(df['total_emissions'])
    
    return df

df = preprocess_data(df)

# Split the data
X = df.drop(['facility_id', 'total_emissions'], axis=1)
y = df['total_emissions']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessor
numeric_features = ['latitude', 'longitude', 'naics_code']
categorical_features = ['state', 'industry_type']

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

# Fit preprocessor
preprocessor.fit(X_train)

# Save preprocessor
joblib.dump(preprocessor, 'preprocessor.joblib')

# Transform data
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save processed data
np.save('X_train_processed.npy', X_train_processed)
np.save('X_test_processed.npy', X_test_processed)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("Data preprocessing completed. Preprocessor and processed data saved.")