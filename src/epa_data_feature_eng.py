import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression

def engineer_features(df):
    print("Engineering features...")
    
    # Total GHG emissions (excluding biogenic CO2)
    df['total_ghg_emissions'] = df['Total reported direct emissions']
    
    # Calculate the percentage of each GHG type
    ghg_columns = [
        'CO2 emissions (non-biogenic) ',
        'Methane (CH4) emissions ',
        'Nitrous Oxide (N2O) emissions ',
        'HFC emissions',
        'PFC emissions',
        'SF6 emissions ',
        'NF3 emissions',
        'Other Fully Fluorinated GHG emissions',
        'HFE emissions',
        'Very Short-lived Compounds emissions'
    ]
    
    for col in ghg_columns:
        if col in df.columns:
            df[f'{col.strip()}_percentage'] = df[col] / df['total_ghg_emissions'] * 100
        else:
            print(f"Warning: Column '{col}' not found in the dataset. Skipping this column.")
    
    # Industry type features
    df['industry_type'] = df['Industry Type (sectors)']
    
    # Location features
    df['latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    
    # Create binary features for each industry subpart
    industry_subparts = [
        'Stationary Combustion', 'Electricity Generation', 'Adipic Acid Production',
        'Aluminum Production', 'Ammonia Manufacturing', 'Cement Production',
        'Electronics Manufacture', 'Ferroalloy Production', 'Fluorinated GHG Production',
        'Glass Production', 'HCFC–22 Production from HFC–23 Destruction', 'Hydrogen Production',
        'Iron and Steel Production', 'Lead Production', 'Lime Production', 'Magnesium Production',
        'Miscellaneous Use of Carbonates', 'Nitric Acid Production',
        'Petroleum and Natural Gas Systems – Offshore Production',
        'Petroleum and Natural Gas Systems – Processing',
        'Petroleum and Natural Gas Systems – Transmission/Compression',
        'Petroleum and Natural Gas Systems – Underground Storage',
        'Petroleum and Natural Gas Systems – LNG Storage',
        'Petroleum and Natural Gas Systems – LNG Import/Export', 'Petrochemical Production',
        'Petroleum Refining', 'Phosphoric Acid Production', 'Pulp and Paper Manufacturing',
        'Silicon Carbide Production', 'Soda Ash Manufacturing', 'Titanium Dioxide Production',
        'Underground Coal Mines', 'Zinc Production', 'Municipal Landfills',
        'Industrial Wastewater Treatment',
        'Manufacture of Electric Transmission and Distribution Equipment',
        'Industrial Waste Landfills'
    ]
    
    for subpart in industry_subparts:
        if subpart in df.columns:
            df[f'is_{subpart.lower().replace(" ", "_")}'] = (df[subpart] == 'Yes').astype(int)
        else:
            print(f"Warning: Column '{subpart}' not found in the dataset. Skipping this column.")
    
    # Additional binary features
    co2_collected_col = 'Is some CO2 collected on-site and used to manufacture other products and therefore not emitted from the affected manufacturing process unit(s)? (as reported under Subpart G or S)'
    co2_transferred_col = 'Is some CO2 reported as emissions from the affected manufacturing process unit(s) under Subpart AA, G or P collected and transferred off-site or injected (as reported under Subpart PP)?'
    continuous_monitoring_col = 'Does the facility employ continuous emissions monitoring? '

    if co2_collected_col in df.columns:
        df['is_co2_collected'] = (df[co2_collected_col] == 'Yes').astype(int)
    if co2_transferred_col in df.columns:
        df['is_co2_transferred'] = (df[co2_transferred_col] == 'Yes').astype(int)
    if continuous_monitoring_col in df.columns:
        df['has_continuous_monitoring'] = (df[continuous_monitoring_col] == 'Yes').astype(int)
    
    # Create state-level features
    state_emissions = df.groupby('State')['total_ghg_emissions'].mean().reset_index()
    state_emissions.columns = ['State', 'state_avg_emissions']
    df = pd.merge(df, state_emissions, on='State', how='left')
    
    # Create industry sector features
    industry_emissions = df.groupby('industry_type')['total_ghg_emissions'].mean().reset_index()
    industry_emissions.columns = ['industry_type', 'industry_avg_emissions']
    df = pd.merge(df, industry_emissions, on='industry_type', how='left')
    
    # Normalize emissions values
    df['log_total_emissions'] = np.log1p(df['total_ghg_emissions'])
    
    return df

def select_and_refine_features(df, target_column, k=20):
    print("Selecting and refining features...")
    
    # Exclude emission-related columns and the target column
    exclude_keywords = ['emissions', 'CO2', 'reported direct', 'GHG']
    feature_columns = [col for col in df.columns if not any(keyword.lower() in col.lower() for keyword in exclude_keywords) and col != target_column]
    
    X = df[feature_columns]
    y = df[target_column]
    
    # Select K best features
    selector = SelectKBest(score_func=f_regression, k=k)
    X_new = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Create interaction terms for numeric features
    X_selected = X[selected_features]
    numeric_features = X_selected.select_dtypes(include=['float64', 'int64']).columns
    for i in range(len(numeric_features)):
        for j in range(i+1, len(numeric_features)):
            feature_name = f'{numeric_features[i]}_{numeric_features[j]}_interaction'
            X_selected[feature_name] = X_selected[numeric_features[i]] * X_selected[numeric_features[j]]
            selected_features.append(feature_name)
    
    # Bin latitude and longitude if present
    if 'latitude' in selected_features:
        X_selected['latitude_bin'] = pd.qcut(X_selected['latitude'], q=10, labels=False)
        selected_features.append('latitude_bin')
    if 'longitude' in selected_features:
        X_selected['longitude_bin'] = pd.qcut(X_selected['longitude'], q=10, labels=False)
        selected_features.append('longitude_bin')
    
    return X_selected, selected_features