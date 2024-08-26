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