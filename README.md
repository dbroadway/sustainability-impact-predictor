# Sustainability Impact Predictor

## Overview

The Sustainability Impact Predictor is a machine learning project that aims to predict the environmental impact of various business activities, specifically focusing on CO2 emissions. This project uses data from the EPA's Greenhouse Gas Reporting Program (GHGRP) to train models that can forecast CO2 emissions based on various factors.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Data](#data)
5. [Models](#models)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

## Project Structure

```
sustainability-impact-predictor/
│
├── data/
│   ├── raw/
│   │   └── ghgrp_data_2022.csv
│   └── processed/
│       └── feature_engineered_data.csv
│
├── models/
│   ├── best_model.joblib
│   ├── preprocessor.joblib
│   ├── random_forest_feature_importance.csv
│   ├── gradient_boosting_feature_importance.csv
│   ├── random_forest_feature_importance.png
│   ├── gradient_boosting_feature_importance.png
│   └── residual_plot.png
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── train_models.py
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/sustainability-impact-predictor.git
   cd sustainability-impact-predictor
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Data Preprocessing:
   ```
   python src/data_preprocessing.py
   ```

2. Feature Engineering:
   ```
   python src/feature_engineering.py
   ```

3. Train Models:
   ```
   python src/train_models.py
   ```

4. For exploratory data analysis, open the Jupyter notebook:
   ```
   jupyter notebook notebooks/exploratory_data_analysis.ipynb
   ```

## Data

This project uses data from the EPA's Greenhouse Gas Reporting Program (GHGRP). The raw data can be found in `data/raw/ghgrp_data_2022.csv`. After preprocessing and feature engineering, the processed data is stored in `data/processed/feature_engineered_data.csv`.

To obtain the raw data:
1. Visit https://www.epa.gov/ghgreporting/ghg-reporting-program-data-sets
2. Navigate to the "2022 Data" section
3. Download the "2022 Data Summary Spreadsheets (zip)" file
4. Extract the contents and place the main CSV file in the `data/raw/` directory

## Models

We train and compare two models:
1. Random Forest Regressor
2. Gradient Boosting Regressor

The best performing model is saved as `models/best_model.joblib`. The data preprocessor is saved as `models/preprocessor.joblib`.

## Results

After training, the following results are generated:
- Feature importance plots: `models/random_forest_feature_importance.png` and `models/gradient_boosting_feature_importance.png`
- Feature importance data: `models/random_forest_feature_importance.csv` and `models/gradient_boosting_feature_importance.csv`
- Residual plot: `models/residual_plot.png`

Model performance metrics, including R2 score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE), are printed to the console during training.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
