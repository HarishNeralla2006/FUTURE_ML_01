# AI-Powered Sales Forecasting Dashboard

## Overview
This project is a predictive analytics dashboard that helps retail businesses forecast their future sales. It uses historical transaction data from a "Superstore", applies machine learning models (Prophet) to predict upcoming trends, and presents insights via a Power BI dashboard.

## Features
- **Data Cleaning & Structure**: Processing raw sales data.
- **Feature Engineering**: Analyzing monthly averages and trends.
- **Time Series Forecasting**: Using Facebook Prophet to predict future sales.
- **Interactive Dashboard**: Visualizing actual vs. forecasted sales in Power BI.

## Project Structure
- `data/`: Contains the raw `superstore.csv` and the generated `forecast_results.csv`.
- `notebooks/`: Jupyter notebooks for EDA and modeling.
- `src/`: Python scripts for data processing and forecasting.
- `requirements.txt`: Python dependencies.
- `POWER_BI_INSTRUCTIONS.md`: Guide to setting up the Power BI dashboard.

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the forecasting model:
   ```bash
   python src/forecast_model.py
   ```
   Or run the Jupyter Notebook in `notebooks/forecast_model.ipynb`.
2. Open Power BI Desktop.
3. Follow the steps in `POWER_BI_INSTRUCTIONS.md` to load `data/forecast_results.csv` and build the dashboard.

## License
MIT
