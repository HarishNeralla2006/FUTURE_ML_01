import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

def main():
    # 1. Load Data
    data_path = 'data/superstore.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please ensure the data is downloaded.")
        return

    print("Loading data...")
    # Try reading with different encodings if default fails
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding='latin1')

    # 2. Preprocessing
    print("Preprocessing...")
    # Ensure columns vary by dataset source; adapt as needed
    # Common columns: 'Order Date', 'Sales'
    # Check if 'Order Date' exists, if not specific to this dataset, might be 'Order_Date'
    date_col = 'Order Date'
    sales_col = 'Sales'
    
    if date_col not in df.columns:
        print(f"Warning: '{date_col}' not found. Available columns: {df.columns}")
        # Attempt to find a date column
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break
        print(f"Using '{date_col}' as date column.")

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.dropna(subset=[date_col], inplace=True)
    
    # Aggregating sales by month
    # Prophet requires columns 'ds' (Date) and 'y' (Value)
    monthly_sales = df.groupby(pd.Grouper(key=date_col, freq='M'))[sales_col].sum().reset_index()
    monthly_sales.columns = ['ds', 'y']
    
    print(f"Data aggregated. Shape: {monthly_sales.shape}")
    print(monthly_sales.head())

    # 3. Model Training
    print("Training Prophet model...")
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(monthly_sales)

    # 4. Forecasting
    print("Generating forecast...")
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)

    # 5. Export Results
    print("Exporting results...")
    # Prepare output dataframe: Date, Actual, Predicted, Lower, Upper
    # Join forecast with actuals
    
    output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    output.rename(columns={'ds': 'Date', 'yhat': 'Predicted_Sales', 'yhat_lower': 'Lower_Bound', 'yhat_upper': 'Upper_Bound'}, inplace=True)
    
    # Merge actuals
    actuals = monthly_sales.rename(columns={'ds': 'Date', 'y': 'Actual_Sales'})
    final_df = pd.merge(output, actuals, on='Date', how='left')
    
    # Fill NaN actuals (future dates) with suitable value or leave as NaN (Power BI handles NaN)
    # Reorder columns
    final_df = final_df[['Date', 'Actual_Sales', 'Predicted_Sales', 'Lower_Bound', 'Upper_Bound']]
    
    output_path = 'data/forecast_results.csv'
    final_df.to_csv(output_path, index=False)
    print(f"Forecast saved to {output_path}")

    # Optional: Save a plot
    fig1 = model.plot(forecast)
    plt.savefig('data/forecast_plot.png')
    print("Forecast plot saved to data/forecast_plot.png")

if __name__ == "__main__":
    main()
