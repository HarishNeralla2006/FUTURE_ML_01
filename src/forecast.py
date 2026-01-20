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
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding='latin1')

    # 2. Preprocessing
    # Map columns to standard names
    # Found columns: 'Order Date', 'Sales', 'Region', 'Product Category'
    date_col = 'Order Date'
    sales_col = 'Sales'
    category_col = 'Product Category'
    region_col = 'Region'

    # Verify columns exist
    required_cols = [date_col, sales_col, category_col, region_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing columns {missing}. Available: {df.columns.tolist()}")
        return

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.dropna(subset=[date_col], inplace=True)

    # 3. Training & Forecasting Loop
    final_results = []
    
    # Get unique combinations of Region and Category
    combinations = df[[region_col, category_col]].drop_duplicates().values
    
    print(f"Found {len(combinations)} segments. Starting training...")

    for region, category in combinations:
        print(f"Training for Region: {region} | Category: {category}")
        
        # Filter data
        input_data = df[(df[region_col] == region) & (df[category_col] == category)].copy()
        
        # Aggregate by Month
        monthly = input_data.groupby(pd.Grouper(key=date_col, freq='M'))[sales_col].sum().reset_index()
        monthly.columns = ['ds', 'y']
        
        if len(monthly) < 2:
            print(f"Skipping {region}-{category}: Not enough data.")
            continue

        # Train Model
        # Daily/Weekly might fail on small segments, stick to yearly if enough data, else simple
        try:
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.fit(monthly)
            
            # Forecast
            future = model.make_future_dataframe(periods=12, freq='M')
            forecast = model.predict(future)
            
            # Formatting Output
            output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            output.rename(columns={'ds': 'Date', 'yhat': 'Predicted_Sales', 'yhat_lower': 'Lower_Bound', 'yhat_upper': 'Upper_Bound'}, inplace=True)
            
            # Merge Actuals
            actuals = monthly.rename(columns={'ds': 'Date', 'y': 'Actual_Sales'})
            segment_df = pd.merge(output, actuals, on='Date', how='left')
            
            # Add Dimension Columns
            segment_df['Region'] = region
            segment_df['Category'] = category
            
            final_results.append(segment_df)
            
        except Exception as e:
            print(f"Failed for {region}-{category}: {e}")

    # 4. Concatenate and Export
    if final_results:
        print("Exporting results...")
        full_df = pd.concat(final_results, ignore_index=True)
        
        # Reorder columns
        cols = ['Date', 'Region', 'Category', 'Actual_Sales', 'Predicted_Sales', 'Lower_Bound', 'Upper_Bound']
        full_df = full_df[cols]
        
        output_path = 'data/forecast_results.csv'
        full_df.to_csv(output_path, index=False)
        print(f"Detailed forecast saved to {output_path}")
    else:
        print("No forecasts were generated.")

if __name__ == "__main__":
    main()
