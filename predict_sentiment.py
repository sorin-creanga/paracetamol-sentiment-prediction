import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import os

print("Loading sentiment data...\n")

# Load data
df = pd.read_csv(r"C:\Users\sorin.creanga\Desktop\Paracetamol_sentiment_predictor\paracetamol-sentiment\data\processed\youtube_comments_sentiment.csv")


df['published_at'] = pd.to_datetime(df['published_at'])


daily_sentiment = df.groupby(df['published_at'].dt.date)['sentiment_score'].mean()
daily_sentiment.index = pd.to_datetime(daily_sentiment.index)

print(f"Daily sentiment data points: {len(daily_sentiment)}\n")
print("Training ARIMA model for prediction...\n")

# Train ARIMA model
try:
    model = ARIMA(daily_sentiment, order=(1, 1, 1))
    fitted_model = model.fit()
    
    print("✓ Model trained successfully!\n")
    print("Model Summary:")
    print(fitted_model.summary())
    
    # Forecast for next 365 days (1 year)
    print("\n\nForecasting sentiment for next 365 days...\n")
    forecast = fitted_model.get_forecast(steps=365)
    forecast_df = forecast.conf_int(alpha=0.05)
    forecast_df.columns = ['lower_ci', 'upper_ci']
    forecast_df['forecast'] = forecast.predicted_mean
    
    # Create folder
    os.makedirs("data/predictions", exist_ok=True)
    
   
    output_file = r"C:\Users\sorin.creanga\Desktop\Paracetamol_sentiment_predictor\paracetamol-sentiment\data\predictions\sentiment_forecast.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    forecast_df.to_csv(output_file)
    
    print(f"✓ Saved forecast to {output_file}\n")
    
    print("Next 30 days forecast:")
    print(forecast_df.head(30))
    
    print("\n\n✓ PROJECT COMPLETE!")
    print("=" * 60)
    print("Summary:")
    print(f"  - Scraped: 1274 YouTube comments")
    print(f"  - Analyzed: Sentiment (positive/negative/neutral)")
    print(f"  - Predicted: Sentiment for next 365 days")
    print(f"  - Distribution: {len(df[df['sentiment_category']=='positive'])} positive, {len(df[df['sentiment_category']=='negative'])} negative, {len(df[df['sentiment_category']=='neutral'])} neutral")
    print("=" * 60)

except Exception as e:
    print(f"Error: {e}")
    print("\nTrying simpler model...")
    
   
    model = ARIMA(daily_sentiment, order=(1, 0, 0))
    fitted_model = model.fit()
    forecast = fitted_model.get_forecast(steps=365)
    forecast_df = forecast.conf_int(alpha=0.05)
    forecast_df.columns = ['lower_ci', 'upper_ci']
    forecast_df['forecast'] = forecast.predicted_mean
    
    output_file = r"C:\Users\sorin.creanga\Desktop\Paracetamol_sentiment_predictor\paracetamol-sentiment\data\predictions\sentiment_forecast.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    forecast_df.to_csv(output_file)
    
    print(f"✓ Saved forecast to {output_file}")
