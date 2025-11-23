import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Set page config
st.set_page_config(
    page_title="Paracetamol Sentiment Analysis",
    page_icon="💊",
    layout="wide"
)

# Title
st.title("Paracetamol Sentiment Analysis Dashboard")
st.markdown("YouTube comments sentiment analysis with ML predictions")

# --- DATA LOADING SECTION ---

@st.cache_data
def load_data():
    """
    Loads data from CSV files. 
    Tries to load from the root directory first. 
    If not found, falls back to the 'data/' subdirectories.
    """
    # Define file names
    comments_file = 'youtube_comments_sentiment.csv'
    forecast_file = 'sentiment_forecast.csv'
    
    # 1. Try loading Comments Data
    if os.path.exists(comments_file):
        df_comments = pd.read_csv(comments_file)
    else:
        # Fallback to original structure if file not found in root
        fallback_path = os.path.join('data', 'processed', comments_file)
        if os.path.exists(fallback_path):
            df_comments = pd.read_csv(fallback_path)
        else:
            raise FileNotFoundError(f"Could not find {comments_file} in root or data/processed/")

    # 2. Try loading Forecast Data
    if os.path.exists(forecast_file):
        df_forecast = pd.read_csv(forecast_file)
    else:
        # Fallback to original structure if file not found in root
        fallback_path = os.path.join('data', 'predictions', forecast_file)
        if os.path.exists(fallback_path):
            df_forecast = pd.read_csv(fallback_path)
        else:
            raise FileNotFoundError(f"Could not find {forecast_file} in root or data/predictions/")
            
    return df_comments, df_forecast

# Load the data using the single wrapper function
try:
    comments_df, forecast_df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please ensure 'youtube_comments_sentiment.csv' and 'sentiment_forecast.csv' are uploaded to your GitHub repository.")
    st.stop()

# --- DASHBOARD INTERFACE ---

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Select Page:", ["Overview", "Analysis", "Predictions"])

# PAGE 1: OVERVIEW
if page == "Overview":
    st.header("📈 Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_comments = len(comments_df)
    positive = len(comments_df[comments_df['sentiment_category'] == 'positive'])
    negative = len(comments_df[comments_df['sentiment_category'] == 'negative'])
    neutral = len(comments_df[comments_df['sentiment_category'] == 'neutral'])
    avg_sentiment = comments_df['sentiment_score'].mean()
    
    with col1:
        st.metric("Total Comments", total_comments)
    with col2:
        st.metric("Positive", positive, f"{(positive/total_comments*100):.1f}%")
    with col3:
        st.metric("Negative", negative, f"{(negative/total_comments*100):.1f}%")
    with col4:
        st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
    
    st.markdown("---")
    
    # Sentiment distribution pie chart
    st.subheader("Sentiment Distribution")
    
    sentiment_counts = comments_df['sentiment_category'].value_counts()
    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        marker=dict(colors=[colors.get(x, '#999') for x in sentiment_counts.index]),
        textposition='auto',
        textinfo='label+percent'
    )])
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

# PAGE 2: ANALYSIS
elif page == "Analysis":
    st.header("🔍 Sentiment Analysis")
    
    # Sentiment score distribution
    st.subheader("Sentiment Score Distribution")
    
    fig_hist = go.Figure(data=[
        go.Histogram(x=comments_df['sentiment_score'], nbinsx=50, marker_color='#3498db')
    ])
    fig_hist.update_layout(
        title="Distribution of Sentiment Scores",
        xaxis_title="Sentiment Score (-1 to 1)",
        yaxis_title="Number of Comments",
        height=400
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    # Sample comments by category
    st.subheader("Sample Comments by Sentiment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Positive Comments**")
        positive_samples = comments_df[comments_df['sentiment_category'] == 'positive']['text'].head(3)
        for i, comment in enumerate(positive_samples, 1):
            st.write(f"{i}. {comment[:100]}...")
    
    with col2:
        st.write("**Neutral Comments**")
        neutral_samples = comments_df[comments_df['sentiment_category'] == 'neutral']['text'].head(3)
        for i, comment in enumerate(neutral_samples, 1):
            st.write(f"{i}. {comment[:100]}...")
    
    with col3:
        st.write("**Negative Comments**")
        negative_samples = comments_df[comments_df['sentiment_category'] == 'negative']['text'].head(3)
        for i, comment in enumerate(negative_samples, 1):
            st.write(f"{i}. {comment[:100]}...")

# PAGE 3: PREDICTIONS
elif page == "Predictions":
    st.header("Sentiment Predictions (Next 365 Days)")
    
    # Parse forecast data
    forecast_df.index = pd.date_range(start=datetime.now(), periods=len(forecast_df), freq='D')
    
    # Forecast chart
    fig_forecast = go.Figure()
    
    # Add forecast line
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='#e74c3c', width=2)
    ))
    
    # Add confidence interval
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['upper_ci'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['lower_ci'],
        fillcolor='rgba(231, 76, 60, 0.2)',
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='95% Confidence Interval'
    ))
    
    fig_forecast.update_layout(
        title="Sentiment Forecast for Next 12 Months",
        xaxis_title="Date",
        yaxis_title="Predicted Sentiment Score",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.markdown("---")
    
    # Forecast statistics
    st.subheader("Forecast Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Forecast", f"{forecast_df['forecast'].mean():.3f}")
    with col2:
        st.metric("Max Forecast", f"{forecast_df['forecast'].max():.3f}")
    with col3:
        st.metric("Min Forecast", f"{forecast_df['forecast'].min():.3f}")
    with col4:
        st.metric("Std Dev", f"{forecast_df['forecast'].std():.3f}")
    
    st.markdown("---")
    
    # Show first 30 days of forecast
    st.subheader("Next 30 Days Forecast (Details)")
    st.dataframe(forecast_df.head(30).round(4), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center'><small>Paracetamol Sentiment Analysis Dashboard | YouTube Comments Analysis</small></div>", unsafe_allow_html=True)
