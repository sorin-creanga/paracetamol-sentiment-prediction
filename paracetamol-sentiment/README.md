# 💊 Paracetamol Sentiment Analysis

A machine learning project that analyzes YouTube comments about paracetamol and predicts sentiment trends for the next year.

## Features

- 🎬 Scrapes 1000+ YouTube videos for paracetamol comments
- 📊 Analyzes sentiment (positive/negative/neutral) using VADER
- 🔮 Predicts sentiment for next 365 days using ARIMA
- 📈 Interactive Streamlit dashboard with visualizations

## Project Structure
```
paracetamol-sentiment/
├── dashboard.py              # Main Streamlit app
├── youtube_scraper_fixed.py # YouTube comment scraper
├── sentiment_analyzer.py    # VADER sentiment analysis
├── predict_sentiment.py     # ARIMA forecasting
├── requirements.txt         # Dependencies
├── .gitignore              # Git ignore file
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/paracetamol-sentiment.git
cd paracetamol-sentiment
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```
YOUTUBE_API_KEY=your_api_key_here
```

## Usage

Run the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

Then visit: `http://localhost:8501`

## Data Processing Pipeline

1. **Scrape**: `youtube_scraper_fixed.py` - Fetch comments from YouTube
2. **Analyze**: `sentiment_analyzer.py` - Score sentiment (VADER)
3. **Predict**: `predict_sentiment.py` - Forecast next year (ARIMA)
4. **Visualize**: `dashboard.py` - Interactive Streamlit dashboard

## Results

- **Total Comments Analyzed**: 1,274
- **Sentiment Distribution**:
  - Positive: 192 (15%)
  - Negative: 176 (14%)
  - Neutral: 906 (71%)
- **Average Sentiment Score**: 0.XX

## Dashboard Pages

### 📈 Overview
- Key metrics (total, positive, negative)
- Sentiment distribution pie chart

### 🔍 Analysis
- Sentiment score histogram
- Sample comments by category

### 🔮 Predictions
- 365-day sentiment forecast
- Confidence intervals
- Forecast statistics

## Technologies

- **Python 3.10+**
- **Streamlit** - Interactive dashboard
- **Plotly** - Visualizations
- **NLTK/VADER** - Sentiment analysis
- **Statsmodels** - ARIMA forecasting
- **Pandas** - Data processing
- **Google API** - YouTube data

## Author

Your Name - [GitHub Profile](https://github.com/YOUR_USERNAME)

## License

MIT License - feel free to use this project

## Future Improvements

- [ ] Add more data sources (Reddit, Twitter)
- [ ] Implement transformer models (DistilBERT)
- [ ] Deploy to Streamlit Cloud
- [ ] Add database (PostgreSQL)
- [ ] Create REST API (FastAPI)