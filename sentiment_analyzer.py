from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import nltk
import os

# Download required data
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')

print("Loading comments...\n")

# Load the CSV 
df = pd.read_csv(r"C:\Users\sorin.creanga\Desktop\Paracetamol_sentiment_predictor\paracetamol-sentiment\youtube_comments.csv")

print(f"Analyzing sentiment for {len(df)} comments...\n")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Analyze each comment
sentiments = []

for idx, comment in enumerate(df['text'], 1):
    try:
        # Get sentiment score
        scores = sia.polarity_scores(comment)
        sentiment_score = scores['compound']  # -1 to 1
        
        # Categorize
        if sentiment_score > 0.33:
            category = 'positive'
        elif sentiment_score < -0.33:
            category = 'negative'
        else:
            category = 'neutral'
        
        sentiments.append({
            'sentiment_score': sentiment_score,
            'sentiment_category': category
        })
        
        if idx % 100 == 0:
            print(f"✓ Analyzed {idx}/{len(df)} comments")
    
    except Exception as e:
        print(f"Error analyzing comment {idx}")
        sentiments.append({
            'sentiment_score': 0,
            'sentiment_category': 'neutral'
        })

# Add sentiment columns to dataframe
sentiment_df = pd.DataFrame(sentiments)
df_with_sentiment = pd.concat([df, sentiment_df], axis=1)

# Create folder
os.makedirs("data/processed", exist_ok=True)

# Save to CSV
output_file = r"C:\Users\sorin.creanga\Desktop\Paracetamol_sentiment_predictor\paracetamol-sentiment\data\processed\youtube_comments_sentiment.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df_with_sentiment.to_csv(output_file, index=False)

print(f"\n✓ Saved to {output_file}")
print("\nSentiment Distribution:")
print(df_with_sentiment['sentiment_category'].value_counts())

print("\nSample analyzed comments:")
for i in range(min(5, len(df_with_sentiment))):
    row = df_with_sentiment.iloc[i]
    print(f"\n{i+1}. {row['text'][:80]}")
    print(f"   Score: {row['sentiment_score']:.3f} ({row['sentiment_category']})")
