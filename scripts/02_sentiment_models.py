import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import boto3

# Load cleaned data
df = pd.read_csv("data/cleaned_reviews.csv")

# VADER
vader = SentimentIntensityAnalyzer()
df['vader_score'] = df['Text'].apply(lambda x: vader.polarity_scores(str(x))['compound'])
df['vader_sentiment'] = df['vader_score'].apply(lambda x: 'positive' if x >= 0.05 else 'negative' if x <= -0.05 else 'neutral')

# Amazon Comprehend (subset)
try:
    comprehend = boto3.client("comprehend", region_name="us-east-1")
    df_comprehend = df.sample(1000)
    df_comprehend['comprehend_sentiment'] = df_comprehend['Text'].apply(
        lambda x: comprehend.detect_sentiment(Text=str(x)[:5000], LanguageCode='en')['Sentiment']
    )
    df_comprehend.to_csv("data/comprehend_sentiment.csv", index=False)
    print("✅ Amazon Comprehend sentiments saved (subset).")
except Exception as e:
    print(f"⚠️ Amazon Comprehend failed: {e}")

# BERT (subset)
bert_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

df_bert = df.sample(1000).copy()
df_bert['bert_sentiment'] = df_bert['Text'].apply(
    lambda x: bert_pipeline(str(x)[:512])[0]['label'].lower()
)

df_bert.to_csv("data/with_model_sentiments.csv", index=False)
print("✅ Sampled BERT predictions saved.")
