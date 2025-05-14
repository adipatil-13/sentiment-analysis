import pandas as pd
from transformers import pipeline, T5Tokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import boto3
import json

# Emotion Detection
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# Sarcasm Detection
sarcasm_model_name = "mrm8488/t5-base-finetuned-sarcasm-twitter"
tokenizer = T5Tokenizer.from_pretrained(sarcasm_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(sarcasm_model_name)
sarcasm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# AWS Comprehend Sentiment Analysis
comprehend = boto3.client("comprehend", region_name="us-east-1")  # Adjust region if needed

def comprehend_sentiment(text):
    try:
        # Call comprehend to analyze sentiment
        response = comprehend.detect_sentiment(Text=text, LanguageCode="en")
        sentiment = response['Sentiment']
        return sentiment
    except Exception as e:
        return "error"

# Sarcasm detection function
def detect_sarcasm(text):
    try:
        if not text.strip():
            return "unknown"
        prompt = f"sarcasm detection: {text}"
        result = sarcasm_pipeline(prompt, max_length=2)[0]['generated_text']
        return result.strip()
    except Exception as e:
        return "error"

# Load data
df = pd.read_csv("data/with_model_sentiments.csv")

# Add progress bar and apply emotion detection
tqdm.pandas(desc="Detecting Emotion")
df["emotion"] = df["Text"].progress_apply(lambda x: emotion_pipeline(str(x)[:512])[0][0]["label"])

# Apply sarcasm detection with progress bar
tqdm.pandas(desc="Detecting Sarcasm")
df["sarcasm"] = df["Text"].progress_apply(lambda x: detect_sarcasm(str(x)[:512]))

# Apply Comprehend sentiment analysis with progress bar
tqdm.pandas(desc="Detecting Comprehend Sentiment")
df["comprehend_sentiment"] = df["Text"].progress_apply(lambda x: comprehend_sentiment(str(x)[:512]))

# Save output
df.to_csv("data/with_emotion_sarcasm_comprehend.csv", index=False)
print("✅ Emotion, Sarcasm, and Comprehend sentiment detection completed.")
