import streamlit as st
import pandas as pd
import plotly.express as px
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set Streamlit layout
st.set_page_config(layout="wide", page_title="Sentiment Dashboard", page_icon="💬")
st.title("Amazon Fine Food Reviews Dataset - Sentiment Analysis Visualization Dashboard")

# Load data
df = pd.read_csv("data/with_emotion_sarcasm_comprehend.csv")

# Load metrics
with open("results/model_metrics.json") as f:
    metrics = json.load(f)

# Sidebar filters
st.sidebar.header("🎛️ Filters")
emotions = st.sidebar.multiselect("🎭 Filter by Emotion", df["emotion"].unique())
if emotions:
    df = df[df["emotion"].isin(emotions)]


for model_name, label in [
    ("vader_sentiment", "VADER"),
    ("bert_sentiment", "BERT"),
    ("comprehend_sentiment", "Amazon Comprehend")
]:
    st.subheader(f"🔍 {label} Sentiment Distribution")
    counts = df[model_name].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    fig = px.bar(counts, x="Sentiment", y="Count", color="Sentiment", title=f"{label} Sentiment")
    st.plotly_chart(fig, use_container_width=True)

# --- Model Performance Metrics ---
st.header("📈 Model Performance Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("VADER")
    st.json(metrics["vader_sentiment"])
with col2:
    st.subheader("BERT")
    st.json(metrics["bert_sentiment"])
with col3:
    st.subheader("Comprehend")
    st.json(metrics["comprehend_sentiment"])

# --- Agreement Analysis ---
st.header("Sentiment Agreement Between Models")
st.write("VADER vs BERT")
st.dataframe(pd.crosstab(df["vader_sentiment"], df["bert_sentiment"]))

st.write("VADER vs Comprehend")
st.dataframe(pd.crosstab(df["vader_sentiment"], df["comprehend_sentiment"]))

st.write("BERT vs Comprehend")
st.dataframe(pd.crosstab(df["bert_sentiment"], df["comprehend_sentiment"]))

# --- Emotion Distribution ---
st.header("🎨 Emotion Distribution")
emotion_counts = df["emotion"].value_counts().reset_index()
emotion_counts.columns = ["Emotion", "Count"]  # Correct column names
fig_emotion = px.bar(emotion_counts, x="Emotion", y="Count",
                     color="Emotion", title="Emotion Distribution in Reviews")
st.plotly_chart(fig_emotion, use_container_width=True)

# --- Sarcasm Comparison ---
st.header("🧠 Sentiment in Sarcastic vs Normal Reviews")
for model in ["vader_sentiment", "bert_sentiment", "comprehend_sentiment"]:
    st.subheader(f"Sentiment by {model.upper()}")
    sarcasm_sentiment = pd.crosstab(df["sarcasm"], df[model])
    st.bar_chart(sarcasm_sentiment)

# --- Search ---
st.header("🔍 Explore Reviews")
search_term = st.text_input("Search in reviews:")
if search_term:
    st.dataframe(df[df['Text'].str.contains(search_term, case=False)][["Summary", "Text", "ProductId", "UserId"]],
                 use_container_width=True)

st.header("🧪 Explore Reviews by Product")
selected_product = st.selectbox("Select ProductId", df["ProductId"].unique())
filtered_reviews = df[df["ProductId"] == selected_product]
st.dataframe(filtered_reviews[["Summary", "Text", "vader_sentiment", "emotion"]],
             use_container_width=True)

# --- Top Reviews ---
st.header("🌟 Top Reviews (Based on VADER Score)")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**👍 Top 5 Most Positive Reviews (VADER)**")
    top_positive = df.sort_values("vader_score", ascending=False).head(5)
    top_positive["UserId"] = top_positive["UserId"].astype(str)
    st.dataframe(top_positive[["Summary", "Text", "vader_score", "ProductId", "UserId"]],
                 use_container_width=True, hide_index=True)

with col2:
    st.markdown("**👎 Top 5 Most Negative Reviews (VADER)**")
    top_negative = df.sort_values("vader_score", ascending=True).head(5)
    top_negative["UserId"] = top_negative["UserId"].astype(str)
    st.dataframe(top_negative[["Summary", "Text", "vader_score", "ProductId", "UserId"]],
                 use_container_width=True, hide_index=True)
    
st.header("🔗 Sentiment Distribution")
for model in ["comprehend_sentiment"]:
    st.subheader(f"{model.upper()} Sentiment vs Emotion")
    fig = px.histogram(df, x="emotion", color=model,
                       barmode="group")
    st.plotly_chart(fig, use_container_width=True)

st.header("⭐ Rating vs Sentiment Distribution")
fig = px.box(df, x="comprehend_sentiment", y="Score",
             title="Rating Distribution across VADER Sentiment",
             points="all")
st.plotly_chart(fig, use_container_width=True)
    
st.header("🗣️ Frequent Words in Reviews using Word C;oud")
all_text = " ".join(df["Text"].dropna().values)
wordcloud = WordCloud(width=1000, height=500, background_color="white").generate(all_text)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by Aditya, Arya, Aditya, Shreyash")



