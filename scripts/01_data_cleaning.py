import pandas as pd

df = pd.read_csv("data/Reviews.csv")

df.dropna(subset=["Text", "Score"], inplace=True)

df.drop_duplicates(subset=["UserId", "ProductId", "Text"], keep="first", inplace=True)

def get_sentiment(score):
    if score <= 2:
        return "negative"
    elif score == 3:
        return "neutral"
    else:
        return "positive"

df["Sentiment"] = df["Score"].apply(get_sentiment)

df["user_review_count"] = df.groupby("UserId")["UserId"].transform("count")
df["product_review_count"] = df.groupby("ProductId")["ProductId"].transform("count")

df.to_csv("data/cleaned_reviews.csv", index=False)
print("✅ Cleaned data saved.")