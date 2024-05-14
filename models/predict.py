from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the model
model = load("./trained_models/bert_enhanced_sentiment_classifier.pkl")

new_data_path = "./datasets/RestaurantReviews_FreshDump.tsv"  # Replace with your file path
new_reviews_df = pd.read_csv(
    new_data_path, sep="\t", usecols=["Review"]
) 
# Make predictions
predictions = model.predict(new_reviews_df["Review"])


if "Actual Sentiment" in new_reviews_df.columns:
    accuracy = accuracy_score(new_reviews_df["Actual Sentiment"], predictions)
    print("Accuracy of the model on new data:", accuracy)

# Output predictions
print(predictions)

new_reviews_df["Predicted Sentiment"] = predictions

new_reviews_df.to_csv("./outputs/predicted_reviews.tsv", sep="\t", index=False)
