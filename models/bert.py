import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import pipeline
import pickle


class BERTSentimentFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Load the sentiment-analysis pipeline from Hugging Face's Transformers
        self.classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply the BERT model to compute sentiment scores and labels
        bert_scores = [self.classifier(text)[0] for text in X]
        # Extract labels and scores into separate features
        features = pd.DataFrame(
            {
                "label": [score["label"] for score in bert_scores],
                "score": [score["score"] for score in bert_scores],
            }
        )
        # Convert labels to numeric format if necessary
        features["label"] = features["label"].map({"POSITIVE": 1, "NEGATIVE": 0})
        return features


def get_text(data):
    return data


file_path = "./datasets/RestaurantReviews_TrainingData.tsv"
df = pd.read_csv(file_path, sep="\t")


X_train, X_test, y_train, y_test = train_test_split(
    df["Review"], df["Sentiment"], test_size=0.25, random_state=0
)

pipeline = Pipeline(
    [
        ("features", FeatureUnion([("bert_sentiment", BERTSentimentFeature())])),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# Train the model
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

with open("./trained_models/bert_enhanced_sentiment_classifier.pkl", "wb") as file:
    pickle.dump(pipeline, file)
