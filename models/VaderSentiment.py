import pickle
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import FunctionTransformer

nltk.download("vader_lexicon")


# Define a custom transformer to extract VADER sentiment features
class VaderSentimentFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Applying VADER to compute sentiment scores
        features = pd.DataFrame([self.sia.polarity_scores(text) for text in X])
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
        (
            "features",
            FeatureUnion(
                [
                    (
                        "tfidf",
                        Pipeline(
                            [
                                (
                                    "selector",
                                    FunctionTransformer(get_text, validate=False),
                                ),
                                (
                                    "tfidf_vector",
                                    TfidfVectorizer(
                                        min_df=5, max_df=0.8, ngram_range=(1, 2)
                                    ),
                                ),
                            ]
                        ),
                    ),
                    ("vader", VaderSentimentFeature()),
                ]
            ),
        ),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

with open("./trained_models/enhanced_sentiment_classifier.pkl", "wb") as file:
    pickle.dump(pipeline, file)