import pickle
import re
from joblib import dump
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\W+", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    text = " ".join(words)
    return text


class TextPreprocessor(TransformerMixin):
    def transform(self, X, *_):
        return [preprocess_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self


file_path = "./datasets/RestaurantReviews_TrainingData.tsv"
df = pd.read_csv(file_path, sep="\t")

X_train, X_test, y_train, y_test = train_test_split(
    df["Review"], df["Sentiment"], test_size=0.25, random_state=42
)

pipeline = Pipeline(
    [
        ("text_preprocessor", TextPreprocessor()),
        ("vectorizer", CountVectorizer()),
        ("classifier", LogisticRegression()),
    ]
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

with open("./trained_models/wordlemitizer_model.pkl", "wb") as file:
    pickle.dump(pipeline, file)