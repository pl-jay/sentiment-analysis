import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, render_template, send_file
import pandas as pd
from transformers import pipeline
import pickle
import matplotlib.pyplot as plt
import io
import base64
from wordcloud import WordCloud
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)


class BERTSentimentFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        bert_scores = [self.classifier(text)[0] for text in X]
        features = pd.DataFrame(
            {
                "label": [score["label"] for score in bert_scores],
                "score": [score["score"] for score in bert_scores],
            }
        )
        features["label"] = features["label"].map({"POSITIVE": 1, "NEGATIVE": 0})
        return features


def predict_sentiment(reviews):
    with open("./trained_models/bert_enhanced_sentiment_classifier.pkl", "rb") as file:
        model = pickle.load(file)
    predictions = model.predict(reviews)
    sentiment_df = pd.DataFrame({"Review": reviews, "Sentiment": predictions})
    return sentiment_df

def generate_visualizations(df):
    positive_counts = df[df["Sentiment"] == 1]["Sentiment"].count()
    negative_counts = df[df["Sentiment"] == 0]["Sentiment"].count()

    labels = ["Positive", "Negative"]
    sizes = [positive_counts, negative_counts]
    colors = ["#98FB98", "#FA8072"]

    buffer = io.BytesIO()
    plt.figure(figsize=(8, 4))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    plt.axis("equal")
    plt.title("Sentiment Distribution")
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    pie_chart = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()

    positive_reviews = " ".join(df[df["Sentiment"] == 1]["Review"])
    negative_reviews = " ".join(df[df["Sentiment"] == 0]["Review"])

    wordcloud_positive = WordCloud(
        width=400,
        height=200,
        random_state=21,
        max_font_size=110,
        background_color="white",
    ).generate(positive_reviews)
    wordcloud_negative = WordCloud(
        width=400,
        height=200,
        random_state=21,
        max_font_size=110,
        background_color="white",
    ).generate(negative_reviews)

    buffer = io.BytesIO()
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud_positive, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    wordcloud_positive = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()

    buffer = io.BytesIO()
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud_negative, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    wordcloud_negative = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()

    return wordcloud_positive, wordcloud_negative, pie_chart

@app.route("/download")
def download_file():
    return send_file("./outputs/output_with_sentiment.csv", as_attachment=True)

@app.route("/index", methods=["GET", "POST"])
def my_form_post():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("form.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("form.html", error="Empty file uploaded")

        if not (file.filename.endswith(".csv") or file.filename.endswith(".tsv")):
            return render_template(
                "form.html",
                error="Invalid file format. Please upload a CSV or TSV file.",
            )
        try:
            df = pd.read_csv(file, sep="\t", usecols=["Review"])
            sentiment_df = predict_sentiment(df["Review"])
            df["Sentiment"] = sentiment_df["Sentiment"]
            wordcloud_positive, wordcloud_negative, pie_chart = generate_visualizations(
                df
            )

            df.to_csv("./outputs/output_with_sentiment.csv", index=False, sep="\t")

            return render_template(
                "form.html",
                pie_chart=pie_chart,
                wordcloud_positive=wordcloud_positive,
                wordcloud_negative=wordcloud_negative,
            )
        except Exception as e:
            return render_template(
                "form.html", error=f"Error processing file: {str(e)}"
            )

    return render_template("form.html")

if __name__ == "__main__":
    app.run(debug=True)