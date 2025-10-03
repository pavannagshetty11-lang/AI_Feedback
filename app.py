from flask import Flask, render_template, request
from transformers import pipeline
import os
import csv
from datetime import datetime

# Initialize Flask
app = Flask(__name__)

# Initialize sentiment-analysis pipeline (small, fast model)
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Route for feedback form
@app.route("/", methods=["GET", "POST"])
def feedback():
    message = ""
    if request.method == "POST":
        text = request.form["feedback"]
        if text.strip() != "":
            # Analyze sentiment
            result = classifier(text)[0]  # result = {'label': 'POSITIVE'/'NEGATIVE', 'score': 0.9}
            label = result["label"]
            
            # Determine message
            if label == "POSITIVE":
                message = "Thank you! Your feedback is much appreciated."
            else:
                message = "Thank you! We will try to improve based on your feedback."

            # Save feedback in CSV
            csv_file = "feedback_responses.csv"
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now(), text, label, message])

    return render_template("index.html", message=message)

# Run app on Render's port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
