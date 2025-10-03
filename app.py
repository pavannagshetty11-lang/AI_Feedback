from flask import Flask, render_template, request
from transformers import pipeline
import random

app = Flask(__name__)

# Load emotion model
classifier = pipeline("text-classification",
                      model="j-hartmann/emotion-english-distilroberta-base",
                      return_all_scores=True)

# Predefined random thank-you messages
positive_responses = [
    "😊 Thank you! Your feedback is much appreciated.",
    "🌟 We’re glad you enjoyed the class!",
    "🙌 Thanks! It’s wonderful to know you understood well.",
    "🎉 Awesome! Keep learning with the same energy.",
    "💡 Great! Your positive energy motivates us."
]

negative_responses = [
    "🙏 Thank you for your feedback, we will try to improve.",
    "💭 We understand you had difficulties, we will clarify further.",
    "📘 Sorry it felt confusing, we’ll simplify it next time.",
    "🤝 Thanks for being honest, we’ll take extra care on this topic.",
    "🔍 Your feedback matters! We’ll work on making it clearer."
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/feedback", methods=["POST"])
def feedback():
    text = request.form["feedback"]

    # Detect emotions
    results = classifier(text)[0]
    top_emotion = max(results, key=lambda x: x["score"])
    label = top_emotion["label"]

    # Pick a random nice message
    if label.lower() in ["joy", "positive"]:
        message = random.choice(positive_responses)
    elif label.lower() in ["sadness", "fear", "anger", "negative"]:
        message = random.choice(negative_responses)
    else:
        message = f"🙏 Thank you! We detected your feedback as {label}."

    # Save feedback
    with open("feedback_responses.txt", "a", encoding="utf-8") as f:
        f.write(f"Feedback: {text} | Detected Emotion: {label}\n")

    return render_template("index.html", message=message)

@app.route("/report")
def report():
    try:
        with open("feedback_responses.txt", "r", encoding="utf-8") as f:
            data = f.readlines()
    except FileNotFoundError:
        data = ["No feedback yet."]
    return render_template("report.html", feedbacks=data)

if __name__ == "__main__":
    app.run(debug=True)
