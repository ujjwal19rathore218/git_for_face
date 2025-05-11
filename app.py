from flask import Flask, request, jsonify
from transformers import pipeline
from PIL import Image
import io

app = Flask(__name__)

emotion_classifier = pipeline(
    "image-classification", 
    model="trpakov/vit-face-expression", 
    device=-1
)

@app.route("/")
def home():
    return "Emotion Detection API is Running"

@app.route("/classify_emotion", methods=["POST"])
def classify_emotion():
    try:
        file = request.files["file"]
        image = Image.open(file.stream).convert("RGB")
        result = emotion_classifier(image)
        emotion = result[0]["label"].lower()
        return jsonify({"emotion": emotion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
