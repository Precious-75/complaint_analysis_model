from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load the model
model = joblib.load('complaint_urgency_classifier.joblib')

# Your text cleaning function from your original code
def simple_clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    import re
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
    except:
        pass
    return ' '.join(words)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    # Clean and predict
    cleaned_text = simple_clean_text(text)
    prediction = model.predict([cleaned_text])[0]
    probabilities = model.predict_proba([cleaned_text])[0]
    
    result = {
        'urgency': prediction,
        'confidence': float(max(probabilities)),
        'probabilities': {
            class_label: float(prob) 
            for class_label, prob in zip(model.classes_, probabilities)
        }
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)