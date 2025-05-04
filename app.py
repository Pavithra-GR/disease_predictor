
import os
from flask import Flask, request, render_template, send_file
import joblib
import numpy as np

# Load trained models
knn_model = joblib.load('knn_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import string
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

nltk.download('averaged_perceptron_tagger_eng')

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def get_wordnet_pos(treebank_tag):
    from nltk.corpus import wordnet
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    words = word_tokenize(text)
    words = [w for w in words if w.lower() not in stop_words]
    pos_tags = pos_tag(words)
    lemmatized = [lemmatizer.lemmatize(w, get_wordnet_pos(pos)) for w, pos in pos_tags]
    cleaned = " ".join(lemmatized).lower()
    cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
    return cleaned

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/nearby-doctors')
def nearby_doctors():
    return render_template("Search-nearby-doctor.html")

@app.route('/chatbot')
def chatbot():
    return render_template("chatbot.html")

@app.route('/how-it-works')
def how_it_works():
    return render_template("how-it-works.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/faq')
def faq():
    return render_template("faq.html")


@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    message = request.form['message']
    return render_template('contact.html', feedback_submitted=True)

@app.route('/styles.css')
def serve_css():
    # Path to styles.css inside the templates folder
    css_path = os.path.join(app.template_folder, 'styles.css')
    return send_file(css_path, mimetype='text/css')

@app.route('/predict', methods=['POST'])
def predict():
    symptom_text = request.form['symptoms']
    cleaned = preprocess_text(symptom_text)
    vectorized = tfidf_vectorizer.transform([cleaned])
    precaution_tips = {
    "Psoriasis": [
        "Keep skin moisturized",
        "Avoid harsh skin products",
        "Manage stress",
        "Avoid triggers like smoking and alcohol"
    ],
    "Varicose Veins": [
        "Avoid standing for long periods",
        "Elevate legs when resting",
        "Exercise regularly",
        "Wear compression stockings"
    ],
    "Typhoid": [
        "Drink boiled water",
        "Eat home-cooked food",
        "Avoid raw vegetables",
        "Complete antibiotic course"
    ],
    "Chicken Pox": [
        "Avoid scratching blisters",
        "Use calamine lotion",
        "Maintain hygiene",
        "Get plenty of rest"
    ],
    "Impetigo": [
        "Wash affected area gently",
        "Avoid close contact with others",
        "Use prescribed antibiotics",
        "Keep nails trimmed"
    ],
    "Dengue": [
        "Avoid mosquito bites",
        "Drink fluids",
        "Avoid NSAIDs",
        "Seek medical attention"
    ],
    "Fungal Infection": [
        "Keep affected area dry",
        "Use antifungal creams",
        "Wear loose clothing",
        "Avoid sharing personal items"
    ],
    "Common Cold": [
        "Stay hydrated",
        "Get plenty of rest",
        "Use steam inhalation",
        "Avoid cold drinks"
    ],
    "Pneumonia": [
        "Complete antibiotic course",
        "Stay hydrated",
        "Avoid smoking",
        "Rest and monitor symptoms"
    ],
    "Dimorphic Hemorrhoids": [
        "Eat high-fiber foods",
        "Stay hydrated",
        "Avoid straining during bowel movements",
        "Exercise regularly"
    ],
    "Arthritis": [
        "Do light exercise",
        "Use hot/cold packs",
        "Take pain relievers as prescribed",
        "Maintain a healthy weight"
    ],
    "Acne": [
        "Wash face twice daily",
        "Avoid oily skincare products",
        "Do not squeeze pimples",
        "Use prescribed acne treatments"
    ],
    "Bronchial Asthma": [
        "Avoid allergens",
        "Use inhalers regularly",
        "Do breathing exercises",
        "Keep environment dust-free"
    ],
    "Hypertension": [
        "Reduce salt intake",
        "Exercise regularly",
        "Avoid stress",
        "Monitor blood pressure"
    ],
    "Migraine": [
        "Avoid bright lights and loud noises",
        "Take prescribed medication",
        "Stay hydrated",
        "Get enough sleep"
    ],
    "Cervical Spondylosis": [
        "Do neck exercises",
        "Maintain proper posture",
        "Avoid lifting heavy weights",
        "Use ergonomic furniture"
    ],
    "Jaundice": [
        "Avoid oily food",
        "Eat fruits and vegetables",
        "Get adequate rest",
        "Avoid alcohol"
    ],
    "Malaria": [
        "Use mosquito nets",
        "Take prescribed antimalarials",
        "Avoid stagnant water",
        "Stay indoors during dusk"
    ],
    "Urinary Tract Infection": [
        "Drink plenty of water",
        "Maintain hygiene",
        "Avoid holding urine",
        "Take antibiotics as prescribed"
    ],
    "Allergy": [
        "Avoid known allergens",
        "Use antihistamines",
        "Keep surroundings clean",
        "Wear a mask if needed"
    ],
    "Gastroesophageal Reflux Disease": [
        "Avoid spicy food",
        "Eat smaller meals",
        "Don’t lie down after eating",
        "Elevate head while sleeping"
    ],
    "Drug Reaction": [
        "Stop the medication immediately",
        "Consult a doctor",
        "Keep a record of allergies",
        "Use prescribed antihistamines"
    ],
    "Peptic Ulcer Disease": [
        "Avoid spicy and acidic foods",
        "Don’t take NSAIDs",
        "Eat smaller meals",
        "Take antacids if prescribed"
    ],
    "Diabetes": [
        "Avoid sugar-rich foods",
        "Regular exercise",
        "Monitor blood sugar",
        "Consult a diabetologist"
    ]
}


    # Get prediction probabilities
    probas = knn_model.predict_proba(vectorized)[0]
    
    # Get class labels
    class_labels = knn_model.classes_
    
    # Pair each label with its probability
    predictions = list(zip(class_labels, probas))
    
    # Sort predictions by probability in descending order
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    
    # Take top 5
    top_predictions = sorted_predictions[:5]
    # After getting top predictions
    most_probable = top_predictions[0][0]  # Top disease (the first item)
    tips = precaution_tips.get(most_probable, ["Consult a doctor for accurate guidance."])

    return render_template('index.html', symptoms=symptom_text, top_predictions=top_predictions, tips=tips)
   # return render_template('index.html', symptoms=symptom_text, top_predictions=top_predictions)

# @app.route('/predict', methods=['POST'])
# def predict():
#     symptom_text = request.form['symptoms']
#     cleaned = preprocess_text(symptom_text)
#     vectorized = tfidf_vectorizer.transform([cleaned])
#     prediction = knn_model.predict(vectorized)[0]

#     return render_template('index.html', symptoms=symptom_text, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)