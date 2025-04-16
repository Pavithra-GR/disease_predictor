
from flask import Flask, request, render_template
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

@app.route('/predict', methods=['POST'])
def predict():
    symptom_text = request.form['symptoms']
    cleaned = preprocess_text(symptom_text)
    vectorized = tfidf_vectorizer.transform([cleaned])
    prediction = knn_model.predict(vectorized)[0]

    return render_template('index.html', symptoms=symptom_text, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)