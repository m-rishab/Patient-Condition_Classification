from flask import Flask, render_template, request
import nltk
import os
import joblib
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
nltk.download('stopwords')
nltk.download('wordnet')

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

app = Flask(__name__)

app.secret_key = os.urandom(24)
MODEL_PATH = 'notebook/model/passmodel.pkl'
TOKENIZER_PATH = 'notebook/model/tfidfvectorizer.pkl'
DATA_PATH = 'data/DrugsTest_raw.csv'

# Loading vectorizer and model
vectorizer = joblib.load(TOKENIZER_PATH)
model = joblib.load(MODEL_PATH)

# Getting stopwords
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

@app.route('/')
def index():
    df = pd.read_csv(DATA_PATH)

    # Fetch top conditions and drugs for specific conditions
    top_conditions_list = top_conditions(df)
    top_drugs_10_rating = top_drugs_for_condition(df, rating=10)

    return render_template('index.html', top_conditions=top_conditions_list, top_drugs_10_rating=top_drugs_10_rating)

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        raw_text = request.form.get('rawtext', '')  # Use get() with default value to avoid KeyError

        if raw_text:
            clean_text = cleanText(raw_text)
            clean_lst = [clean_text]

            tfidf_vect = vectorizer.transform(clean_lst)
            prediction = model.predict(tfidf_vect)
            predicted_cond = prediction[0]
            df = pd.read_csv(DATA_PATH)
            top_drugs = top_drugs_extractor(predicted_cond, df)

            return render_template('predict.html', rawtext=raw_text, result=predicted_cond, top_drugs=top_drugs)
        else:
            raw_text = "There is no text to select"

    return render_template('index.html')

def cleanText(raw_review):
    # 1. Delete HTML
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords
    meaningful_words = [w for w in words if not w in stop]
    # 6. lemmatization
    lemmatize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. space join words
    return ' '.join(lemmatize_words)

def top_drugs_extractor(condition, df):
    df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 100)].sort_values(by=['rating', 'usefulCount'],
                                                                                 ascending=[False, False])
    drug_lst = df_top[df_top['condition'] == condition]['drugName'].head(3).tolist()
    return drug_lst

def top_conditions(df):
    top_conditions = df['condition'].value_counts().head(10).index.tolist()
    return top_conditions

def top_drugs_for_condition(df, rating):
    df_top = df[df['rating'] == rating].head(20)
    top_drugs = df_top['drugName'].tolist()
    return top_drugs

if __name__ == "__main__":
    # Use the PORT environment variable if available, otherwise default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
