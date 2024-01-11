# Patient-Condition_Classification
*Patient condition classification*, which predicts the medical issue of a sentence and recommends drugs to prevent or treat that issue, involves the use of natural language processing (NLP) and machine learning techniques to analyze text input and provide relevant medical information. 
![Project flow](https://github.com/m-rishab/Patient-Condition_Classification/assets/113618652/b55a1e1b-43bc-4cde-a7ab-b5dbc05b9e37)

## Project Overview

This project focuses on classifying the condition of a patient based on their reviews, enabling the recommendation of suitable drugs. The implementation leverages Natural Language Processing (NLP) techniques through a structured pipeline.

## NLP Pipeline Steps

1. **Tokenization:**
   - Break down sentences into individual tokens.

2. **Clean Reviews:**
   - Remove punctuation.
   - Eliminate special characters and numbers.
   - Convert text to lowercase.
   - Perform lemmatization.

3. **Bag of Words Model:**
   - Create a bag of words model to vectorize the preprocessed reviews.

4. **Apply ML Algorithms (Naive Bayes, Passive Aggressive Classifier):**
   - Train and test the model using Naive Bayes and Passive Aggressive Classifier.

5. **TFIDF Model:**
   - Create a TFIDF model to vectorize the preprocessed reviews.

6. **Apply ML Algorithms (Naive Bayes, Passive Aggressive Classifier):**
   - Train and test the model using Naive Bayes and Passive Aggressive Classifier.

7. **Comparison:**
   - Evaluate and compare the performance of both models.

## Project Demo
## *DEMO-1*
![f09fb82c-1725-470c-baf4-1d9a584a0047](https://github.com/m-rishab/Patient-Condition_Classification/assets/113618652/0e13d667-fd4a-468d-947a-a82dd4476819)
## *DEMO-2*
![4753fca3-e12a-47c6-be53-3f5a0e0f9f4c](https://github.com/m-rishab/Patient-Condition_Classification/assets/113618652/40496c5b-69bf-47cf-96c8-2a2950ab967e)



[![Project Demo](https://example.com/path/to/your/demo.gif)](https://example.com/path/to/your/demo.gif)

## Technologies Used

- ![Python](https://img.shields.io/badge/Python-3.11%2B-blue): The core programming language used for NLP and machine learning algorithms.
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-blue), ![NLTK](https://img.shields.io/badge/NLTK-3.8.1-blue), ![spaCy](https://img.shields.io/badge/spaCy-3.6-green): Libraries for natural language processing and machine learning.
- ![Flask](https://img.shields.io/badge/Flask-2.3.3-green): For building the web-based user interface.

## Running the Project

To run the project, execute the following command:

```bash
python app.py
# or
flask run
