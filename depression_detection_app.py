import streamlit as st
import streamlit.components.v1 as components
import tweepy
import re
from joblib import load
import shap
from sentence_transformers import SentenceTransformer, util
import numpy as np

# PHQ-9 questions
phq9_questions = [
    "Little interest or pleasure in doing things?",
    "Feeling down, depressed, or hopeless?",
    "Trouble falling or staying asleep, or sleeping too much?",
    "Feeling tired or having little energy?",
    "Poor appetite or overeating?",
    "Feeling bad about yourself — or that you are a failure?",
    "Trouble concentrating on things?",
    "Moving or speaking so slowly that other people could have noticed?",
    "Thoughts that you would be better off dead or of hurting yourself?"
]

# Load model and vectorizer
vectorizer = load("tfidf_vectorizer.joblib")
model = load("logistic_model.joblib")
modelst = SentenceTransformer('all-MiniLM-L6-v2')

# Clean text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    return text.lower()

# Setup SHAP explainer
text_explainer = shap.Explainer(
    lambda x: model.predict_proba(vectorizer.transform([clean_text(i) for i in x])),
    shap.maskers.Text(r"\W+")
)

# Twitter API setup
def get_api():
    auth = tweepy.OAuth2BearerHandler("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx") #Replace with your Bearer Handle
    api = tweepy.API(auth)
    return api

def map_tweets_to_phq9(tweets):
    tweet_embeddings = modelst.encode(tweets, convert_to_tensor=True)
    phq9_embeddings = modelst.encode(phq9_questions, convert_to_tensor=True)
    similarities = util.cos_sim(tweet_embeddings, phq9_embeddings)
    results = []
    for idx, tweet in enumerate(tweets):
        top_match = similarities[idx].cpu().numpy().argmax()
        score = similarities[idx][top_match].item()
        results.append((tweet, phq9_questions[top_match], score))
    return results

def predict_depression_from_tweets(tweets):
    combined = " ".join([clean_text(t) for t in tweets])
    pred = model.predict(vectorizer.transform([combined]))[0]
    prob = model.predict_proba(vectorizer.transform([combined]))[0][1]
    return ("Depressed" if pred else "Not Depressed"), prob, combined

st.title("Depression Detection from Tweets with PHQ-9 Mapping")
st.write("This app analyzes recent tweets from a specified Twitter user to determine their depression status and maps the tweets to PHQ-9 symptoms.")
username = st.text_input("Enter Twitter username:")
num_tweets = st.slider("Number of recent tweets to analyze:", 5, 100, 10)

if st.button("Determine Depression Status"):
    api = get_api()
    try:
        tweets = [status.full_text for status in tweepy.Cursor(api.user_timeline, screen_name=username, tweet_mode="extended").items(num_tweets)]

        if not tweets:
            st.warning("No tweets found.")
        else:
            st.session_state['tweets'] = tweets
            result, confidence, combined = predict_depression_from_tweets(tweets)
            st.session_state['prediction'] = result
            st.session_state['confidence'] = confidence
            st.session_state['shap_values'] = text_explainer([combined])
            st.session_state['phq9_results'] = map_tweets_to_phq9(tweets)

    except Exception as e:
        st.error(f"Error: {e}")

if 'tweets' in st.session_state:
    st.markdown("### Tweets Analyzed:")
    for i, tweet in enumerate(st.session_state['tweets'], 1):
        st.markdown(f"- **Tweet {i}:** {tweet}")

    st.subheader(f"Prediction: {st.session_state['prediction']}")
    st.write(f"Confidence: {st.session_state['confidence']:.2f}")

    shap_html = shap.plots.text(st.session_state['shap_values'][0], display=False)
    components.html(shap_html, height=400, scrolling=True)

    st.subheader("PHQ-9 Symptom Mapping")
    threshold = st.slider(
        "Set similarity threshold to filter relevant tweets:",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05
    )

    relevant_results = [r for r in st.session_state['phq9_results'] if r[2] > threshold]

    if relevant_results:
        st.markdown(f"### Tweets matching PHQ-9 symptoms (Threshold: {threshold:.2f})")
        for tweet, question, score in relevant_results:
            st.markdown(f"**Tweet:** {tweet}")
            st.markdown(f"→ **Closest PHQ-9 Symptom:** *{question}*")
            st.markdown(f"→ **Similarity Score:** `{score:.2f}`")
            st.markdown("---")
    else:
        st.info(f"No tweets matched PHQ-9 symptoms above threshold {threshold:.2f}.")
