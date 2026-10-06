import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

# 1. Comprehensive Knowledge Base (Questions & Answers)
qa_data = {
    "question": [
        "What is NLP?", "What are Regular Expressions (Regex)?", "What is a Finite State Machine (FSM)?",
        "What is the NLP Pipeline?", "What is Tokenization?", "What is Word Normalization?",
        "What is Stemming?", "What is Edit Distance?", "Why is Python used for NLP?",
        "How do we process raw text?", "When should I use Stemming vs Lemmatization?",
        "What are the common Python libraries for NLP?", "What are the challenges in Arabic NLP?",
        "Can you explain the difference between Bag of Words and TF-IDF?",
        "What is the first step in any NLP project?", "How do I remove stop words in Python?",
        "What is the difference between Top-down and Bottom-up parsing?",
        "How can I handle emojis or special characters in text?",
        "What is a common use case for Edit Distance?", "Explain the concept of N-grams"
    ],
    "answer": [
        "Natural Language Processing (NLP) is a field of AI that enables computers to understand, interpret, and generate human language.",
        "Regular Expressions are sequences of characters that define search patterns, used for text matching and manipulation.",
        "A Finite State Machine is a mathematical model used in NLP to represent patterns or grammar rules through states and transitions.",
        "The NLP Pipeline is a series of steps to process raw text, including tokenization, cleaning, and feature extraction.",
        "Tokenization is the process of breaking down a stream of text into smaller units called tokens, like words or sentences.",
        "Word normalization is the process of converting text into a standard format, such as converting all letters to lowercase.",
        "Stemming is an NLP technique that reduces words to their root or base form by removing suffixes (e.g., 'running' to 'run').",
        "Edit Distance (like Levenshtein distance) is a way of quantifying how dissimilar two strings are by counting minimum operations to transform one to another.",
        "Python is the leading language for NLP due to its simple syntax and powerful libraries like NLTK, Spacy, and Scikit-learn.",
        "Processing raw text involves cleaning (removing noise), tokenization, and transforming text into numerical data for models.",
        "Use Stemming for speed. Use Lemmatization for accuracy and getting the dictionary root (lemma).",
        "Popular libraries: NLTK (academic), Spacy (industrial), Scikit-learn (ML), and Hugging Face (Deep Learning).",
        "Arabic NLP is hard due to complex morphology (roots), many dialects, and letters changing shape based on position.",
        "Bag of Words counts frequency. TF-IDF penalizes common words (like 'the') and weights unique, meaningful words higher.",
        "The first step is 'Data Cleaning' to ensure the model isn't confused by noise like HTML tags or punctuation.",
        "Use NLTK: 'filtered_words = [w for w in words if not w in stop_words]'. This removes meaningless semantic words.",
        "Top-down parsing starts from grammar rules and works down to words. Bottom-up starts from words and builds up to rules.",
        "Use Python's 're' library with Unicode patterns to either keep or remove emojis based on project needs.",
        "Common use cases include 'Auto-correct' features and 'Plagiarism Detection'.",
        "N-grams are sequences of N items from text. E.g., a Bigram (N=2) of 'NLP is fun' is ('NLP is', 'is fun')."
    ]
}
df = pd.DataFrame(qa_data)

# 2. Text Preprocessing
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(tokens)

# 3. Chatbot Logic
def get_bot_response(user_input):
    processed_questions = df['question'].apply(preprocess_text).tolist()
    processed_user_input = preprocess_text(user_input)
    processed_questions.append(processed_user_input)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_questions)
    scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    index = scores.argmax()
    if scores[0][index] < 0.2:
        return "I'm sorry, I don't quite understand that. Can you try rephrasing?"
    return df['answer'].iloc[index]

# 4. Streamlit UI (Enhanced Design)
st.set_page_config(page_title="Ahmed's NLP Assistant", page_icon="🤖", layout="wide")

# --- CUSTOM CSS FOR GRADIENT DESIGN ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    
    h1 {
        background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        text-align: center;
        font-size: 3rem !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stChatInputContainer {
        border-radius: 20px;
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="stChatMessage"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stMarkdown p {
        color: #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)
    st.title("Project Info")
    st.write("**devoloped by:** Ahmed Gehad ")
    st.write("**Project:** NLP Smart Assistant")
    st.write("**Course:** Natural Language Processing")
    st.divider()
    st.info("This bot uses TF-IDF and Cosine Similarity to provide intelligent responses.")

# Main Header
st.title("🤖 NLP Smart Assistant")
st.markdown(f"Welcome to my project. I've built this assistant to help explain core NLP concepts.")
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask me about Regex, Tokenization, Stemming..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = get_bot_response(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
