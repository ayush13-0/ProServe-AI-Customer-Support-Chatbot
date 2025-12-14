import streamlit as st
import time
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Apply ChatGPT-Style CSS
# ---------------------------
st.markdown("""
<style>

body {
    background-color: #0d1117;
}

.chat-container {
    width: 100%;
    padding: 10px;
}

/* USER (Right Side) */
.user-row {
    display: flex;
    justify-content: flex-end;
    margin: 10px 0;
}

.user-msg {
    background: #238636;
    color: white;
    padding: 12px 16px;
    border-radius: 15px 15px 0 15px;
    max-width: 70%;
    box-shadow: 0 0 8px rgba(0,0,0,0.25);
    font-size: 16px;
}

.user-icon {
    background:#238636;
    color:white;
    padding:8px;
    height:32px;
    width:32px;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    margin-left:8px;
}

/* BOT (Left Side) */
.bot-row {
    display: flex;
    justify-content: flex-start;
    margin: 10px 0;
}

.bot-msg {
    background: #1e1e1e;
    color: #f1f1f1;
    padding: 12px 16px;
    border-radius: 15px 15px 15px 0;
    max-width: 70%;
    box-shadow: 0 0 8px rgba(0,0,0,0.25);
    font-size: 16px;
}

.bot-icon {
    background:#1e1e1e;
    color:white;
    padding:8px;
    height:32px;
    width:32px;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    margin-right:8px;
}

/* Timestamp */
.timestamp {
    font-size: 11px;
    color:#aaa;
    margin-top: 3px;
    margin-left: 45px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Preprocess Function
# ---------------------------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ---------------------------
# Load Model & Dataset (Notebook-Aligned)
# ---------------------------
@st.cache_resource
def load_kb():
    df = pd.read_csv("customer_support_QA_dataset.csv")
    df["clean_query"] = df["query"].apply(preprocess)
    vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df["clean_query"])
    return df, vectorizer, X

df, vectorizer, X = load_kb()

# ---------------------------
# Retrieval Function
# ---------------------------
def get_response(user_input):
    clean = preprocess(user_input)
    vec = vectorizer.transform([clean])
    sims = cosine_similarity(vec, X).flatten()
    idx = sims.argmax()

    if sims[idx] < 0.25:
        return "I'm not fully sure — would you like to connect with a human support agent?"
    return df.iloc[idx]["response"]

# ---------------------------
# Session State
# ---------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []        # conversation
if "memory" not in st.session_state:
    st.session_state.memory = []      # memory panel

# ---------------------------
# Sidebar — Memory Panel
# ---------------------------
st.sidebar.title("🧠 AI Memory")
if st.session_state.memory:
    for item in st.session_state.memory:
        st.sidebar.write(f"- {item}")
else:
    st.sidebar.info("No memory stored yet.")

# ---------------------------
# Chat Title
# ---------------------------
st.title("🤖 ProServe AI — Customer Service Chatbot")

# ---------------------------
# User Input
# ---------------------------
prompt = st.text_input("Ask ProServe AI", placeholder="Type your message...")

if st.button("Send"):
    if prompt.strip():
        # Store user message
        st.session_state.chat.append({"role":"user", "text":prompt, "time":time.strftime("%H:%M:%S")})

        # Retrieve bot reply
        bot_reply = get_response(prompt)

        # Save memory if important
        if len(prompt.split()) > 4:
            st.session_state.memory.append(prompt)

        # Add placeholder for typing animation
        placeholder = st.empty()

        typed = ""
        for char in bot_reply:
            typed += char
            time.sleep(0.01)
            placeholder.markdown(f"""
            <div class="chat-container">
                <div class="bot-row">
                    <div class="bot-icon">🤖</div>
                    <div class="bot-msg">{typed}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Finally save bot message
        st.session_state.chat.append({"role":"bot", "text":bot_reply, "time":time.strftime("%H:%M:%S")})
        st.rerun()

# ---------------------------
# Chat Display
# ---------------------------
st.subheader("💬 Conversation")

for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-container">
            <div class="user-row">
                <div class="user-msg">{msg['text']}</div>
                <div class="user-icon">🧑</div>
            </div>
            <div class="timestamp">{msg['time']}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="chat-container">
            <div class="bot-row">
                <div class="bot-icon">🤖</div>
                <div class="bot-msg">{msg['text']}</div>
            </div>
            <div class="timestamp">{msg['time']}</div>
        </div>
        """, unsafe_allow_html=True)
