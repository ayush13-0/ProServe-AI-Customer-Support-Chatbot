# 📞 ProServe AI — Customer Service Chatbot (Streamlit) — Notebook-aligned (STRICT)
# - Requires a cleaned QA CSV with columns: 'query' and 'response'
# - Uses same preprocessing & TF-IDF settings as the notebook:
#     preprocess(...) and TfidfVectorizer(max_features=6000, ngram_range=(1,2))
# - Lightweight retrieval + admin controls + chat export

import streamlit as st
import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io
import datetime

st.set_page_config(page_title="ProServe AI — Customer Service Chatbot", layout="wide")

# -------------------------
# Header (single professional emoji)
# -------------------------
st.title("📞 ProServe AI — Customer Service Chatbot")
st.markdown("**Notebook-aligned (STRICT)** — Uses `customer_support_QA_dataset.csv` with `query` & `response` columns. "
            "This app follows the same preprocessing and TF-IDF settings as the project notebook.")

# -------------------------
# Utilities (match notebook)
# -------------------------
def preprocess(text: str) -> str:
    """Notebook-identical preprocessing: lowercase, remove non-alphanumerics, collapse spaces."""
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_data(show_spinner=False)
def load_and_prepare_kb(path: str):
    """
    Load CSV from path (must have 'query' and 'response') and prepare clean columns.
    Returns (df, vectorizer, doc_matrix)
    """
    df = pd.read_csv(path)
    expected = {'query', 'response'}
    if not expected.issubset(set([c.lower() for c in df.columns])):
        # Normalize columns to lowercase for check, but preserve original names
        raise ValueError("CSV must contain 'query' and 'response' columns (case-insensitive).")
    # Ensure canonical column names
    # find real column names (case-insensitive match)
    cols_lower = {c.lower(): c for c in df.columns}
    df = df[[cols_lower['query'], cols_lower['response']]].rename(columns={cols_lower['query']:'query', cols_lower['response']:'response'})
    df = df.dropna(subset=['query','response']).reset_index(drop=True)
    # create clean columns
    df['clean_query'] = df['query'].apply(preprocess)
    df['clean_response'] = df['response'].apply(preprocess)  # optional, kept for parity with notebook
    # Vectorize
    vect = TfidfVectorizer(max_features=6000, ngram_range=(1,2))
    doc_matrix = vect.fit_transform(df['clean_query'])
    return df, vect, doc_matrix

class RetrievalEngine:
    def __init__(self, df, vect, doc_matrix):
        self.df = df
        self.vect = vect
        self.doc_matrix = doc_matrix

    def retrieve(self, user_text: str, top_k: int = 3):
        if self.vect is None or self.doc_matrix is None or self.df.shape[0] == 0:
            return []
        user_clean = preprocess(user_text)
        vec = self.vect.transform([user_clean])
        sims = cosine_similarity(vec, self.doc_matrix).flatten()
        idxs = np.argsort(-sims)[:top_k]
        results = [(int(i), self.df.loc[i, 'query'], self.df.loc[i, 'response'], float(sims[i])) for i in idxs]
        return results

# -------------------------
# Load KB (strict)
# -------------------------
DEFAULT_KB_PATH = "customer_support_QA_dataset.csv"
kb_path_input = st.sidebar.text_input("KB path (strict)", value=DEFAULT_KB_PATH,
                                     help="Path to cleaned CSV with columns 'query' and 'response'. App will enforce this schema.")
load_kb_btn = st.sidebar.button("Load KB")

# Use session state to store engine and data
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'kb_df' not in st.session_state:
    st.session_state.kb_df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # list of (user, bot, timestamp, meta)

# Load on start (attempt), unless user chooses different path
if load_kb_btn or st.session_state.engine is None:
    path = Path(kb_path_input)
    if not path.exists():
        st.sidebar.error(f"KB file not found at: {path.resolve()}")
    else:
        try:
            with st.spinner("Loading knowledge base..."):
                df_kb, vect_kb, doc_mat_kb = load_and_prepare_kb(str(path))
                st.session_state.kb_df = df_kb
                st.session_state.engine = RetrievalEngine(df_kb, vect_kb, doc_mat_kb)
            st.sidebar.success(f"Loaded KB — {df_kb.shape[0]} pairs.")
        except Exception as e:
            st.sidebar.error(f"Error loading KB: {e}")

# -------------------------
# Sidebar — Admin & Settings
# -------------------------
st.sidebar.markdown("## Admin & Settings")
st.sidebar.markdown("KB is *strict*: CSV must contain `query` & `response` columns.")
threshold = st.sidebar.slider("Confidence threshold (min similarity)", 0.0, 1.0, 0.25, 0.01,
                              help="Minimum cosine similarity to auto-respond. Below this, the bot will fallback or show suggestions.")
top_k = st.sidebar.number_input("Top-K suggestions", min_value=1, max_value=10, value=3, step=1)
show_explain = st.sidebar.checkbox("Show explanation (why answer chosen)", value=True)

# Admin: sample KB preview & export
if st.sidebar.checkbox("Show KB preview", value=False):
    if st.session_state.kb_df is not None:
        st.sidebar.dataframe(st.session_state.kb_df.head(10))
        if st.sidebar.button("Export KB (CSV)"):
            csv_bytes = st.session_state.kb_df.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button("Download KB CSV", data=csv_bytes, file_name="kb_export.csv", mime="text/csv")
    else:
        st.sidebar.info("No KB loaded yet.")

# -------------------------
# Main Chat UI
# -------------------------
col_left, col_right = st.columns([2,1])

with col_left:
    st.subheader("Live Chat")
    st.markdown("Type your question below. The bot uses TF-IDF retrieval (not semantic embeddings) and returns the most similar KB response.")
    user_input = st.text_input("Ask a question:", key="user_input_box", placeholder="e.g. How do I reset my password?")

    submit = st.button("Send")

    if submit and user_input:
        if st.session_state.engine is None:
            st.error("Knowledge base not loaded. Load `customer_support_QA_dataset.csv` from sidebar first.")
        else:
            # Intent-level quick rules (very small set; notebook uses only retrieval)
            # We keep only polite fallbacks: greeting/thanks/bye
            quick_intent = None
            text_l = user_input.lower()
            if any(x in text_l for x in ["hi", "hello", "hey"]):
                quick_intent = "greeting"
            elif any(x in text_l for x in ["thank", "thanks", "thx"]):
                quick_intent = "thanks"
            elif any(x in text_l for x in ["bye", "goodbye", "see ya"]):
                quick_intent = "bye"

            timestamp = datetime.datetime.now().isoformat()
            if quick_intent == "greeting":
                bot_resp = "Hello! How can I assist you today?"
                st.info(bot_resp)
                st.session_state.chat_history.append((user_input, bot_resp, timestamp, {"intent":"greeting"}))
            elif quick_intent == "thanks":
                bot_resp = "You're welcome! Anything else I can help with?"
                st.info(bot_resp)
                st.session_state.chat_history.append((user_input, bot_resp, timestamp, {"intent":"thanks"}))
            elif quick_intent == "bye":
                bot_resp = "Goodbye! Have a great day."
                st.info(bot_resp)
                st.session_state.chat_history.append((user_input, bot_resp, timestamp, {"intent":"bye"}))
            else:
                results = st.session_state.engine.retrieve(user_input, top_k=top_k)
                if not results:
                    bot_resp = "Sorry — the knowledge base is empty or engine not initialized."
                    st.warning(bot_resp)
                    st.session_state.chat_history.append((user_input, bot_resp, timestamp, {"intent":"none","score":None}))
                else:
                    # results = list of (idx, query, response, score)
                    best_idx, best_q, best_r, best_score = results[0]
                    if best_score >= threshold:
                        bot_resp = best_r
                        st.success(bot_resp)
                        if show_explain:
                            with st.expander("Why this answer?"):
                                st.write(f"Matched KB query: **{best_q}**")
                                st.write(f"Similarity score: **{best_score:.3f}**")
                        st.session_state.chat_history.append((user_input, bot_resp, timestamp, {"intent":"retrieval","score":best_score, "kb_idx":best_idx}))
                    else:
                        # Show suggestions and let user pick
                        st.write("I found some suggestions. Pick the best match or ask to connect to a human agent.")
                        for rank, (idx, q, r, s) in enumerate(results, start=1):
                            st.write(f"**Option {rank}** — {q}  _(score: {s:.3f})_")
                            if st.button(f"Use Option {rank}", key=f"use_{idx}"):
                                st.success(r)
                                st.session_state.chat_history.append((user_input, r, timestamp, {"intent":"manual_select","score":s,"kb_idx":idx}))
                        # fallback button
                        if st.button("Connect to human agent", key=f"human_{len(st.session_state.chat_history)}"):
                            human_msg = "Request forwarded to human agent."
                            st.warning(human_msg)
                            st.session_state.chat_history.append((user_input, human_msg, timestamp, {"intent":"escalate"}))

    # Chat history display
    st.markdown("### Conversation")
    if st.session_state.chat_history:
        # show last 20 messages
        for user_msg, bot_msg, ts, meta in reversed(st.session_state.chat_history[-40:]):
            st.markdown(f"**You:** {user_msg}")
            st.markdown(f"> **Bot:** {bot_msg}")
            st.caption(f"{ts} • meta: { {k:v for k,v in meta.items() if k!='kb_idx'} }")
    else:
        st.info("No conversation yet. Start by asking a question above.")

    # Export chat log
    if st.button("Export chat log"):
        if not st.session_state.chat_history:
            st.warning("No chat history to export.")
        else:
            buf = io.StringIO()
            df_hist = pd.DataFrame(st.session_state.chat_history, columns=['user','bot','timestamp','meta'])
            df_hist.to_csv(buf, index=False)
            st.download_button("Download chat CSV", data=buf.getvalue().encode('utf-8'), file_name="proserve_chat_history.csv", mime="text/csv")

with col_right:
    st.subheader("KB & Analytics")
    if st.session_state.kb_df is None:
        st.info("No KB loaded. Load `customer_support_QA_dataset.csv` from the sidebar.")
    else:
        st.markdown(f"**KB size:** {st.session_state.kb_df.shape[0]} QA pairs")
        st.markdown("**Top sample pairs:**")
        st.table(st.session_state.kb_df[['query','response']].sample(5).reset_index(drop=True))

        # Simple analytics: most common words in queries (basic)
        st.markdown("**Simple KB insights**")
        sample_n = min(20000, st.session_state.kb_df.shape[0])  # limit for perf
        sample = st.session_state.kb_df['clean_query'].sample(sample_n, random_state=42).str.cat(sep=" ")
        words = pd.Series(sample.split()).value_counts().head(25)
        st.bar_chart(words)

        st.markdown("---")
        st.markdown("**KB maintenance**")
        if st.button("Rebuild vectorizer (re-fit TF-IDF)"):
            try:
                # Force a reload of vectorizer by clearing cache and reloading
                load_and_prepare_kb.clear()
            except Exception:
                pass
            try:
                df_kb, vect_kb, doc_mat_kb = load_and_prepare_kb(str(Path(kb_path_input)))
                st.session_state.kb_df = df_kb
                st.session_state.engine = RetrievalEngine(df_kb, vect_kb, doc_mat_kb)
                st.success("Rebuilt vectorizer and refreshed engine.")
            except Exception as e:
                st.error(f"Rebuild failed: {e}")

st.markdown("---")
st.caption("ProServe AI — Notebook-aligned prototype • Built with Streamlit. Use for demos & experiments; not production ready without testing, monitoring & privacy review.")
