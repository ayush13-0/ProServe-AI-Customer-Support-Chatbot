<!-- PROJECT BADGE -->
<p align="center">
  <img src="https://img.shields.io/badge/ProServe%20AI-Intelligent%20Customer%20Support%20Chatbot-blueviolet?style=for-the-badge&logo=python&logoColor=white" />
</p>

<!-- TITLE -->
<h1 align="center">🤖💬 ProServe AI – Customer Support Chatbot</h1>

<!-- TAGLINE -->
<p align="center">
  <b>An intelligent ML-powered customer support chatbot using NLP & semantic similarity</b>
</p>

<!-- CORE TECH BADGES -->
<p align="center">
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/NLP-TF--IDF%20%7C%20Cosine%20Similarity-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Data-Pandas-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/UI-Streamlit-red?style=for-the-badge&logo=streamlit" />
</p>

<!-- ADVANCED / STATUS BADGES -->
<p align="center">
  <img src="https://img.shields.io/badge/Model-Retrieval--Based%20AI-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Deployment-Streamlit%20App-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge" />
</p>

<!-- INTERNSHIP / PROGRAM BADGES -->
<p align="center">
  <img src="https://img.shields.io/badge/Internship-CODEC Technologies%20ML%20Internship-black?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Project-Type%20AI%20Application-blue?style=for-the-badge" />
</p>

<!-- OPTIONAL LIVE DEMO BADGE -->
<p align="center">
  <img src="https://img.shields.io/badge/Live%20Demo-Available-success?style=for-the-badge&logo=streamlit" />
</p>


# 🚀 Project Overview

Modern businesses rely on AI-driven customer support systems to deliver fast and accurate responses at scale. **ProServe AI** replicates this real-world customer service automation using **Machine Learning and Natural Language Processing**.

ProServe AI is a **retrieval-based customer support chatbot** that understands user queries, converts them into numerical vectors, and retrieves the most relevant response from a structured knowledge base using **TF-IDF vectorization and cosine similarity**.

The system performs:

* Query preprocessing & cleaning
* TF-IDF feature extraction
* Semantic similarity computation
* Confidence-based response selection
* Human-agent fallback handling

# 📂 Dataset Used

This project uses a **Customer Support Q&A Dataset**, containing:

* customer_support_QA_dataset.csv

Each record includes:

* ❓ Customer query
* 💬 Predefined support response

The dataset simulates real-world customer service conversations across multiple support scenarios.

# 🏗️ Project Workflow

1️⃣ Importing Libraries
Used for data handling, NLP processing, ML modeling, and UI development.

* import pandas as pd
* import re
* from sklearn.feature_extraction.text import TfidfVectorizer
* from sklearn.metrics.pairwise import cosine_similarity
* import streamlit as st

2️⃣ Loading the Dataset

* df = pd.read_csv("customer_support_QA_dataset.csv")

3️⃣ Text Preprocessing

* Lowercasing text
* Removing special characters
* Normalizing extra spaces

```python
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
```

4️⃣ TF-IDF Vectorization

* Converts cleaned queries into numerical vectors.

```python
vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['clean_query'])
```

5️⃣ Cosine Similarity Computation

* Measures semantic similarity between user query and knowledge base.

```python
similarity = cosine_similarity(user_vector, X)
```

6️⃣ Response Retrieval Logic

* Highest similarity score determines the response.
* Low-confidence queries trigger fallback to human support.

# 🧠 Chatbot Intelligence

The system includes:

* Confidence threshold handling
* Contextual chat memory
* Session-based conversation history
* Typing animation for realistic UX

# 🎨 User Interface

* ChatGPT-style dark theme
* Real-time chat interaction
* Sidebar memory panel
* Smooth typing animation

Built entirely using **Streamlit + custom HTML/CSS**.

# 📁 Project Structure

```
ProServe-AI/
│
├── customer_support_QA_dataset.csv   # Support knowledge base
├── ProServeAI.py                     # Streamlit chatbot app
├── ProServeAI-v2                     # Streamlit v2 chatbot app 
├── ProServe-AI-Main.ipynb            # Model development notebook
├── requirements.txt                 # Dependencies
└── README.md                         # Documentation
```

# ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run ProServeAI.py
```

# 📌 Use Cases

* Customer Support Automation
* FAQ & Helpdesk Systems
* SaaS Support Bots
* AI Knowledge Base Assistants
* Business Service Chatbots

# 📈 Future Enhancements

* Sentence-BERT embeddings
* FAISS vector database
* Multilingual support
* Intent classification
* Cloud deployment (AWS / GCP)

# 🏁 Conclusion

* ProServe AI demonstrates how **Machine Learning and NLP** can automate customer support effectively.
* By leveraging TF-IDF and cosine similarity, the system delivers fast, accurate, and scalable responses.
* The project serves as a strong foundation for production-grade AI-powered support systems.

# 👨‍💻 Author

## Ayush

* 🔗 GitHub: [https://github.com/ayush13-0](https://github.com/ayush13-0)
* 🔗 LinkedIn: [https://linkedin.com/in/ayush130](https://linkedin.com/in/ayush130)

