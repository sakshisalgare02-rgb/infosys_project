import streamlit as st
from rag_pipeline import ask_question

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #e6f0ff;
}

/* Title */
.title {
    font-size:36px;
    font-weight:bold;
    text-align:center;
    color:#003366;
}

/* Subtitle */
.subtitle {
    font-size:22px;
    text-align:center;
    color:#004080;
}

/* Answer */
.answer {
    font-size:18px;
    color:#202124;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🤖 AI Document Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask your question from uploaded documents</p>', unsafe_allow_html=True)

query = st.text_input("Ask your question", key="question_box")

if query:
    answer = ask_question(query)
    st.markdown('<p class="answer"><b>Answer:</b></p>', unsafe_allow_html=True)
    st.write(answer)