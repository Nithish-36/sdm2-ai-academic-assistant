import streamlit as st
import os
import re
from datetime import datetime

# ===============================
# ✅ SAFE API KEY LOAD
# ===============================
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("❌ Please add GOOGLE_API_KEY in secrets.toml")
    st.stop()

# ===============================
# IMPORTS
# ===============================
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

# ===============================
# SESSION STATE
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "context_state" not in st.session_state:
    st.session_state.context_state = None

if "zoom_state" not in st.session_state:
    st.session_state.zoom_state = None

# ===============================
# PAGE
# ===============================
st.set_page_config(page_title="AI Assistant", layout="centered")
st.title("📚 Statistical Data Mining-2 Assistant")
st.markdown(
    "<h4 style='text-align: center; color: #9CA3AF;'>🤖 RHB (Racheal Hageman’s Bot)</h4>",
    unsafe_allow_html=True
)

# ===============================
# UI
# ===============================
st.markdown("""
<style>
.block-container { max-width: 700px; margin: auto; padding-bottom: 80px; }
[data-testid="stChatMessageAvatar"] { display: none; }

[data-testid="stChatMessage"] {
    display: flex;
    width: 100%;
    margin-bottom: 12px;
}

[data-testid="stChatMessage"]:nth-child(even) {
    justify-content: flex-end;
}

[data-testid="stChatMessage"]:nth-child(odd) {
    justify-content: flex-start;
}

[data-testid="stChatMessage"]:nth-child(even)
[data-testid="stMarkdownContainer"] {
    background-color: #374151;
    color: white;
    text-align: right;
}

[data-testid="stChatMessage"]:nth-child(odd)
[data-testid="stMarkdownContainer"] {
    background-color: #1f2937;
    color: #e5e7eb;
}

[data-testid="stMarkdownContainer"] {
    max-width: 70%;
    padding: 12px 16px;
    border-radius: 18px;
    font-size: 15px;
}

section[data-testid="stChatInput"] {
    position: fixed;
    bottom: 0;
    width: 100%;
    background: #0e1117;
    padding: 10px;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===============================
# HELPERS
# ===============================
def days_left(date):
    diff = (datetime.strptime(date, "%Y-%m-%d") - datetime.now()).days
    if diff < 0:
        return "Assignment due date is over."
    return f"{diff} days left"

def contact_info():
    return """For better clarification, you may consider contacting:

👨‍🏫 Professor: Dr. Raacheal Hageman Blair  
📧 hageman@buffalo.edu 
Office Hours: Monday & Wednesday (1:00 PM - 2:30 PM)

👨‍💻 TA: Nithish Kumar Reddy Yerreddy  
📧 nyerredd@buffalo.edu  
Office Hours: Monday & Thursday (3:00 PM - 4:00 PM) 

🧑‍🏫 Grader: Nuthan Teja Reddy  
📧 nuthan@buffalo.edu  
🕒 Office Hours: Tuesday & Thursday (12:00 PM – 1:00 PM)
"""

# ===============================
# ✅ NEW INTENT HELPERS
# ===============================
def has_word(q, words):
    pattern = r'\b(?:' + '|'.join(map(re.escape, words)) + r')\b'
    return re.search(pattern, q.lower()) is not None

def detect_normal_intent(q):
    q = q.lower().strip()

    if len(q.split()) <= 3 and has_word(q, ["hi", "hello", "hey"]):
        return "greeting"

    if "you too" in q or "you as well" in q:
        return "you_too"

    if has_word(q, ["thanks", "thank you", "thank"]):
        return "thanks"

    if "how are you" in q:
        return "how_are_you"

    if has_word(q, ["bye", "goodbye"]):
        return "bye"

    if "who are you" in q:
        return "who_are_you"

    if "your name" in q:
        return "your_name"

    if "what is rhb" in q:
        return "what_is_rhb"

    if has_word(q, ["vacancy", "position", "openings"]):
        return "vacancy"

    return "other"

# ===============================
# RAG SETUP
# ===============================
def build_retriever(pdf_list):
    docs_all = []
    for f in pdf_list:
        if os.path.exists(f):
            docs_all.extend(PyPDFLoader(f).load())

    if not docs_all:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(docs_all)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    db = FAISS.from_documents(docs, embeddings)
    return db.as_retriever(search_kwargs={"k": 5})

@st.cache_resource
def load_subject_data():
    return build_retriever([
        "14. CART.pdf", "biplots_and_outliers.pdf", "Overview.pdf",
        "Association_presA.pdf", "PGM_I .pdf", "PGM_structure.pdf",
        "BN_Prob_Reasoning.pdf", "Association_presB.pdf",
        "Association_presD.pdf", "Clustering_1-2.pdf",
        "Clustering_2-2.pdf", "Clustering_3.pdf",
        "Clustering_4.pdf", "Clustering_4b.pdf",
        "SOM.pdf", "JC_Clauset.pdf", "PCA.pdf",
        "PGM_II_2025.pdf", "UDG_A.pdf"
    ])

@st.cache_resource
def load_project_data():
    return build_retriever(["SDM-2 Project proposal guidlines.pdf"])

# ===============================
# MODEL
# ===============================
def load_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        max_retries=3
    )

# ===============================
# PROMPT
# ===============================
prompt = ChatPromptTemplate.from_template("""
You are an intelligent and student-friendly academic assistant for the course Statistical Data Mining-2.

Your goal is to help students understand concepts clearly, prepare for exams, and reduce confusion.

CONTEXT:
{context}

QUESTION:
{input}

ANSWER:
""")

# ===============================
# CACHE
# ===============================
@st.cache_data(show_spinner=False)
def cached_subject_response(query):
    retriever = load_subject_data()
    llm = load_model()

    chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt)
    )

    result = chain.invoke({"input": query})
    return result.get("answer", "")

@st.cache_data(show_spinner=False)
def cached_project_response(query):
    retriever = load_project_data()
    llm = load_model()

    chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt)
    )

    result = chain.invoke({"input": query})
    return result.get("answer", "")

# ===============================
# SHOW HISTORY
# ===============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=None):
        st.markdown(msg["content"])

# ===============================
# INPUT
# ===============================
query = st.chat_input("Ask your question")

if query:
    response = ""
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user", avatar=None):
        st.markdown(query)

    q = query.lower()
    clean_q = re.sub(r'\s+', ' ', q.strip())

    # ===============================
    # SMART EXIT FROM ASSIGNMENT MODE
    # ===============================
    assignment_keywords = [
        "assignment", "homework", "hw",
        "question", "problem", "q1", "q2",
        "calculate", "find", "compute", "implement"
    ]

    is_assignment_related = any(word in q for word in assignment_keywords)

    if (
        st.session_state.context_state == "assignment_doubt"
        and not is_assignment_related
    ):
        st.session_state.context_state = None

    # ===============================
    # IMMEDIATE FACTS
    # ===============================
    if st.session_state.zoom_state == "ask_person":

        if "professor" in q:
            response = "Professor Zoom: https://buffalo.zoom.us/j/7342873196"
            st.session_state.zoom_state = None

        elif re.search(r"\bta\b", q):
            response = "TA Zoom: https://buffalo.zoom.us/j/93740724275"
            st.session_state.zoom_state = None

        elif "grader" in q:
            response = "Grader Zoom: https://buffalo.zoom.us/j/4027519593"
            st.session_state.zoom_state = None

        else:
            response = "Please choose: Professor / TA / Grader"

    elif has_word(q, ["professor"]):
        response = """👨‍🏫 **Professor: Dr. Raacheal Hageman Blair**
📧 hageman@buffalo.edu
Office Hours: Monday & Wednesday (1:00 PM - 2:30 PM)"""

    elif re.search(r"\bta\b", q) or "nithish" in q:
        response = """👨‍💻 **TA: Nithish Kumar Reddy Yerreddy**
📧 nyerredd@buffalo.edu
Office Hours: Monday & Thursday (3:00 PM - 4:00 PM)"""

    elif "grader" in q or "nuthan" in q:
        response = """🧑‍🏫 **Grader: Nuthan Teja Reddy**
📧 nuthan@buffalo.edu
Office Hours: Tuesday & Thursday (12:00 PM – 1:00 PM)"""

    # ===============================
    # DOUBT FLOW
    # ===============================
    elif "doubt" in q and st.session_state.context_state is None:
        st.session_state.context_state = "doubt_type"
        response = "Is your doubt related to subject, assignment, or project?"

    elif st.session_state.context_state == "doubt_type":

        if any(x in q for x in [
            "nothing", "cancel", "never mind",
            "clarified", "solved", "no doubt",
            "something else"
        ]):
            st.session_state.context_state = None
            response = "No problem 😊 What would you like help with now?"

        elif len(q.split()) <= 3 and has_word(q, ["hi", "hello", "hey"]):
            st.session_state.context_state = None
            response = "Hello 😊 How can I help you today?"

        elif "subject" in q:
            st.session_state.context_state = "subject_doubt"
            response = "Please explain your subject doubt."

        elif "assignment" in q or "homework" in q:
            st.session_state.context_state = "assignment_doubt"
            response = "What exactly is your assignment doubt?"

        elif "project" in q:
            st.session_state.context_state = "project_doubt"
            response = "Please explain your project doubt."

        else:
            response = "Please choose: subject / assignment / project or type cancel."

    elif st.session_state.context_state == "project_doubt":
        try:
            response = cached_project_response(query)
        except Exception as e:
            response = f"I ran into an issue: {str(e)}"

    elif st.session_state.context_state == "subject_doubt":
        try:
            response = cached_subject_response(query)
        except Exception as e:
            response = f"I ran into an issue: {str(e)}"
        st.session_state.context_state = None

    # ===============================
    # NORMAL FLOW
    # ===============================
    else:

        if "simple" in q:
            query += " Explain in very simple words."

        intent = detect_normal_intent(q)

        if intent == "greeting":
            response = "Hello! 😊 I'm here to help you. What would you like to explore today?"

        elif intent == "you_too":
            response = "Thanks 😊"

        elif intent == "thanks":
            response = "You're very welcome 😊 Happy to help! Have a great day ahead!"

        elif intent == "how_are_you":
            response = "I'm doing great, thanks for asking! 😊 How can I help you today?"

        elif intent == "bye":
            response = "Goodbye! 👋 Take care and have a wonderful day 😊"

        elif intent == "who_are_you":
            response = "I am the Statistical Data Mining-2 Assistant 😊"

        elif intent == "your_name":
            response = "My name is RHB."

        elif intent == "what_is_rhb":
            response = "RHB stands for Racheal Hageman's Bot."

        elif any(day in q for day in ["monday", "tuesday", "wednesday", "thursday"]):
            response = contact_info()

        elif intent == "vacancy":
            response = "Currently, there are no open positions."

        elif any(x in q for x in ["research", "thesis"]) and "project" not in q:
            response = "Professor is not currently working on research projects.\nContact: hageman@buffalo.edu"

        elif "zoom" in q:

            if "professor" in q:
                response = "Professor Zoom: https://buffalo.zoom.us/j/7342873196"

            elif re.search(r"\bta\b", q):
                response = "TA Zoom: https://buffalo.zoom.us/j/93740724275"

            elif "grader" in q:
                response = "Grader Zoom: https://buffalo.zoom.us/j/4027519593"

            else:
                st.session_state.zoom_state = "ask_person"
                response = "Whose Zoom link do you need? (Professor / TA / Grader)"

        else:
            try:
                project_words = [
                    "project", "proposal", "submission",
                    "rate my proposal",
                    "score my proposal",
                    "evaluate proposal",
                    "dataset idea",
                    "project guidelines"
                ]

                if any(word in q for word in project_words):
                    response = cached_project_response(query)
                else:
                    response = cached_subject_response(query)

            except Exception as e:
                response = f"I ran into an issue: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant", avatar=None):
        st.markdown(response)

    st.rerun()