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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Statistical Data Mining-2 Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)




# ===============================
# MODERN UI
# ===============================
st.markdown("""
<style>

/* =========================
BACKGROUND
========================= */

.stApp {
    background:
        radial-gradient(circle at top left, rgba(120,119,198,0.15), transparent 25%),
        radial-gradient(circle at bottom right, rgba(91,134,229,0.12), transparent 25%),
        #030712;
    color: white;
    overflow-x: hidden;
}

.stApp::before {
    content: "";
    position: fixed;
    width: 700px;
    height: 700px;
    background: radial-gradient(
        circle,
        rgba(139,92,246,0.18),
        transparent 70%
    );
    top: -250px;
    left: -200px;
    z-index: -1;
    filter: blur(80px);
}

/* =========================
REMOVE STREAMLIT STUFF
========================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* =========================
SIDEBAR
========================= */

section[data-testid="stSidebar"] {
    background: transparent;
    border: none;
    width: 0px !important;
    min-width: 0px !important;
}

section[data-testid="stSidebar"] > div {
    background: transparent;
}

[data-testid="collapsedControl"] {
    display: none;
}

/* =========================
CLEAR BUTTON
========================= */

.stButton > button {
    background: rgba(10,14,35,0.78);
    color: white;
    border: 1px solid rgba(99,102,241,0.28);
    border-radius: 16px;
    width: 120px;
    height: 52px;
    font-size: 16px;
    font-weight: 500;
    backdrop-filter: blur(14px);
    transition: all 0.25s ease;
    box-shadow:
        0 0 25px rgba(99,102,241,0.08);
}

.stButton > button:hover {
    background: rgba(18,24,48,0.95);
    border: 1px solid rgba(139,92,246,0.45);
    transform: translateY(-1px);
}

/* =========================
FIX WHITE BACKGROUND
========================= */

html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stMain"] {
    background: #030712 !important;
}

.main .block-container {
    background: transparent !important;
}

/* =========================
MAIN CONTAINER
========================= */

.block-container {
    max-width: 950px;
    padding-top: 1rem;
    padding-bottom: 8rem;
}

/* =========================
HERO SECTION
========================= */

.hero-container {
    text-align: center;
    margin-top: -120px;
    margin-bottom: 40px;
}

.hero-title {
    font-size: 72px;
    letter-spacing: -2px;
    font-weight: 800;
    line-height: 1.1;
    text-shadow: 0 0 25px rgba(139,92,246,0.25);
    background: linear-gradient(
        90deg,
        #d8b4fe,
        #818cf8,
        #60a5fa
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 18px;
}

.hero-subtitle {
    font-size: 26px;
    color: #d1d5db;
    margin-bottom: 10px;
}

.hero-description {
    font-size: 18px;
    color: #9ca3af;
}

/* =========================
CHAT MESSAGES
========================= */

[data-testid="stChatMessageAvatar"] {
    display: none;
}

[data-testid="stChatMessage"] {
    border: none;
    background: transparent;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    border-radius: 22px;
    padding: 16px 20px;
    font-size: 16px;
    line-height: 1.7;
    backdrop-filter: blur(14px);
    box-shadow:
        0 0 25px rgba(139,92,246,0.08);
}

/* USER MESSAGE */

[data-testid="stChatMessage"]:nth-child(even) [data-testid="stMarkdownContainer"] {
    background: linear-gradient(
        135deg,
        rgba(99,102,241,0.28),
        rgba(139,92,246,0.20)
    );
    border: 1px solid rgba(139,92,246,0.25);
    color: white;
}

/* BOT MESSAGE */

[data-testid="stChatMessage"]:nth-child(odd) [data-testid="stMarkdownContainer"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.05);
    color: #f3f4f6;
}

/* =========================
CHAT INPUT
========================= */

section[data-testid="stChatInput"] {
    position: fixed;
    bottom: 28px;
    left: 50%;
    transform: translateX(-50%);
    width: 70%;
    background: rgba(17,24,39,0.82);
    border: 1px solid rgba(139,92,246,0.28);
    border-radius: 24px;
    backdrop-filter: blur(20px);
    padding: 8px 16px;
    transition: all 0.3s ease;
    box-shadow:
        0 0 60px rgba(139,92,246,0.18),
        0 0 120px rgba(59,130,246,0.08);
}

section[data-testid="stChatInput"]:focus-within {
    border: 1px solid rgba(168,85,247,0.6);
    box-shadow:
        0 0 80px rgba(139,92,246,0.25),
        0 0 140px rgba(59,130,246,0.10);
}

/* =========================
INPUT TEXT
========================= */

[data-testid="stChatInputTextArea"] textarea {
    color: white !important;
    caret-color: white !important;
    font-size: 18px !important;
    background: transparent !important;
    -webkit-text-fill-color: white !important;
}

[data-testid="stChatInputTextArea"] textarea::placeholder {
    color: #9ca3af !important;
    opacity: 1 !important;
}

[data-testid="stChatInputTextArea"] {
    background: transparent !important;
}

[data-testid="stChatInputTextArea"] textarea:focus {
    outline: none !important;
    box-shadow: none !important;
    border: none !important;
}

/* =========================
SCROLLBAR
========================= */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #4b5563;
    border-radius: 20px;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# HERO SECTION
# ===============================

hero_html = """
<div class="hero-container">

<div style="font-size:80px; margin-bottom:10px; filter: drop-shadow(0 0 18px rgba(139,92,246,0.35));">
📚
</div>

<div class="hero-title">
Statistical Data <br>
Mining-2 Assistant
</div>

<div class="hero-subtitle">
✨ RHB • Racheal Hageman’s Bot
</div>

<div class="hero-description">
Intelligent AI assistant for Statistical Data Mining-2.
</div>

</div>
"""

st.markdown(hero_html, unsafe_allow_html=True)


# ===============================
# HELPERS
# ===============================
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
# INTENT HELPERS
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
            try:
                docs_all.extend(PyPDFLoader(f).load())
            except Exception as e:
                st.error(f"Error loading {f}: {e}")

    if not docs_all:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    docs = splitter.split_documents(docs_all)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    db = FAISS.from_documents(docs, embeddings)

    return db.as_retriever(
    search_kwargs={"k": 5}
    )

# ===============================
# LOAD DATA
# ===============================
@st.cache_resource
def load_subject_data():

    return build_retriever([
        "14. CART.pdf",
        "biplots_and_outliers.pdf",
        "Overview.pdf",
        "Association_presA.pdf",
        "PGM_I .pdf",
        "PGM_structure.pdf",
        "BN_Prob_Reasoning.pdf",
        "Association_presB.pdf",
        "Association_presD.pdf",
        "Clustering_1-2.pdf",
        "Clustering_2-2.pdf",
        "Clustering_3.pdf",
        "Clustering_4.pdf",
        "Clustering_4b.pdf",
        "SOM.pdf",
        "JC_Clauset.pdf",
        "PCA.pdf",
        "PGM_II_2025.pdf",
        "UDG_A.pdf"
    ])


@st.cache_resource
def load_project_data():

    return build_retriever([
        "SDM-2 Project proposal guidlines.pdf"
    ])

# ===============================
# MODEL
# ===============================
def load_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0,
        max_retries=3
    )

# ===============================
# PROMPT
# ===============================
prompt = ChatPromptTemplate.from_template("""
You are an expert Teaching Assistant for the course Statistical Data Mining-2.

Your responsibilities:
- Help students understand concepts clearly
- Explain topics step-by-step
- Use simple language whenever possible
- Give intuitive examples
- Be concise but educational
- Never hallucinate information
- If information is not found in context, clearly say:
  "I could not find this in the course materials."

CONTEXT:
{context}


QUESTION:
{input}

ANSWER:
""")

# ===============================
# CACHED CHAINS
# ===============================
@st.cache_resource
def get_subject_chain():

    retriever = load_subject_data()
    if retriever is None:
        st.error("No documents loaded.")
        st.stop()

    llm = load_model()

    return create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(
            llm,
            prompt
        )
    )


@st.cache_resource
def get_project_chain():

    retriever = load_project_data()
    if retriever is None:
        st.error("No documents loaded.")
        st.stop()

    llm = load_model()

    return create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(
            llm,
            prompt
        )
    )

# ===============================
# CACHE FUNCTIONS
# ===============================

def cached_subject_response(query):

    chain = get_subject_chain()

    result = chain.invoke({
    "input": query
    })

    return result



def cached_project_response(query):

    chain = get_project_chain()

    result = chain.invoke({
    "input": query
    })

    return result

# ===============================
# SHOW CHAT HISTORY
# ===============================
for msg in st.session_state.messages:

    with st.chat_message(msg["role"], avatar=None):
        st.markdown(msg["content"])

# ===============================
# QUICK PROMPT SUPPORT
# ===============================
if "quick_prompt" in st.session_state:

    query = st.session_state.quick_prompt

    del st.session_state.quick_prompt

else:
    query = st.chat_input("Ask your question")

# ===============================
# MAIN INPUT
# ===============================
if query:

    response = ""

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user", avatar=None):
        st.markdown(query)

    q = query.lower()

    # ===============================
    # ASSIGNMENT MODE EXIT
    # ===============================
    assignment_keywords = [
        "assignment", "homework", "hw",
        "question", "problem", "q1", "q2",
        "calculate", "find", "compute", "implement"
    ]

    is_assignment_related = any(
        word in q for word in assignment_keywords
    )

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
            response = (
                "Professor Zoom: "
                "https://buffalo.zoom.us/j/7342873196"
            )
            st.session_state.zoom_state = None

        elif re.search(r"\bta\b", q):
            response = (
                "TA Zoom: "
                "https://buffalo.zoom.us/j/93740724275"
            )
            st.session_state.zoom_state = None

        elif "grader" in q:
            response = (
                "Grader Zoom: "
                "https://buffalo.zoom.us/j/4027519593"
            )
            st.session_state.zoom_state = None

        else:
            response = (
                "Please choose: Professor / TA / Grader"
            )

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

        response = (
            "Is your doubt related to "
            "subject, assignment, or project?"
        )

    elif st.session_state.context_state == "doubt_type":

        if any(x in q for x in [
            "nothing", "cancel", "never mind",
            "clarified", "solved", "no doubt",
            "something else"
        ]):

            st.session_state.context_state = None

            response = (
                "No problem 😊 "
                "What would you like help with now?"
            )

        elif len(q.split()) <= 3 and has_word(
            q,
            ["hi", "hello", "hey"]
        ):

            st.session_state.context_state = None

            response = (
                "Hello 😊 How can I help you today?"
            )

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

            response = (
                "Please choose: subject / assignment "
                "/ project or type cancel."
            )

    elif st.session_state.context_state == "project_doubt":

        with st.spinner("Thinking..."):

            try:

                result = cached_project_response(
                    query
                )

                response = result["answer"]

                if not response or len(response.strip()) < 5:
                    response = (
                        "I could not find this "
                        "in the course materials."
                    )

            except Exception as e:
                response = (
                    f"I ran into an issue: {str(e)}"
                )

    elif st.session_state.context_state == "subject_doubt":

        with st.spinner("Thinking..."):

            try:

                result = cached_subject_response(
                    query
                )

                response = result["answer"]

                if not response or len(response.strip()) < 5:
                    response = (
                        "I could not find this "
                        "in the course materials."
                    )

            except Exception as e:
                response = (
                    f"I ran into an issue: {str(e)}"
                )

        st.session_state.context_state = None

    # ===============================
    # NORMAL FLOW
    # ===============================
    else:

        intent = detect_normal_intent(q)

        if intent == "greeting":

            response = (
                "Hello! 😊 I'm here to help you. "
                "What would you like to explore today?"
            )

        elif intent == "you_too":

            response = "Thanks 😊"

        elif intent == "thanks":

            response = (
                "You're very welcome 😊 "
                "Happy to help! Have a great day ahead!"
            )

        elif intent == "how_are_you":

            response = (
                "I'm doing great, thanks for asking! 😊 "
                "How can I help you today?"
            )

        elif intent == "bye":

            response = (
                "Goodbye! 👋 Take care and "
                "have a wonderful day 😊"
            )

        elif intent == "who_are_you":

            response = (
                "I am the Statistical "
                "Data Mining-2 Assistant 😊"
            )

        elif intent == "your_name":

            response = "My name is RHB."

        elif intent == "what_is_rhb":

            response = (
                "RHB stands for "
                "Racheal Hageman's Bot."
            )

        elif any(day in q for day in [
            "monday",
            "tuesday",
            "wednesday",
            "thursday"
        ]):

            response = contact_info()

        elif intent == "vacancy":

            response = (
                "Currently, there are no open positions."
            )

        elif any(x in q for x in [
            "research",
            "thesis"
        ]) and "project" not in q:

            response = (
                "Professor is not currently working "
                "on research projects.\n"
                "Contact: hageman@buffalo.edu"
            )

        elif "zoom" in q:

            if "professor" in q:

                response = (
                    "Professor Zoom: "
                    "https://buffalo.zoom.us/j/7342873196"
                )

            elif re.search(r"\bta\b", q):

                response = (
                    "TA Zoom: "
                    "https://buffalo.zoom.us/j/93740724275"
                )

            elif "grader" in q:

                response = (
                    "Grader Zoom: "
                    "https://buffalo.zoom.us/j/4027519593"
                )

            else:

                st.session_state.zoom_state = "ask_person"

                response = (
                    "Whose Zoom link do you need? "
                    "(Professor / TA / Grader)"
                )

        else:

            with st.spinner("Thinking..."):

                try:

                    project_words = [
                        "project",
                        "proposal",
                        "submission",
                        "rate my proposal",
                        "score my proposal",
                        "evaluate proposal",
                        "dataset idea",
                        "project guidelines"
                    ]

                    modified_query = query

                    if "simple" in q:
                        modified_query += (
                            " Explain in very simple words."
                        )

                    if any(word in q for word in project_words):

                        result = cached_project_response(
                            modified_query
                        )

                    else:

                        result = cached_subject_response(
                            modified_query
                        )

                    response = result["answer"]

                    if not response or len(response.strip()) < 5:

                        response = (
                            "I could not find this "
                            "in the course materials."
                        )

                except Exception as e:

                    response = (
                        f"I ran into an issue: {str(e)}"
                    )

    # ===============================
    # MEMORY
    # ===============================
    st.session_state.chat_history.append(
        ("user", query)
    )

    st.session_state.chat_history.append(
        ("assistant", response)
    )

    # ===============================
    # STORE MESSAGES
    # ===============================
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    # ===============================
    # SHOW RESPONSE
    # ===============================
    with st.chat_message("assistant", avatar=None):
        st.markdown(response)


    st.rerun()

