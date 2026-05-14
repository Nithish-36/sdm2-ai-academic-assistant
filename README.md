# 📘 SDM-2 AI Academic Assistant (RHB Bot)

AI-powered academic assistant designed to help students understand Statistical Data Mining concepts through conversational AI, quizzes, project evaluation, and course support.

---

## 📌 Project Overview

The SDM-2 AI Academic Assistant is an educational AI chatbot developed to support students learning Statistical Data Mining concepts.

The system helps students:
- Understand complex topics with simplified explanations
- Generate quizzes for self-practice
- Get project guidance based on course guidelines
- Access course-related academic support
- Interact through a conversational Streamlit interface

The assistant uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded course PDFs before generating contextual responses.

---

## 🚀 Features

### ✅ Statistical Data Mining Concept Support
Students can ask questions related to:
- Clustering
- PCA
- Bayesian Networks
- Association Rules
- Self-Organizing Maps
- Probabilistic Graphical Models
- CART
- Recommendation Systems
- And other SDM topics

---

### ✅ AI-Powered PDF Question Answering
The system retrieves relevant information from uploaded lecture PDFs and course materials before generating responses.

This improves:
- Context awareness
- Concept understanding
- Response relevance

---

### ✅ Quiz Generation
Students can request quizzes on different SDM topics for self-assessment and concept revision.

---

### ✅ Project Guidance Support
The assistant can evaluate and guide students based on project proposal guidelines and course expectations.

---

### ✅ Conversational Learning Interface
The chatbot provides an interactive learning experience through a Streamlit-based conversational interface.

---

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application framework |
| LangChain | RAG workflow orchestration |
| FAISS | Vector database for semantic retrieval |
| Gemini AI | Conversational response generation |
| HuggingFace Embeddings | Text embeddings generation |
| PyPDFLoader | PDF document processing |

---

## 🔄 System Workflow

1. Upload course PDFs and lecture materials
2. Split PDFs into text chunks
3. Generate embeddings from text
4. Store vectors in FAISS database
5. Student asks question
6. Relevant content is retrieved
7. Gemini AI generates contextual response
8. Streamlit displays conversational output

---

## ⚙️ Key Functionalities

- SDM topic explanations
- Conversational academic support
- Quiz generation
- Project proposal guidance
- PDF-based semantic search
- Interactive AI learning experience

---

## 📊 Learning Outcomes

Through this project, I gained hands-on experience with:
- Retrieval-Augmented Generation (RAG)
- Conversational AI systems
- Semantic search workflows
- Prompt engineering
- Educational AI applications
- Vector databases
- Streamlit deployment

---

## ▶️ Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Nithish-36/sdm2-ai-academic-assistant.git