import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import psycopg2
import openai
import tempfile
import os
from openai import OpenAI  # Import OpenAI's latest API client

# Load API Keys from Streamlit Secrets
DATABASE_URL = st.secrets["DATABASE_URL"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# Connect to PostgreSQL (Supabase)
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

# Save chat history to the database
def save_chat_history(user_input, bot_response, citations):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO chat_history (user_query, bot_response, citations)
        VALUES (%s, %s, %s)
        """,
        (user_input, bot_response, " | ".join(citations))
    )
    conn.commit()
    cur.close()
    conn.close()

# Retrieve chat history from the database
def load_chat_history():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT user_query, bot_response, citations FROM chat_history ORDER BY id DESC LIMIT 10")
    history = cur.fetchall()
    cur.close()
    conn.close()
    return history

# Process PDFs and store embeddings
def process_pdfs(uploaded_files):
    all_chunks = []
    
    for pdf_file in uploaded_files:
        temp_pdf_path = os.path.join(tempfile.gettempdir(), pdf_file.name)
        with open(temp_pdf_path, "wb") as temp_pdf:
            temp_pdf.write(pdf_file.read())
        
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(all_chunks, embeddings)

    # Store processed documents in session state
    st.session_state["vector_store"] = vector_store
    st.session_state["qa_chain"] = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4-turbo", streaming=True),
        retriever=vector_store.as_retriever(search_kwargs={"k": 15}),
        chain_type="stuff",
        return_source_documents=True
    )

# Query Classification with Updated OpenAI API
def classify_query(user_input):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Classify the user query into one of these categories: Regulatory Compliance, Manufacturing Standards, Product Safety, FDA Procedures, General Inquiry. Respond with ONLY the category name."},
            {"role": "user", "content": user_input}
        ]
    )

    category = response.choices[0].message.content.strip()
    valid_categories = ["Regulatory Compliance", "Manufacturing Standards", "Product Safety", "FDA Procedures", "General Inquiry"]
    return category if category in valid_categories else "General Inquiry"

# Query Routing with Enhanced Formatting
def route_query(user_input, selected_category):
    if "vector_store" not in st.session_state or "qa_chain" not in st.session_state:
        return "❌ Error: Please upload and process a document first.", []

    retriever = st.session_state["vector_store"].as_retriever(search_kwargs={"k": 15})
    retrieved_docs = retriever.get_relevant_documents(user_input)

    if not retrieved_docs:
        return "⚠️ No relevant documents found for your query.", []

    response = st.session_state["qa_chain"].invoke({"query": user_input})

    citations = []
    seen_citations = set()

    if "source_documents" in response:
        for doc in response["source_documents"]:
            source_name = os.path.basename(doc.metadata.get("source", "Unknown.pdf"))
            page_number = doc.metadata.get("page", "N/A")
            text_excerpt = doc.page_content[:400]  

            citation_key = (source_name, page_number)
            if citation_key in seen_citations:
                continue
            seen_citations.add(citation_key)

            citations.append({
                "source": source_name,
                "page": page_number,
                "text": text_excerpt
            })

    return response["result"], citations

# Streamlit UI
st.set_page_config(page_title="📄 RAG AI PDF Assistant", layout="wide")
st.title("📄 RAG AI Assistant for PDFs")
st.write("Upload multiple PDFs, ask questions, and use AI-powered retrieval.")

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/7d/Food_and_Drug_Administration_logo.svg", width=150)
    st.title("⚙️ Settings")
    persona = st.selectbox("Select Chatbot Persona", ["Regulatory Expert", "Manufacturing Specialist", "General User"])
    
    st.subheader("📄 Your Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process"):
        if uploaded_files:
            st.success(f"Processing {len(uploaded_files)} PDFs...")
            process_pdfs(uploaded_files)
            st.success("Processing complete! Ask a question below.")

# Chat UI
st.subheader("💬 Chat with Your PDFs")
chat_history = load_chat_history()

# **Display Past Queries**
if chat_history:
    st.subheader("🕒 Past Queries")
    for idx, (past_query, past_response, past_citations) in enumerate(chat_history):
        if st.button(f"🔄 {past_query}", key=f"history_{idx}"):
            user_query = past_query

# User Query Input
user_query = st.text_input("Type your question here:")

if user_query:
    category = classify_query(user_query)
    selected_category = st.selectbox(
        "Select Query Category (AI Suggested):",
        ["Regulatory Compliance", "Manufacturing Standards", "Product Safety", "FDA Procedures", "General Inquiry"],
        index=["Regulatory Compliance", "Manufacturing Standards", "Product Safety", "FDA Procedures", "General Inquiry"].index(category)
    )

    st.subheader("🤖 AI Response:")
    response_placeholder = st.empty()
    response_text = ""

    # **Formatted AI Response**
    response, citations = route_query(user_query, selected_category)
    formatted_response = response.replace("1.", "\n### 1.").replace("2.", "\n### 2.").replace("3.", "\n### 3.").replace("4.", "\n### 4.").replace("5.", "\n### 5.").replace("6.", "\n### 6.").replace("7.", "\n### 7.").replace("8.", "\n### 8.").replace("9.", "\n### 9.").replace("10.", "\n### 10.")

    response_placeholder.markdown(formatted_response)

    # **Display Citations**
    if citations:
        st.subheader("📄 Relevant Citations:")
        for citation in citations:
            with st.expander(f"📄 {citation['source']} (Page {citation['page']})"):
                st.write(f"**Excerpt:** {citation['text']}")