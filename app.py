import streamlit as st
import time
import concurrent.futures  # Parallel processing
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
from openai import OpenAI

# Load API Keys from Streamlit Secrets
DATABASE_URL = st.secrets["DATABASE_URL"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# Ensure session state variables exist
if "selected_query" not in st.session_state:
    st.session_state["selected_query"] = ""

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = None

# Connect to PostgreSQL (Supabase)
def get_db_connection():
    """Establish a connection to the database using a context manager."""
    return psycopg2.connect(DATABASE_URL)

# Save chat history to PostgreSQL
def save_chat_history(user_input, bot_response, citations):
    """Save user queries and responses to the database."""
    citations_text = " | ".join([f"{c['source']} (Page {c['page']})" for c in citations])

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_history (user_query, bot_response, citations)
                VALUES (%s, %s, %s)
                """,
                (user_input, bot_response, citations_text)
            )
            conn.commit()

# Retrieve past chat history
def load_chat_history():
    """Retrieve the latest chat history from the database."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_query, bot_response, citations FROM chat_history ORDER BY id DESC LIMIT 10")
            return cur.fetchall()

# Process a single PDF file
def process_single_pdf(pdf_file):
    """Process an individual PDF file and extract document chunks."""
    temp_pdf_path = os.path.join(tempfile.gettempdir(), pdf_file.name)
    with open(temp_pdf_path, "wb") as temp_pdf:
        temp_pdf.write(pdf_file.read())

    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

# Cache FAISS vector store using st.cache_resource
@st.cache_resource(show_spinner=True)
def get_vector_store(uploaded_files):
    """Cache the FAISS vector store as a resource (instead of using pickle-based caching)."""
    all_chunks = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_single_pdf, uploaded_files))
    
    for chunks in results:
        all_chunks.extend(chunks)

    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(all_chunks, embeddings)  # ✅ Now properly cached!

# Process multiple PDFs
def process_pdfs(uploaded_files):
    """Processes PDFs and stores embeddings in session state."""
    st.session_state["vector_store"] = get_vector_store(uploaded_files)
    st.session_state["qa_chain"] = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4-turbo", streaming=True),
        retriever=st.session_state["vector_store"].as_retriever(search_kwargs={"k": 15}),
        chain_type="stuff",
        return_source_documents=True
    )

# Query classification using OpenAI API
def classify_query(user_input):
    """Classifies the query into predefined categories."""
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Classify the user query into one of these categories: Regulatory Compliance, Manufacturing Standards, Product Safety, FDA Procedures, General Inquiry. Respond with ONLY the category name."},
            {"role": "user", "content": user_input}
        ]
    )
    category = response.choices[0].message.content.strip()
    return category if category in ["Regulatory Compliance", "Manufacturing Standards", "Product Safety", "FDA Procedures", "General Inquiry"] else "General Inquiry"

# Retrieve relevant documents and generate AI response
def route_query(user_input, selected_category):
    """Fetches relevant documents and generates AI response."""
    if st.session_state["vector_store"] is None or st.session_state["qa_chain"] is None:
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
st.write("Upload PDFs, ask questions, and retrieve AI-powered answers with citations.")

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

# Display Past Queries
if chat_history:
    st.subheader("🕒 Past Queries")
    for idx, (past_query, _, _) in enumerate(chat_history):
        if st.button(f"🔄 {past_query}", key=f"history_{idx}"):
            st.session_state["selected_query"] = past_query

# User Query Input
user_query = st.text_input("Type your question here:", value=st.session_state["selected_query"])

if user_query:
    category = classify_query(user_query)
    selected_category = st.selectbox(
        "Select Query Category (AI Suggested):",
        ["Regulatory Compliance", "Manufacturing Standards", "Product Safety", "FDA Procedures", "General Inquiry"],
        index=["Regulatory Compliance", "Manufacturing Standards", "Product Safety", "FDA Procedures", "General Inquiry"].index(category)
    )

    st.subheader("🤖 AI Response:")
    response_placeholder = st.empty()

    # Retrieve AI Response & Citations
    response, citations = route_query(user_query, selected_category)

    # Format AI Response for better readability
    formatted_response = response.replace("\n", "\n\n")
    response_placeholder.markdown(formatted_response)

    # Display Citations
    if citations:
        st.subheader("📄 Relevant Citations:")
        for citation in citations:
            with st.expander(f"📄 {citation['source']} (Page {citation['page']})", expanded=False):
                st.write(f"**Excerpt:** {citation['text']}")