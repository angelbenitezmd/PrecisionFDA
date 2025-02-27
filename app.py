import streamlit as st
import time
import concurrent.futures
import pandas as pd
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
import anthropic
import google.generativeai as genai

# ✅ Set Page Config
st.set_page_config(page_title="📄 RAG AI PDF Assistant", layout="wide")

# ✅ Load API Keys
DATABASE_URL = st.secrets["DATABASE_URL"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
CLAUDE_API_KEY = st.secrets["CLAUDE_API_KEY"]
GOOGLE_GEMINI_API_KEY = st.secrets["GOOGLE_GEMINI_API_KEY"]

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["CLAUDE_API_KEY"] = CLAUDE_API_KEY
os.environ["GOOGLE_GEMINI_API_KEY"] = GOOGLE_GEMINI_API_KEY

# ✅ Initialize OpenAI Client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ Initialize Claude Client (Anthropic)
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# ✅ Initialize Gemini Client (Google AI)
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

# ✅ Connect to PostgreSQL
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

# ✅ Save chat history
def save_chat_history(user_input, bot_response, citations):
    citations_text = " | ".join([f"{c['source']} (Page {c['page']})" for c in citations])
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO chat_history (user_query, bot_response, citations) VALUES (%s, %s, %s)", 
                        (user_input, bot_response, citations_text))
            conn.commit()

# ✅ Retrieve chat history
def load_chat_history():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_query, bot_response, citations FROM chat_history ORDER BY id DESC LIMIT 10")
            return cur.fetchall()

# ✅ Export Chat History as CSV
def export_chat_history():
    history = load_chat_history()
    df = pd.DataFrame(history, columns=["Query", "Response", "Citations"])
    csv = df.to_csv(index=False).encode("utf-8")
    return csv

# ✅ Process PDFs
def process_pdfs(uploaded_files):
    def process_single_pdf(pdf_file):
        temp_pdf_path = os.path.join(tempfile.gettempdir(), pdf_file.name)
        with open(temp_pdf_path, "wb") as temp_pdf:
            temp_pdf.write(pdf_file.read())
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return text_splitter.split_documents(documents)
    
    all_chunks = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_single_pdf, uploaded_files))
    
    for chunks in results:
        all_chunks.extend(chunks)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(all_chunks, embeddings)

    # ✅ Store in session state
    st.session_state["vector_store"] = vector_store
    st.session_state["qa_chain"] = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4-turbo", temperature=0, streaming=True),
        retriever=vector_store.as_retriever(search_kwargs={"k": 15}),
        chain_type="stuff",
        return_source_documents=True
    )

# ✅ Query classification
def classify_query(user_input):
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "Classify the user query into Regulatory Compliance, Manufacturing Standards, Product Safety, FDA Procedures, or General Inquiry."},
                  {"role": "user", "content": user_input}]
    )
    category = response.choices[0].message.content.strip()
    return category if category in ["Regulatory Compliance", "Manufacturing Standards", "Product Safety", "FDA Procedures", "General Inquiry"] else "General Inquiry"

# ✅ Retrieve relevant documents
def route_query(user_input, selected_category, persona):
    if "vector_store" not in st.session_state or "qa_chain" not in st.session_state:
        return "❌ Error: Please upload and process a document first.", []
    
    retriever = st.session_state["vector_store"].as_retriever(search_kwargs={"k": 15})
    retrieved_docs = retriever.get_relevant_documents(user_input)

    if not retrieved_docs:
        return "⚠️ No relevant documents found for your query.", []

    response = st.session_state["qa_chain"].invoke({"query": user_input})

    citations = []
    for doc in response.get("source_documents", []):
        citations.append({"source": os.path.basename(doc.metadata.get("source", "Unknown.pdf")),
                          "page": doc.metadata.get("page", "N/A"),
                          "text": doc.page_content[:400]})

    return response["result"], citations

# ✅ Streamlit UI
# ✅ Introduction: Purpose of the App
st.title("📄 RAG AI Assistant for FDA Regulatory Documents")
st.write("""
This AI-powered assistant helps users efficiently navigate **FDA regulatory documents**.  
By leveraging **Retrieval-Augmented Generation (RAG)**, the app enables users to:
- Upload **PDF documents** containing regulatory guidance.
- Ask **questions** about the documents.
- Get **AI-powered responses** with citations from the uploaded content.
- Select different **AI personas** based on their expertise needs.
- Choose between **multiple LLMs** for response generation.

This tool is ideal for **regulatory professionals, pharmaceutical manufacturers, and researchers** who need  
quick, reliable, and **context-aware answers** from large regulatory texts.
""")

# ℹ️ **Info Tabs**
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    **1️⃣ Upload PDFs** 📤  
    - Click 'Browse Files' in the sidebar to upload your documents.  
    - Hit 'Process' to generate embeddings.  

    **2️⃣ Select Persona** 👤 *(What kind of assistant do you need?)*  
    - **Regulatory Expert** 🏛️: Ideal for FDA regulations, legal, and compliance-related queries.  
    - **Manufacturing Specialist** 🏭: Best for queries related to pharmaceutical and industrial manufacturing standards.  
    - **Product Safety Analyst** 🛡️: Questions on risk assessments, toxicity, and safety compliance.  
    - **FDA Procedures Guide** 📑: Covers approval processes, documentation, and FDA guidelines.  
    - **General User** 🌎: Suitable for everyday use, providing easy-to-understand responses.

    **3️⃣ Ask Questions** 💬  
    - Type a question in the input box.  
    - The AI will retrieve relevant answers and sources.  

    **4️⃣ Select Category** 🎯 *(Refine the focus of your query)*  
    - **Regulatory Compliance** ⚖️: Queries related to FDA rules, industry regulations, and compliance.  
    - **Manufacturing Standards** 🏭: Best for Good Manufacturing Practices (GMP) and industry safety standards.  
    - **Product Safety** 🛡️: Questions on risk assessments, toxicity, and safety compliance.  
    - **FDA Procedures** 📑: Covers approval processes, documentation, and FDA guidelines.  
    - **General Inquiry** 🤔: Default for other questions not covered by the categories.  

    **5️⃣ View Citations** 📄  
    - Expand 'Relevant Citations' to verify answers.  

    **6️⃣ Export Chat History** 📝  
    - Click 'Download Chat History' to save past queries.  
    """)

# ✅ **Sidebar**
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/7d/Food_and_Drug_Administration_logo.svg", width=150)
    st.title("⚙️ Settings")
    persona = st.radio("Persona Selection", ["Regulatory Expert", "Manufacturing Specialist", "Product Safety Analyst", "FDA Procedures Guide", "General User"])
    
    st.subheader("📄 Your Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    st.subheader("🧠 Model Selection")
    model_choice = st.selectbox("Choose a model:", ["OpenAI GPT-4", "Claude", "Gemini"])

    if st.button("Process"):
        if uploaded_files:
            with st.spinner("Processing PDFs... Please wait."):
                process_pdfs(uploaded_files)
            st.success("Processing complete! Ask a question below.")

# ✅ **Chat History**
st.subheader("💬 Chat")
chat_history = load_chat_history()

# ✅ **Chat History Download**
csv_data = export_chat_history()
st.download_button(label="📥 Download Chat History", data=csv_data, file_name="chat_history.csv", mime="text/csv")

# ✅ **Collapsible Past Queries**
with st.expander("🕒 Past Queries"):
    for idx, (past_query, _, _) in enumerate(chat_history):
        if st.button(f"🔄 {past_query}", key=f"history_{idx}"):
            st.session_state["user_query"] = past_query  

# ✅ **User Query**
user_query = st.text_input("💬 Type your question here:", value=st.session_state.get("user_query", ""))

if user_query:
    category = classify_query(user_query)
    st.subheader("🤖 AI Response:")
    response, citations = route_query(user_query, category, persona)
    st.markdown(response.replace("\n", "\n\n"))

    # Save chat history
    save_chat_history(user_query, response, citations)

    if citations:
        st.subheader("📄 Citations")
        for citation in citations:
            with st.expander(f"📄 {citation['source']} (Page {citation['page']})"):
                st.write(f"**Excerpt:** {citation['text']}")