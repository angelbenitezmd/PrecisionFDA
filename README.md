📄 PrecisionFDA RAG AI Assistant

Angel Benitez, 2025
LinkedIn https://www.linkedin.com/in/angel-benitez-md-mba-mshs-msc-6b409717b/

🚀 A Retrieval-Augmented Generation (RAG) AI Assistant for PrecisionFDA
This project is part of the PrecisionFDA Generative AI Community Challenge, providing an AI-powered chatbot that retrieves and summarizes regulatory guidance documents using GPT-4, Claude, Gemini, and LLaMA 2.

🛠️ Features

✅ Upload PDFs: AI-powered search and summarization of FDA regulatory documents
✅ Multiple LLMs: Supports GPT-4, Claude, Gemini, and LLaMA 2
✅ Query Classification: Automatically categorizes user questions
✅ Retrieval-Augmented Generation (RAG): Combines vector search with LLMs
✅ Chat History: Stores user queries and responses with citations
✅ Streamlit UI: Interactive web interface for users

📦 Installation

1️⃣ Clone the Repository

git clone https://github.com/your-username/PrecisionFDA-RAG-Assistant.git
cd PrecisionFDA-RAG-Assistant

2️⃣ Set Up a Virtual Environment (Optional but Recommended)

python -m venv env
source env/bin/activate  # Mac/Linux
env\Scripts\activate  # Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Set Up API Keys (Secrets Management)

Create a .streamlit/secrets.toml file and add your API keys:

DATABASE_URL = "your_postgres_db_url"
OPENAI_API_KEY = "your_openai_key"
GOOGLE_GEMINI_API_KEY = "your_google_gemini_key"
CLAUDE_API_KEY = "your_claude_key"
HUGGINGFACE_API_KEY = "your_huggingface_key"

5️⃣ Run the Streamlit App

streamlit run app.py

🎯 How to Use

📄 Upload PDFs
	1.	Click “Browse Files” in the sidebar.
	2.	Upload FDA regulatory documents (PDF format).
	3.	Click “Process” to generate AI embeddings.

🏛 Choose an AI Persona
	•	Regulatory Expert 🏛️
	•	Manufacturing Specialist 🏭
	•	General User 🌎

💬 Ask Questions
	1.	Type a question in the chat box.
	2.	The AI will retrieve relevant answers from your documents.
	3.	View citations for accuracy.

🧠 Select LLM Model
	•	GPT-4 (Default)
	•	Claude
	•	Gemini
	•	LLaMA 2

📥 Download Chat History
	•	Export chat logs and citations as a CSV file.

🏗️ Project Structure

📂 PrecisionFDA-RAG-Assistant/
├── 📄 LICENSE                # MIT License
├── 📜 README.md              # Project Documentation
├── 📜 requirements.txt       # Dependencies
├── 📂 .streamlit/            
│   ├── secrets.toml          # API Keys (ignored in Git)
├── 📂 data/                  # Uploaded PDFs
├── 📂 models/                # Fine-tuned models (if applicable)
├── 📂 src/                   
│   ├── pdf_processing.py     # Handles PDF uploads
│   ├── llm_handler.py        # Manages different LLMs
│   ├── database.py           # PostgreSQL connection
│   ├── chat.py               # Chatbot logic
│   ├── app.py                # Main Streamlit application

🚀 Future Improvements
	•	Add multi-modal RAG (supporting images & voice inputs)
	•	Implement fine-tuned models for FDA-specific queries
	•	Enhance scalability with GPU deployment options

🤝 Contributing
	1.	Fork this repository
	2.	Create a feature branch (git checkout -b feature-xyz)
	3.	Commit changes (git commit -m "Added new feature")
	4.	Push to GitHub (git push origin feature-xyz)
	5.	Submit a PR for review 🎉

📜 License

MIT License © Angel Benitez, 2025
See LICENSE for details.

📬 Contact

📧 Email: angelbenitezmd@gmail.com
🔗 LinkedIn: Angel Benitez