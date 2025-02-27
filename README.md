📄 PrecisionFDA RAG AI Assistant

Angel Benitez, 2025
LinkedIn: https://www.linkedin.com/in/angel-benitez-md-mba-mshs-msc-6b409717b/

🚀 A Retrieval-Augmented Generation (RAG) AI Assistant for PrecisionFDA
This project is part of the PrecisionFDA Generative AI Community Challenge, providing an AI-powered chatbot that retrieves and summarizes regulatory guidance documents using GPT-4, Claude, Gemini, and LLaMA.

🛠️ Features

✅ Upload PDFs – AI-powered search and summarization of FDA regulatory documents
✅ Multiple LLMs – Supports GPT-4, Claude, Gemini, and LLaMA
✅ Query Classification – Automatically categorizes user questions
✅ Retrieval-Augmented Generation (RAG) – Combines vector search with LLMs for precise answers
✅ Chat History & Citations – Stores responses and provides references to source documents
✅ Streamlit UI – Interactive web interface

📦 Installation Guide

1️⃣ Clone the Repository

git clone https://github.com/angelbenitezmd/PrecisionFDA.git
cd PrecisionFDA

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
	2.	Upload FDA regulatory documents in PDF format.
	3.	Click “Process” to generate AI-powered document embeddings.

🏛 Choose an AI Persona
	•	Regulatory Expert 🏛️ – Focused on FDA regulations, compliance, and legal queries
	•	Manufacturing Specialist 🏭 – Covers Good Manufacturing Practices (GMP) and industry safety standards
	•	Product Safety Analyst 🛡️ – Provides insights on risk assessments, toxicity, and safety compliance
	•	FDA Procedures Guide 📑 – Assists with FDA approval processes, documentation, and submission guidelines
	•	General User 🌎 – Offers easy-to-understand answers for non-technical users

💬 Ask Questions
	1.	Type your question in the chat box.
	2.	The AI retrieves answers from uploaded documents.
	3.	View citations for accuracy and reference.

🧠 Select LLM Model
	•	GPT-4 (Default)
	•	Claude
	•	Gemini
	•	LLaMA

📥 Download Chat History
	•	Export chat logs and citations as a CSV file

🏗️ Project Structure

📂 PrecisionFDA/
├── 📄 LICENSE                # MIT License
├── 📜 README.md              # Project Documentation
├── 📜 requirements.txt       # Dependencies
├── 📂 .streamlit/            
│   ├── secrets.toml          # API Keys (ignored in Git)
├── 📂 data/                  # Uploaded PDFs
│   ├── MERGED_cosmetic_guidances.pdf  # Example PDF
├── 📜 .gitignore             # Git ignore file
├── 📜 app.py                 # Main Streamlit application

🚀 Future Improvements
	•	Multi-modal RAG – Supporting images & voice inputs
	•	Fine-tuned models – Optimized for FDA-specific queries
	•	Scalability Enhancements – GPU-based deployment

🤝 Contributing

Want to contribute? Follow these steps:
	1.	Fork this repository
	2.	Create a feature branch

git checkout -b feature-xyz


	3.	Commit changes

git commit -m "Added new feature"


	4.	Push to GitHub

git push origin feature-xyz


	5.	Submit a PR for review 🎉

📜 License

MIT License © Angel Benitez, 2025
See LICENSE for details.

📬 Contact

📧 Email: angelbenitezmd@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/angel-benitez-md-mba-mshs-msc-6b409717b/
🔗 GitHub Repository: https://github.com/angelbenitezmd/PrecisionFDA

🌟 Purpose and Use of the App

The PrecisionFDA RAG AI Assistant helps users efficiently navigate FDA regulatory documents.
By leveraging Retrieval-Augmented Generation (RAG), the app enables users to:

✅ Upload PDFs containing regulatory guidance
✅ Ask questions about FDA compliance, manufacturing, or product safety
✅ Get AI-powered responses with citations from the uploaded documents
✅ Select AI personas based on specific regulatory needs
✅ Choose between multiple AI models to generate responses

🔹 Key Use Cases

💡 Regulatory Professionals – Quickly access FDA regulations and compliance information
🏭 Pharmaceutical Manufacturers – Retrieve GMP guidelines and industry safety standards
🛡 Product Safety Officers – Get guidance on risk assessments and safety compliance
🔬 Researchers & Biotech Companies – Navigate FDA approval processes and documentation
🌎 General Users – Receive easy-to-understand answers to FDA-related questions

🎥 Live Demo

🚀 A live demo will be available soon! Stay tuned.

🎉 Why This README is Improved

✅ No shortened links or hyperlinked text
✅ Better structure & clear sections
✅ Step-by-step installation & usage guide
✅ AI personas with detailed descriptions
✅ Precise formatting for readability