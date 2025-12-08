# VisaWise - AI Immigration Information Assistant

An advanced RAG (Retrieval-Augmented Generation) chatbot that provides factual U.S. immigration information from official USCIS sources. Built with state-of-the-art retrieval techniques and legal safety as a top priority.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green)
![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-purple)

**⚠️ Legal Disclaimer:** This tool provides **informational content only** and does NOT give legal advice. All responses are sourced from official USCIS materials with proper citations. For personalized guidance, consult a qualified immigration attorney.

---

## 🚀 Key Features

### 🧠 Agentic Chunking
Unlike traditional fixed-size chunking, VisaWise uses **LLM-powered intelligent chunking** that:
- Preserves semantic coherence within chunks
- Maintains complete topics without arbitrary splits
- Generates topic summaries for each chunk
- Retains full citation information (URL, page title, section)

### 🔍 Advanced Retrieval Pipeline
```
User Query
    ↓
Query Expansion (resolves pronouns from conversation)
    ↓
Keyword Extraction (EB-1, H-1B, etc.)
    ↓
Metadata Filtering (pre-filters by visa type)
    ↓
Vector Search (text-embedding-3-small)
    ↓
Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
    ↓
Top 10 Most Relevant Chunks → GPT-4o-mini Response
```

### 🎯 Metadata Filtering
Automatically detects visa categories and immigration terms in queries to pre-filter results:
- Visa types: EB-1, EB-2, H-1B, F-1, L-1, O-1, etc.
- Forms: I-140, I-485, I-765, etc.
- Processes: NIW, PERM, naturalization, green card

### ⚡ Cross-Encoder Reranking
Uses a dedicated cross-encoder model (`ms-marco-MiniLM-L-6-v2`) for accurate relevance scoring:
- **50x faster** than LLM-based reranking
- **More accurate** - trained specifically for passage relevance
- **Free** - runs locally, no API costs

### 💬 Conversation Memory
Maintains context across multiple turns:
- Remembers previous questions and answers
- Resolves pronouns ("What about that one?", "Tell me more")
- Expands queries using conversation history
- Supports up to 10 conversation turns

### 🛡️ Legal Safety
Every response includes:
- Source citations with links to official USCIS pages
- Legal disclaimer clarifying it's not legal advice
- Language like "USCIS states..." instead of "you should..."
- Explicit refusal to provide personalized advice

---

## 📦 Installation & Local Setup

### Prerequisites
- Python 3.10+
- OpenAI API key
- (Optional) Qdrant Cloud account for deployment

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/VisaWise.git
cd VisaWise
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-your-openai-api-key

# For Qdrant Cloud (optional, for deployment)
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

### 4. Run the Application

**Option A: Streamlit Web UI**
```bash
python -m streamlit run app.py
```
Open http://localhost:8501 in your browser.

**Option B: Command Line**
```bash
python scripts/rag_chatbot.py "What is the difference between EB-1 and EB-2?"
```

**Interactive CLI Mode:**
```bash
python scripts/rag_chatbot.py --interactive
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│  USCIS Website → Scraper → Agentic Chunking → Embeddings        │
│                              (GPT-4o-mini)    (text-embedding-  │
│                                                3-small)         │
│                                    ↓                            │
│                            Qdrant Cloud                         │
│                         (4,955 vectors)                         │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                      RETRIEVAL PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│  User Query                                                     │
│      ↓                                                          │
│  Query Expansion (conversation context)                         │
│      ↓                                                          │
│  Keyword Extraction (visa types, forms)                         │
│      ↓                                                          │
│  Metadata Filtering (pre-filter by keywords)                    │
│      ↓                                                          │
│  Vector Search (semantic similarity)                            │
│      ↓                                                          │
│  Cross-Encoder Reranking (relevance scoring)                    │
│      ↓                                                          │
│  Top 10 Chunks → GPT-4o-mini → Response with Citations          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
VisaWise/
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not in git)
├── .gitignore               # Git ignore rules
│
├── scripts/
│   ├── rag_chatbot.py       # Main RAG chatbot with all features
│   ├── agentic_chunker.py   # LLM-powered intelligent chunking
│   ├── embed_corpus.py      # Generate embeddings
│   ├── upload_to_qdrant.py  # Upload to Qdrant (local or cloud)
│   └── query_qdrant.py      # Test vector search
│
├── .streamlit/
│   └── secrets.toml.example # Template for Streamlit secrets
│
└── data/                    # Local data (not in git)
    └── embeddings_agentic/  # Agentic chunk embeddings
```

---

## ☁️ Deployment (Streamlit Cloud)

### 1. Upload Embeddings to Qdrant Cloud
```bash
python scripts/upload_to_qdrant.py \
  --embeddings-dir data/embeddings_agentic \
  --qdrant-url YOUR_QDRANT_URL \
  --api-key YOUR_QDRANT_API_KEY
```

### 2. Push Code to GitHub
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 3. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Add secrets in App Settings:
   ```toml
   OPENAI_API_KEY = "sk-..."
   QDRANT_URL = "https://..."
   QDRANT_API_KEY = "..."
   ```
4. Deploy!

---

## 💰 Cost Structure

| Component | Cost |
|-----------|------|
| Qdrant Cloud | **Free** (1GB free tier) |
| Streamlit Cloud | **Free** (public apps) |
| OpenAI API | **~$0.01 per query** |

**Monthly Estimates:**
- 100 queries: ~$1
- 1,000 queries: ~$10
- 10,000 queries: ~$100

---

## 🔧 Configuration Options

### RAG Chatbot Parameters
```python
VisaWiseChatbot(
    top_k=10,           # Number of chunks to retrieve
    model="gpt-4o-mini", # LLM for response generation
    use_rerank=True,     # Enable cross-encoder reranking
    memory_turns=10      # Conversation history length
)
```

### CLI Arguments
```bash
python scripts/rag_chatbot.py "query" \
  --top-k 10 \
  --no-memory \
  --no-rerank \
  --interactive
```

---

## 📚 Sample Queries

- "What is the difference between EB-1 and EB-2?"
- "How do I apply for an H-1B visa?"
- "What are the requirements for naturalization?"
- "Explain the PERM process for green card"
- "What is cap-gap extension for F-1 students?"

---

## ⚖️ Legal Safety

VisaWise is designed to be **informational only**:

**❌ Does NOT:**
- Give legal advice or personalized recommendations
- Make eligibility decisions
- Suggest immigration strategies
- Replace consultation with an attorney

**✅ DOES:**
- Explain publicly available USCIS information
- Cite official sources for every fact
- Include legal disclaimers
- Use objective language ("USCIS states...")

---

## 🤝 Contributing

Contributions are welcome! Please ensure any changes maintain the legal safety guidelines.

## 📄 License

MIT License - See LICENSE file for details.

---

**Built with ❤️ for the immigration community**
