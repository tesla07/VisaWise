"""
VisaWise - Immigration Information Assistant
Streamlit UI for RAG-based immigration chatbot
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from dotenv import load_dotenv
load_dotenv()

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="VisaWise - Immigration Assistant",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional immigration-themed UI
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .main-header p {
        color: #b8d4e8;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Chat container */
    .chat-container {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Message styling */
    .user-message {
        background: #1e3a5f;
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 85%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(30, 58, 95, 0.2);
    }
    
    .assistant-message {
        background: white;
        color: #1e293b;
        padding: 1.25rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 90%;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Disclaimer box */
    .disclaimer-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.85rem;
        color: #78350f;
    }
    
    /* Source cards */
    .source-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }
    
    .source-card:hover {
        border-color: #1e3a5f;
        box-shadow: 0 2px 8px rgba(30, 58, 95, 0.1);
    }
    
    .source-title {
        font-weight: 600;
        color: #1e3a5f;
        margin-bottom: 0.25rem;
    }
    
    .source-meta {
        color: #64748b;
        font-size: 0.75rem;
    }
    
    /* Evaluation metrics */
    .eval-container {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .metric-row {
        display: flex;
        justify-content: space-between;
        padding: 0.25rem 0;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .metric-label {
        color: #475569;
        font-size: 0.85rem;
    }
    
    .metric-value {
        font-weight: 600;
        color: #1e3a5f;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    .sidebar-section h3 {
        color: #1e3a5f;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1.25rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1e3a5f;
        box-shadow: 0 0 0 3px rgba(30, 58, 95, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 58, 95, 0.3);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Spinner */
    .stSpinner > div {
        border-color: #1e3a5f;
    }
</style>
""", unsafe_allow_html=True)


def init_chatbot():
    """Initialize the chatbot (cached for session)."""
    if 'chatbot' not in st.session_state:
        try:
            from qdrant_client import QdrantClient
            from openai import OpenAI
            import os
            import json
            import re
            
            # Import components from rag_chatbot
            sys.path.insert(0, str(Path(__file__).parent / "scripts"))
            from rag_chatbot import (
                ConversationMemory, 
                RetrievedChunk,
                ChatResponse,
                DEEPEVAL_AVAILABLE
            )
            
            # Cross-encoder for reranking (faster and more accurate than LLM)
            CROSS_ENCODER_AVAILABLE = False
            cross_encoder_model = None
            try:
                from sentence_transformers import CrossEncoder
                cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                CROSS_ENCODER_AVAILABLE = True
            except (ImportError, Exception) as e:
                # Fall back to LLM-based reranking if cross-encoder fails to load
                CROSS_ENCODER_AVAILABLE = False
                cross_encoder_model = None
            
            # Configuration
            qdrant_url = os.getenv("QDRANT_URL", "")
            qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
            collection_name = "visawise_immigration"
            model = "gpt-4o-mini"
            
            # Initialize OpenAI
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Try Qdrant connection
            qdrant_client = None
            
            # Option 1: Qdrant Cloud (if URL and API key provided)
            if qdrant_url and qdrant_api_key:
                try:
                    qdrant_client = QdrantClient(
                        url=qdrant_url,
                        api_key=qdrant_api_key,
                        timeout=30
                    )
                    info = qdrant_client.get_collection(collection_name)
                    st.session_state.qdrant_mode = f"cloud ({info.points_count:,} vectors)"
                except Exception as cloud_err:
                    st.session_state.chatbot_ready = False
                    st.session_state.chatbot_error = f"""
Could not connect to Qdrant Cloud.

**Error:** {str(cloud_err)}

Please check your QDRANT_URL and QDRANT_API_KEY in Streamlit secrets.
"""
                    return
            
            # Option 2: Local Qdrant (for development)
            if qdrant_client is None:
                qdrant_path = "./qdrant_db"
                lock_file = Path(qdrant_path) / ".lock"
                
                # Try to clean up stale lock file first
                if lock_file.exists():
                    try:
                        lock_file.unlink()
                    except:
                        pass
                
                try:
                    qdrant_client = QdrantClient(path=qdrant_path)
                    info = qdrant_client.get_collection(collection_name)
                    st.session_state.qdrant_mode = f"local ({info.points_count:,} vectors)"
                except Exception as local_err:
                    st.session_state.chatbot_ready = False
                    st.session_state.chatbot_error = f"""
Could not connect to Qdrant.

**Error:** {str(local_err)}

**For Streamlit Cloud:** Add QDRANT_URL and QDRANT_API_KEY to secrets.

**For local development:** Run `python -m streamlit run app.py`
"""
                    return
            
            # Create a lightweight chatbot class for Streamlit
            class StreamlitChatbot:
                DISCLAIMER = """

---
*⚠️ Disclaimer: Educational purposes only. Not legal advice. Consult an immigration attorney for personalized guidance.*"""

                SYSTEM_PROMPT = """You are VisaWise, an immigration information assistant.

⚠️ CRITICAL SAFETY RULES:
1. PRIORITIZE CONTEXT - Base your answer on the provided context. If the EXACT scenario isn't covered, provide RELATED information that helps.
2. NO ASSUMPTIONS - Never infer spousal/dependent rules from principal rules
3. NO PERSONAL ADVICE - Never say "you would", "in your case", "you may be eligible"
4. Use phrases like "USCIS states that...", "According to USCIS..."
5. ALWAYS cite sources using [1], [2], etc.
6. BE HELPFUL - Share useful background info even if the exact question isn't directly answered.

📋 FORMATTING RULES (IMPORTANT - Make responses readable):
- Use **numbered lists** for steps, requirements, or options
- Use **bullet points** for related items
- Use **bold** for key terms and important phrases
- Break up long text into short paragraphs
- Use headers with ### for different sections if multiple topics

EXAMPLE FORMAT:
"According to USCIS, if your H-1B employment is terminated, you may take one of the following actions [1]:

1. **File for change of nonimmigrant status**
2. **File for adjustment of status**  
3. **Apply for compelling circumstances EAD**
4. **Be beneficiary of a petition to change employer**

**Important:** You must take action within the 60-day grace period [1]."

IF EXACT SCENARIO ISN'T IN CONTEXT:
- Share related background information with citations
- Then note: "The specific scenario of [X] is not directly addressed. For personalized guidance, consult an immigration attorney or your school's DSO."
- Do NOT just say "not covered" if there's useful related info available.

Do NOT add disclaimers - system adds them automatically."""

                def __init__(self):
                    self.client = openai_client
                    self.qdrant = qdrant_client
                    self.collection_name = collection_name
                    self.model = model
                    self.top_k = 10  # Increased for better coverage with reranking
                    self.memory = ConversationMemory(max_turns=10)
                    self.use_rerank = True
                    self.cross_encoder = cross_encoder_model if CROSS_ENCODER_AVAILABLE else None
                
                def _embed_query(self, query):
                    response = self.client.embeddings.create(
                        model="text-embedding-3-small",
                        input=[query]
                    )
                    return response.data[0].embedding
                
                def _extract_keywords(self, query):
                    """Extract immigration-specific keywords from query for metadata filtering."""
                    # Immigration visa categories and terms to look for
                    visa_patterns = [
                        # EB categories
                        r'\bEB-?1A\b', r'\bEB-?1B\b', r'\bEB-?1C\b', r'\bEB-?1\b',
                        r'\bEB-?2\b', r'\bEB-?3\b', r'\bEB-?4\b', r'\bEB-?5\b',
                        # Work visas
                        r'\bH-?1B\b', r'\bH-?2A\b', r'\bH-?2B\b', r'\bH-?4\b',
                        r'\bL-?1A?\b', r'\bL-?1B?\b', r'\bL-?2\b',
                        r'\bO-?1\b', r'\bO-?2\b', r'\bP-?1\b',
                        r'\bTN\b', r'\bE-?1\b', r'\bE-?2\b', r'\bE-?3\b',
                        # Student/exchange visas
                        r'\bF-?1\b', r'\bJ-?1\b', r'\bM-?1\b',
                        r'\bOPT\b', r'\bSTEM\s*OPT\b', r'\bCPT\b',
                        # Family-based
                        r'\bI-?130\b', r'\bI-?485\b', r'\bI-?140\b', r'\bI-?765\b',
                        # Other terms
                        r'\bNIW\b', r'\bPERM\b', r'\bgreen\s*card\b',
                        r'\bnaturalization\b', r'\bcitizenship\b',
                        r'\bgrace\s*period\b', r'\bcap-?gap\b',
                    ]
                    
                    keywords = []
                    for pattern in visa_patterns:
                        matches = re.findall(pattern, query, re.IGNORECASE)
                        keywords.extend([m.upper().replace(' ', '-') for m in matches])
                    
                    # Normalize keywords (e.g., "EB1" -> "EB-1")
                    normalized = []
                    for kw in keywords:
                        normalized_kw = re.sub(r'(EB|H|L|O|P|E|F|J|M|I)(\d)', r'\1-\2', kw)
                        normalized.append(normalized_kw)
                    
                    return list(set(normalized))
                
                def _extract_topic_from_history(self):
                    """Extract the main visa/immigration topic from recent conversation."""
                    if not self.memory or len(self.memory) == 0:
                        return None
                    
                    # Get recent user messages
                    recent_user_msgs = [t.content for t in self.memory.history if t.role == "user"]
                    if not recent_user_msgs:
                        return None
                    
                    # Visa patterns to look for
                    visa_patterns = {
                        'F-1': [r'\bf-?1\b', r'\bf1\b', r'\bstudent visa\b', r'\bstudent status\b'],
                        'H-1B': [r'\bh-?1b\b', r'\bh1b\b'],
                        'EB-1': [r'\beb-?1\b'],
                        'EB-2': [r'\beb-?2\b', r'\bniw\b'],
                        'EB-3': [r'\beb-?3\b'],
                        'EB-5': [r'\beb-?5\b'],
                        'L-1': [r'\bl-?1\b'],
                        'O-1': [r'\bo-?1\b'],
                        'OPT': [r'\bopt\b', r'\bstem opt\b', r'\boptional practical training\b'],
                        'Green Card': [r'\bgreen card\b', r'\bpermanent resident\b', r'\bi-?485\b'],
                        'Naturalization': [r'\bnaturalization\b', r'\bcitizenship\b'],
                        'H-4': [r'\bh-?4\b'],
                        'TN': [r'\btn visa\b', r'\btn\b'],
                        'J-1': [r'\bj-?1\b', r'\bexchange visitor\b'],
                    }
                    
                    # Search from most recent to oldest (last 3 user messages)
                    for msg in reversed(recent_user_msgs[-3:]):
                        msg_lower = msg.lower()
                        for topic, patterns in visa_patterns.items():
                            for pattern in patterns:
                                if re.search(pattern, msg_lower):
                                    return topic
                    return None
                
                def _expand_query(self, query):
                    """Expand query using conversation context - DETERMINISTIC approach."""
                    if not self.memory or len(self.memory) == 0:
                        return query
                    
                    # Check if query is already specific (contains visa category keywords)
                    visa_keywords = ['h-1b', 'h1b', 'eb-1', 'eb-2', 'eb-3', 'f-1', 'f1', 'opt', 'green card', 
                                   'naturalization', 'i-140', 'i-485', 'l-1', 'o-1', 'tn', 'perm', 'niw', 'j-1']
                    if any(kw in query.lower() for kw in visa_keywords):
                        return query  # Already specific, no expansion needed
                    
                    # DETERMINISTIC: Extract topic from conversation history
                    topic = self._extract_topic_from_history()
                    
                    if topic:
                        # Prepend topic to make query specific
                        expanded = f"{topic} visa {query}"
                        return expanded
                    
                    # No topic found, return original query
                    return query
                
                def _rerank_with_llm(self, query, chunks, top_n=5):
                    """Re-rank retrieved chunks using LLM to improve relevance."""
                    if not chunks or len(chunks) <= top_n:
                        return chunks
                    
                    try:
                        chunk_summaries = []
                        for i, chunk in enumerate(chunks):
                            summary = f"[{i}] {chunk.section_heading}: {chunk.text[:200]}..."
                            chunk_summaries.append(summary)
                        
                        rerank_prompt = f"""Score the relevance of each document to the query on a scale of 0-10.
Query: "{query}"

Documents:
{chr(10).join(chunk_summaries)}

Return ONLY a JSON object with document indices as keys and scores as values.
Example: {{"0": 8, "1": 3, "2": 9, "3": 2, "4": 7}}
Focus on exact topic match - e.g., if query asks about "EB-1", documents about "EB-2" should score LOW."""

                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": "You are a document relevance scorer. Return only valid JSON."},
                                {"role": "user", "content": rerank_prompt}
                            ],
                            temperature=0,
                            max_tokens=200,
                            response_format={"type": "json_object"}
                        )
                        
                        scores = json.loads(response.choices[0].message.content)
                        
                        reranked = []
                        for i, chunk in enumerate(chunks):
                            score = scores.get(str(i), scores.get(i, 5))
                            chunk.rerank_score = float(score)
                            reranked.append(chunk)
                        
                        reranked.sort(key=lambda x: (x.rerank_score, x.score), reverse=True)
                        return reranked[:top_n]
                        
                    except Exception as e:
                        return chunks[:top_n]
                
                def _rerank_with_cross_encoder(self, query, chunks, top_n=5):
                    """Re-rank retrieved chunks using cross-encoder model (faster & more accurate)."""
                    if not chunks or len(chunks) <= top_n:
                        return chunks
                    
                    if self.cross_encoder is None:
                        return self._rerank_with_llm(query, chunks, top_n)
                    
                    try:
                        # Create query-document pairs for scoring
                        pairs = [[query, chunk.text] for chunk in chunks]
                        
                        # Get relevance scores from cross-encoder
                        scores = self.cross_encoder.predict(pairs)
                        
                        # Attach scores to chunks
                        for chunk, score in zip(chunks, scores):
                            chunk.rerank_score = float(score)
                        
                        # Sort by cross-encoder score (descending)
                        reranked = sorted(chunks, key=lambda x: x.rerank_score, reverse=True)
                        return reranked[:top_n]
                        
                    except Exception as e:
                        # Fall back to LLM reranking if cross-encoder fails
                        return self._rerank_with_llm(query, chunks, top_n)
                
                def _retrieve(self, query):
                    """Retrieve with keyword-aware multi-term retrieval for comparison queries."""
                    query_vector = self._embed_query(query)
                    keywords = self._extract_keywords(query)
                    
                    all_chunks = []
                    seen_texts = set()
                    
                    # If multiple keywords detected (comparison query), retrieve for each
                    if len(keywords) >= 2:
                        chunks_per_keyword = max(3, self.top_k // len(keywords) + 1)
                        
                        for keyword in keywords:
                            try:
                                from qdrant_client.models import Filter, FieldCondition, MatchText
                                
                                keyword_filter = Filter(
                                    should=[
                                        FieldCondition(key="section_heading", match=MatchText(text=keyword)),
                                        FieldCondition(key="page_title", match=MatchText(text=keyword)),
                                        FieldCondition(key="text", match=MatchText(text=keyword))
                                    ]
                                )
                                
                                results = self.qdrant.query_points(
                                    collection_name=self.collection_name,
                                    query=query_vector,
                                    query_filter=keyword_filter,
                                    limit=chunks_per_keyword
                                )
                                
                                for result in results.points:
                                    text_hash = hash(result.payload.get("text", "")[:100])
                                    if text_hash not in seen_texts:
                                        seen_texts.add(text_hash)
                                        payload = result.payload
                                        chunk = RetrievedChunk(
                                            text=payload.get("text", ""),
                                            section_heading=payload.get("section_heading", ""),
                                            url=payload.get("citation_url", payload.get("url", "")),
                                            page_title=payload.get("page_title", "USCIS"),
                                            score=result.score,
                                            retrieved_at=payload.get("retrieved_at", ""),
                                            last_updated=payload.get("last_updated", ""),
                                            accessed_date=payload.get("accessed_date", "")
                                        )
                                        all_chunks.append(chunk)
                            except Exception:
                                pass
                        
                        # If we got chunks for multiple keywords, rerank and return
                        if len(all_chunks) >= self.top_k:
                            if self.use_rerank:
                                return self._rerank_with_cross_encoder(query, all_chunks, self.top_k)
                            return sorted(all_chunks, key=lambda x: x.score, reverse=True)[:self.top_k]
                    
                    # Fallback: standard vector search (for single-topic or when multi failed)
                    results = self.qdrant.query_points(
                        collection_name=self.collection_name,
                        query=query_vector,
                        limit=self.top_k * 2 if self.use_rerank else self.top_k
                    )
                    
                    chunks = []
                    for result in results.points:
                        payload = result.payload
                        chunk = RetrievedChunk(
                            text=payload.get("text", ""),
                            section_heading=payload.get("section_heading", ""),
                            url=payload.get("citation_url", payload.get("url", "")),
                            page_title=payload.get("page_title", "USCIS"),
                            score=result.score,
                            retrieved_at=payload.get("retrieved_at", ""),
                            last_updated=payload.get("last_updated", ""),
                            accessed_date=payload.get("accessed_date", "")
                        )
                        chunks.append(chunk)
                    
                    if self.use_rerank and keywords:
                        return self._rerank_with_cross_encoder(query, chunks, self.top_k)
                    
                    return chunks[:self.top_k]
                
                def _build_context(self, chunks):
                    parts = []
                    for i, chunk in enumerate(chunks, 1):
                        parts.append(f"[Source {i}] {chunk.section_heading}\nURL: {chunk.url}\nContent: {chunk.text}\n")
                    return "\n---\n".join(parts)
                
                def chat(self, query, evaluate=False):
                    # Expand query using conversation context
                    expanded_query = self._expand_query(query)
                    chunks = self._retrieve(expanded_query)
                    
                    if not chunks:
                        return ChatResponse(
                            answer=f"I couldn't find relevant information.\n{self.DISCLAIMER}",
                            citations=[],
                            sources=[]
                        )
                    
                    context = self._build_context(chunks)
                    
                    # Build messages with conversation history
                    messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
                    
                    # Add conversation history for context
                    if self.memory and len(self.memory) > 0:
                        history_messages = self.memory.get_messages_for_openai()
                        messages.extend(history_messages)
                    
                    # Add current query with context
                    messages.append({
                        "role": "user", 
                        "content": f"Context from USCIS knowledge base:\n{context}\n\nQuestion: {query}"
                    })
                    
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=1000
                    )
                    
                    answer = response.choices[0].message.content
                    
                    # Remove any LLM-added disclaimer
                    if "DISCLAIMER" in answer.upper():
                        for marker in ["⚠️ **IMPORTANT DISCLAIMER**", "**DISCLAIMER**", "DISCLAIMER:"]:
                            if marker in answer:
                                answer = answer.split(marker)[0].strip()
                    
                    answer = f"{answer}\n{self.DISCLAIMER}"
                    
                    # Update memory
                    if self.memory:
                        self.memory.add_user_message(query)
                        self.memory.add_assistant_message(answer)
                    
                    return ChatResponse(
                        answer=answer,
                        citations=[c.citation for c in chunks],
                        sources=chunks,
                        evaluation=None
                    )
                
                def chat_stream(self, query):
                    """Chat with streaming response for real-time display."""
                    # Expand query using conversation context
                    expanded_query = self._expand_query(query)
                    
                    # Retrieve chunks using expanded query
                    chunks = self._retrieve(expanded_query)
                    
                    if not chunks:
                        yield {"type": "complete", "answer": f"I couldn't find relevant information.\n{self.DISCLAIMER}", "chunks": []}
                        return
                    
                    context = self._build_context(chunks)
                    
                    # Build messages with conversation history
                    messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
                    
                    # Add conversation history for context
                    if self.memory and len(self.memory) > 0:
                        history_messages = self.memory.get_messages_for_openai()
                        messages.extend(history_messages)
                    
                    # Add current query with context
                    messages.append({
                        "role": "user", 
                        "content": f"Context from USCIS knowledge base:\n{context}\n\nQuestion: {query}"
                    })
                    
                    # Stream the LLM response
                    stream = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=1000,
                        stream=True
                    )
                    
                    full_answer = ""
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            token = chunk.choices[0].delta.content
                            full_answer += token
                            yield {"type": "token", "content": token}
                    
                    # Clean up disclaimer if LLM added one
                    if "DISCLAIMER" in full_answer.upper():
                        for marker in ["⚠️ **IMPORTANT DISCLAIMER**", "**DISCLAIMER**", "DISCLAIMER:"]:
                            if marker in full_answer:
                                full_answer = full_answer.split(marker)[0].strip()
                    
                    full_answer = f"{full_answer}\n{self.DISCLAIMER}"
                    
                    # Update memory
                    if self.memory:
                        self.memory.add_user_message(query)
                        self.memory.add_assistant_message(full_answer)
                    
                    yield {"type": "complete", "answer": full_answer, "chunks": chunks}
                
                def clear_memory(self):
                    self.memory.clear()
            
            st.session_state.chatbot = StreamlitChatbot()
            st.session_state.chatbot_ready = True
            
        except Exception as e:
            st.session_state.chatbot_ready = False
            st.session_state.chatbot_error = str(e)
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🛂 VisaWise</h1>
        <p>Your AI-powered immigration information assistant • Powered by official USCIS sources</p>
    </div>
    """, unsafe_allow_html=True)


def render_disclaimer():
    """Render the legal disclaimer."""
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Important:</strong> This tool provides <strong>information only</strong> from official USCIS sources. 
        It does NOT provide legal advice. For personalized guidance, please consult a qualified immigration attorney.
    </div>
    """, unsafe_allow_html=True)


def clear_conversation():
    """Callback to clear conversation."""
    st.session_state.messages = []
    if hasattr(st.session_state, 'chatbot') and st.session_state.chatbot:
        st.session_state.chatbot.clear_memory()


def render_sidebar():
    """Render the sidebar with settings and info."""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Clear conversation with callback
        st.button("🗑️ Clear Conversation", use_container_width=True, on_click=clear_conversation)
        
        st.markdown("---")
        
        # Info section
        st.markdown("### 📊 About")
        st.markdown("""
        <div class="sidebar-section">
            <p style="font-size: 0.85rem; color: #475569; margin: 0;">
                VisaWise uses Retrieval-Augmented Generation (RAG) to provide 
                accurate immigration information from official USCIS sources.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Sample questions
        st.markdown("### 💡 Try asking about:")
        sample_questions = [
            "What are EB-1 visa requirements?",
            "How does OPT work for F-1 students?",
            "What is the H-1B cap-gap extension?",
            "EB-2 NIW eligibility criteria",
            "Grace period for F-1 students"
        ]
        for q in sample_questions:
            if st.button(q, key=f"sample_{q}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()


def render_sources(sources):
    """Render source citations - compact view."""
    if not sources:
        return
    
    with st.expander("📚 Sources Used", expanded=False):
        for i, source in enumerate(sources, 1):
            updated_info = f" • 📅 {source.last_updated[:10]}" if source.last_updated else ""
            # Compact single-line format
            st.markdown(f"**[{i}]** [{source.section_heading}]({source.url}){updated_info}", unsafe_allow_html=False)


def render_evaluation(evaluation):
    """Render evaluation metrics."""
    if not evaluation or 'error' in evaluation:
        return
    
    with st.expander("📊 Answer Quality Evaluation", expanded=False):
        method = evaluation.get('evaluation_method', 'unknown').upper()
        st.markdown(f"**Method:** {method}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            faithfulness = evaluation.get('faithfulness', 0)
            relevancy = evaluation.get('relevance', 0)
            st.metric("Faithfulness", f"{faithfulness:.2f}/5")
            st.metric("Answer Relevancy", f"{relevancy:.2f}/5")
        
        with col2:
            contextual = evaluation.get('contextual_relevancy', 0)
            hallucination = evaluation.get('hallucination_score', 0)
            st.metric("Contextual Relevancy", f"{contextual:.2f}/5")
            st.metric("Hallucination", f"{hallucination:.2f}", delta="lower is better", delta_color="inverse")
        
        overall = evaluation.get('overall_score', 0)
        all_passed = evaluation.get('all_passed', False)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Score", f"{overall:.2f}/5")
        with col2:
            status = "✅ All Checks Passed" if all_passed else "⚠️ Some Issues Found"
            st.markdown(f"**Status:** {status}")


def render_chat_message(role, content, sources=None, evaluation=None):
    """Render a single chat message."""
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🛂"):
            st.markdown(content)
            
            if sources:
                render_sources(sources)
            
            if evaluation and st.session_state.show_evaluation:
                render_evaluation(evaluation)


def main():
    """Main application."""
    # Initialize session state FIRST (before any UI that accesses it)
    if 'show_evaluation' not in st.session_state:
        st.session_state.show_evaluation = False  # DeepEval disabled for production
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    init_chatbot()
    render_header()
    render_disclaimer()
    render_sidebar()
    
    # Check if chatbot is ready
    if not st.session_state.get('chatbot_ready', False):
        error = st.session_state.get('chatbot_error', 'Unknown error')
        st.error(f"""
        ⚠️ **Chatbot initialization failed**
        
        Error: {error}
        
        Please ensure:
        1. Qdrant database is set up at `./qdrant_db`
        2. OpenAI API key is configured in `.env`
        3. Required packages are installed
        """)
        return
    
    # Main chat area
    st.markdown("### 💬 Ask about U.S. Immigration")
    
    # Display chat history
    for msg in st.session_state.messages:
        render_chat_message(
            msg['role'], 
            msg['content'],
            msg.get('sources'),
            msg.get('evaluation')
        )
    
    # Check for pending sample question
    if hasattr(st.session_state, 'pending_question'):
        query = st.session_state.pending_question
        del st.session_state.pending_question
    else:
        query = None
    
    # Chat input
    user_input = st.chat_input("Ask a question about U.S. immigration...")
    
    # Use either chat input or sample question
    if user_input:
        query = user_input
    
    if query:
        # Add user message
        st.session_state.messages.append({
            'role': 'user',
            'content': query
        })
        
        # Display user message
        render_chat_message('user', query)
        
        # Generate streaming response
        try:
            with st.chat_message("assistant", avatar="🛂"):
                # Show searching status
                status = st.empty()
                status.markdown("🔍 *Searching USCIS knowledge base...*")
                
                # Stream the response
                response_placeholder = st.empty()
                streamed_text = ""
                chunks = []
                
                for event in st.session_state.chatbot.chat_stream(query):
                    if event["type"] == "token":
                        streamed_text += event["content"]
                        status.empty()  # Clear the searching status
                        response_placeholder.markdown(streamed_text + "▌")
                    elif event["type"] == "complete":
                        streamed_text = event["answer"]
                        chunks = event["chunks"]
                
                # Final render without cursor
                response_placeholder.markdown(streamed_text)
                
                # Show sources
                if chunks:
                    render_sources(chunks)
            
            # Add to session state
            st.session_state.messages.append({
                'role': 'assistant',
                'content': streamed_text,
                'sources': chunks,
                'evaluation': None
            })
                
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
        
        st.rerun()


if __name__ == "__main__":
    main()

