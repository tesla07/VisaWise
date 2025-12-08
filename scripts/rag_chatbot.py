"""
RAG Chatbot for VisaWise Immigration Assistant

Features:
- Retrieves relevant chunks from Qdrant
- Generates answers using GPT-4o-mini
- Includes citations in responses
- Evaluates answer quality (faithfulness, relevance, completeness)
- Conversation memory for multi-turn context
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

try:
    from qdrant_client import QdrantClient
except ImportError:
    print("Install with: pip install qdrant-client")
    exit(1)

# DeepEval for RAG evaluation
DEEPEVAL_AVAILABLE = False
try:
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        HallucinationMetric
    )
    DEEPEVAL_AVAILABLE = True
except ImportError:
    pass  # DeepEval not installed, will use fallback evaluation

# Cross-encoder for reranking
CROSS_ENCODER_AVAILABLE = False
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    pass  # Will fall back to LLM-based reranking

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONVERSATION MEMORY
# =============================================================================

@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    sources_summary: Optional[str] = None  # Brief summary of sources used


class ConversationMemory:
    """Manages conversation history for multi-turn context."""
    
    def __init__(self, max_turns: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_turns: Maximum number of turns to keep (older ones are dropped)
        """
        self.max_turns = max_turns
        self.history: List[ConversationTurn] = []
        self.topic_context: Optional[str] = None  # Current topic being discussed
    
    def add_user_message(self, message: str):
        """Add a user message to history."""
        self.history.append(ConversationTurn(role="user", content=message))
        self._trim_history()
    
    def add_assistant_message(self, message: str, sources_summary: str = None):
        """Add an assistant message to history."""
        self.history.append(ConversationTurn(
            role="assistant", 
            content=message,
            sources_summary=sources_summary
        ))
        self._trim_history()
    
    def _trim_history(self):
        """Keep only the last max_turns * 2 messages (user + assistant pairs)."""
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]
    
    def get_context_for_llm(self) -> str:
        """Format conversation history for LLM context."""
        if not self.history:
            return ""
        
        context_parts = ["Previous conversation:"]
        for turn in self.history[-6:]:  # Last 3 exchanges
            role_label = "User" if turn.role == "user" else "Assistant"
            context_parts.append(f"{role_label}: {turn.content[:500]}")  # Truncate long messages
        
        return "\n".join(context_parts)
    
    def get_messages_for_openai(self) -> List[Dict[str, str]]:
        """Format history as OpenAI messages format."""
        messages = []
        for turn in self.history[-6:]:  # Last 3 exchanges
            messages.append({
                "role": turn.role,
                "content": turn.content
            })
        return messages
    
    def clear(self):
        """Clear all conversation history."""
        self.history = []
        self.topic_context = None
        logger.info("🗑️ Conversation memory cleared")
    
    def get_last_topic(self) -> Optional[str]:
        """Extract the main topic from recent conversation."""
        # Look at recent user messages to identify the topic
        user_messages = [t.content for t in self.history if t.role == "user"]
        if user_messages:
            return user_messages[-1]  # Most recent user question
        return None
    
    def __len__(self) -> int:
        return len(self.history)
    
    def __bool__(self) -> bool:
        return len(self.history) > 0


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector database."""
    text: str
    section_heading: str
    url: str
    page_title: str
    score: float
    citation: str = ""
    retrieved_at: str = ""  # When USCIS page was scraped
    last_updated: str = ""  # When USCIS page was last modified
    accessed_date: str = ""  # When chunk was processed
    rerank_score: float = 0.0  # LLM reranking score (0-10)
    
    def __post_init__(self):
        if not self.citation:
            self.citation = f"[{self.page_title} - {self.section_heading}]({self.url})"
    
    @property
    def freshness_info(self) -> str:
        """Return formatted freshness/timestamp info for user awareness."""
        parts = []
        if self.last_updated:
            parts.append(f"Page updated: {self.last_updated[:10]}")
        if self.retrieved_at:
            parts.append(f"Scraped: {self.retrieved_at[:10]}")
        return " | ".join(parts) if parts else ""


@dataclass
class ChatResponse:
    """Response from the chatbot with evaluation."""
    answer: str
    citations: List[str]
    sources: List[RetrievedChunk]
    evaluation: Optional[Dict[str, Any]] = None


class VisaWiseChatbot:
    """RAG-based immigration chatbot with answer evaluation and conversation memory."""
    
    SYSTEM_PROMPT = """You are VisaWise, an immigration information assistant that provides factual information from official USCIS sources.

⚠️ CRITICAL SAFETY RULES (VIOLATION = IMMEDIATE STOP):
1. **ANSWER ONLY WHAT'S IN CONTEXT** - If information is not explicitly stated in the provided context, say "This specific scenario is not covered in the USCIS information provided."
2. **NO ASSUMPTIONS** - NEVER infer spousal/dependent rules from principal applicant rules. They are NOT interchangeable.
3. **NO PERSONAL ADVICE** - NEVER say "you would", "in your case", "you may be eligible", "you should". Only state what USCIS documents say.
4. **COMPLEX SCENARIOS** - If the query involves multiple people, relationships, or conditions not explicitly addressed together in the context, say "This scenario involves multiple factors not explicitly covered together in the available USCIS information. Please consult an immigration attorney."
5. **NO SPECULATION** - NEVER interpret, extrapolate, or combine rules that aren't explicitly combined in the source material.

🚨 F-1/H-1B SPECIFIC RULES:
- Cap-gap extensions apply ONLY to the principal F-1 student filing for H-1B, NEVER to spouses or dependents
- F-2 dependent rules are NOT the same as F-1 principal rules - do not mix them
- H-4 spousal rules are NOT the same as H-1B principal rules - do not mix them
- If asked about spouse/dependent scenarios not explicitly in context, state this limitation clearly

GENERAL RULES:
1. You provide INFORMATION ONLY - you do NOT give advice, recommendations, or suggestions
2. ONLY use information from the provided context to answer questions
3. Use phrases like "USCIS states that...", "According to USCIS...", "The official information indicates..."
4. ALWAYS cite your sources using [1], [2], etc. format
5. When the user refers to something from conversation history, use context to understand what they mean

FORMAT YOUR RESPONSE:
- State facts as: "USCIS states: [fact from context] [1]"
- Use clear paragraphs with citations after each fact
- End with a "Sources:" section listing [1], [2], etc.
- Do NOT add any disclaimer - the system will add one automatically

IF UNCLEAR OR NOT IN CONTEXT:
Say: "This specific scenario is not covered in the USCIS information provided. For personalized guidance, please consult an immigration attorney.\""""

    QUERY_EXPANSION_PROMPT = """You are a query expansion assistant. Your job is to rewrite user queries that contain pronouns or references to previous conversation into standalone, searchable queries.

Given the conversation history and the current query, rewrite the query to be self-contained and specific.

Examples:
- History: "What are EB-1B requirements?" -> Current: "Summarize that" -> Rewritten: "Summarize the EB-1B visa requirements"
- History: "Explain EB-2 NIW" -> Current: "What about its advantages?" -> Rewritten: "What are the advantages of EB-2 NIW?"
- History: "Tell me about H-1B" -> Current: "Is it faster than the one before?" -> Rewritten: "Is H-1B processing faster than the previously discussed visa category?"

If the query is already self-contained, return it unchanged.
Return ONLY the rewritten query, nothing else."""

    DISCLAIMER = """
⚠️ **IMPORTANT DISCLAIMER**
This information is provided for educational purposes only and is sourced from official USCIS materials. This chatbot does NOT provide legal advice and is NOT responsible for any decisions made based on this information. Immigration laws are complex and subject to change. For personalized advice regarding your specific situation, please consult a qualified immigration attorney."""

    EVALUATION_PROMPT = """You are an expert evaluator assessing the quality of an AI immigration assistant's response.

Evaluate the following response on these criteria (score 1-5):

1. **Faithfulness** (1-5): Does the answer only use information from the provided context? No hallucinations?
2. **Relevance** (1-5): Does the answer directly address the user's question?
3. **Completeness** (1-5): Does the answer cover all important aspects from the available context?
4. **Citation Accuracy** (1-5): Are citations used correctly and do they support the claims?
5. **Clarity** (1-5): Is the answer clear, well-organized, and easy to understand?

Return a JSON object with:
{
    "faithfulness": <score>,
    "relevance": <score>,
    "completeness": <score>,
    "citation_accuracy": <score>,
    "clarity": <score>,
    "overall_score": <average>,
    "issues": ["list of any issues found"],
    "suggestions": ["list of improvement suggestions"]
}

ONLY return the JSON object, no other text."""

    def __init__(
        self,
        qdrant_path: str = "./qdrant_db",
        collection_name: str = "visawise_immigration",
        model: str = "gpt-4o-mini",
        top_k: int = 10,  # Increased for better coverage with cross-encoder reranking
        memory_turns: int = 10,
        use_deepeval: bool = False,  # Disabled for production
        use_rerank: bool = True
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name
        self.model = model
        self.top_k = top_k
        self.use_deepeval = use_deepeval and DEEPEVAL_AVAILABLE
        self.use_rerank = use_rerank
        
        if use_deepeval and not DEEPEVAL_AVAILABLE:
            logger.warning("DeepEval requested but not installed. Install with: pip install deepeval")
            logger.warning("Falling back to LLM-based evaluation.")
        elif self.use_deepeval:
            logger.info("✅ DeepEval enabled for RAG evaluation")
        
        # Initialize cross-encoder for reranking
        self.cross_encoder = None
        if self.use_rerank:
            if CROSS_ENCODER_AVAILABLE:
                try:
                    self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                    logger.info("✅ Cross-encoder reranking enabled (ms-marco-MiniLM-L-6-v2)")
                except Exception as e:
                    logger.warning(f"Failed to load cross-encoder: {e}. Falling back to LLM reranking.")
            else:
                logger.info("✅ LLM reranking enabled (install sentence-transformers for faster cross-encoder)")
        
        # Initialize conversation memory
        self.memory = ConversationMemory(max_turns=memory_turns)
        
        # Verify connection
        try:
            info = self.qdrant.get_collection(collection_name)
            logger.info(f"✅ Connected to Qdrant: {info.points_count:,} vectors")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def _expand_query(self, query: str) -> str:
        """
        Expand a query that may contain pronouns/references using conversation history.
        Returns a self-contained, searchable query.
        """
        # Check if query likely needs expansion (has pronouns or is very short)
        needs_expansion_indicators = [
            "that", "this", "it", "its", "they", "them", "those", "these",
            "the one", "same", "other", "previous", "before", "above",
            "summarize", "explain more", "tell me more", "what about"
        ]
        
        query_lower = query.lower()
        needs_expansion = any(indicator in query_lower for indicator in needs_expansion_indicators)
        
        # If no history or doesn't need expansion, return original
        if not self.memory or not needs_expansion:
            return query
        
        try:
            # Get conversation context
            history_context = self.memory.get_context_for_llm()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.QUERY_EXPANSION_PROMPT},
                    {"role": "user", "content": f"""Conversation history:
{history_context}

Current query: {query}

Rewritten query:"""}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            expanded = response.choices[0].message.content.strip()
            
            # Log if query was expanded
            if expanded.lower() != query.lower():
                logger.info(f"🔄 Query expanded: '{query}' → '{expanded}'")
            
            return expanded
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}, using original query")
            return query
    
    def _embed_query(self, query: str) -> List[float]:
        """Embed a query using OpenAI."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        return response.data[0].embedding
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract immigration-specific keywords from query for metadata filtering.
        These keywords help filter/boost relevant results.
        """
        import re
        
        # Immigration visa categories and terms to look for
        visa_patterns = [
            # EB categories (must match exactly to avoid EB-1 matching EB-2)
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
        query_upper = query.upper()
        
        for pattern in visa_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            keywords.extend([m.upper().replace(' ', '-') for m in matches])
        
        # Normalize keywords (e.g., "EB1" -> "EB-1")
        normalized = []
        for kw in keywords:
            # Add hyphen if missing (e.g., "EB1" -> "EB-1")
            normalized_kw = re.sub(r'(EB|H|L|O|P|E|F|J|M|I)(\d)', r'\1-\2', kw)
            normalized.append(normalized_kw)
        
        return list(set(normalized))
    
    def _build_metadata_filter(self, keywords: List[str]):
        """
        Build Qdrant filter based on extracted keywords.
        Returns None if no keywords found.
        """
        if not keywords:
            return None
        
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchText
            
            # Create OR conditions for each keyword to match in text or section_heading
            should_conditions = []
            for keyword in keywords:
                # Match in section_heading (most reliable)
                should_conditions.append(
                    FieldCondition(
                        key="section_heading",
                        match=MatchText(text=keyword)
                    )
                )
                # Match in page_title
                should_conditions.append(
                    FieldCondition(
                        key="page_title",
                        match=MatchText(text=keyword)
                    )
                )
            
            if should_conditions:
                return Filter(should=should_conditions)
            
        except Exception as e:
            logger.warning(f"Failed to build metadata filter: {e}")
        
        return None
    
    def _rerank_with_llm(self, query: str, chunks: List['RetrievedChunk'], top_n: int = 5) -> List['RetrievedChunk']:
        """
        Re-rank retrieved chunks using LLM to improve relevance.
        Uses a lightweight scoring approach.
        """
        if not chunks or len(chunks) <= top_n:
            return chunks
        
        try:
            # Build a prompt to score relevance
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
            
            # Apply rerank scores
            reranked = []
            for i, chunk in enumerate(chunks):
                score = scores.get(str(i), scores.get(i, 5))  # Default score of 5
                chunk.rerank_score = float(score)
                reranked.append(chunk)
            
            # Sort by rerank score (descending), then by original score
            reranked.sort(key=lambda x: (x.rerank_score, x.score), reverse=True)
            
            logger.info(f"🔄 Reranked {len(chunks)} chunks, top scores: {[f'{c.section_heading[:30]}({c.rerank_score})' for c in reranked[:3]]}")
            
            return reranked[:top_n]
            
        except Exception as e:
            logger.warning(f"LLM reranking failed: {e}, using original order")
            return chunks[:top_n]
    
    def _rerank_with_cross_encoder(self, query: str, chunks: List['RetrievedChunk'], top_n: int = 5) -> List['RetrievedChunk']:
        """
        Re-rank retrieved chunks using cross-encoder model.
        Faster and more accurate than LLM-based reranking.
        """
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
            
            logger.info(f"🔄 Cross-encoder reranked {len(chunks)} chunks, top scores: {[f'{c.section_heading[:30]}({c.rerank_score:.2f})' for c in reranked[:3]]}")
            
            return reranked[:top_n]
            
        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}, falling back to LLM reranking")
            return self._rerank_with_llm(query, chunks, top_n)
    
    def _retrieve(self, query: str, top_k: Optional[int] = None, use_rerank: bool = True) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks from Qdrant with metadata filtering and reranking.
        
        Args:
            query: The search query
            top_k: Number of final results to return
            use_rerank: Whether to apply LLM reranking
        """
        k = top_k or self.top_k
        query_vector = self._embed_query(query)
        
        # Extract keywords for metadata filtering
        keywords = self._extract_keywords(query)
        metadata_filter = self._build_metadata_filter(keywords) if keywords else None
        
        if keywords:
            logger.info(f"🔍 Detected keywords: {keywords}")
        
        # First, try with metadata filter if we have keywords
        results = None
        if metadata_filter:
            try:
                # Retrieve more candidates for reranking, filtered by metadata
                results = self.qdrant.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=metadata_filter,
                    limit=k * 3  # Get more for reranking
                )
                
                if results.points:
                    logger.info(f"✅ Metadata filter matched {len(results.points)} chunks")
                else:
                    logger.info("⚠️ Metadata filter returned no results, falling back to unfiltered search")
                    results = None
                    
            except Exception as e:
                logger.warning(f"Metadata filtering failed: {e}, falling back to unfiltered search")
                results = None
        
        # Fallback: search without filter
        if results is None or not results.points:
            results = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=k * 2 if use_rerank else k  # Get more candidates for reranking
            )
        
        # Convert to RetrievedChunk objects
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
        
        # Apply reranking if enabled and we have keywords (indicating specific query)
        if use_rerank and self.use_rerank and keywords and len(chunks) > k:
            chunks = self._rerank_with_cross_encoder(query, chunks, top_n=k)
        else:
            chunks = chunks[:k]
        
        return chunks
    
    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}] {chunk.section_heading}\n"
                f"URL: {chunk.url}\n"
                f"Content: {chunk.text}\n"
            )
        return "\n---\n".join(context_parts)
    
    def _generate_answer(self, query: str, context: str, original_query: str = None) -> str:
        """Generate an answer using the LLM, with conversation history for context."""
        # Build messages with conversation history
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        
        # Add conversation history if available
        if self.memory:
            history_messages = self.memory.get_messages_for_openai()
            messages.extend(history_messages)
        
        # Add current query with context
        user_message = f"""Context from USCIS knowledge base:
---
{context}
---

User Question: {original_query or query}

Please provide a comprehensive answer using ONLY the information from the context above. Include citations [1], [2], etc."""
        
        messages.append({"role": "user", "content": user_message})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        # Remove any disclaimer the LLM may have added (to avoid duplicates)
        # Then append our standard disclaimer
        if "IMPORTANT DISCLAIMER" in answer or "DISCLAIMER" in answer.upper():
            # Strip LLM-generated disclaimer variations
            for marker in ["⚠️ **IMPORTANT DISCLAIMER**", "**DISCLAIMER**", "DISCLAIMER:", "**Disclaimer**"]:
                if marker in answer:
                    answer = answer.split(marker)[0].strip()
        
        return f"{answer}\n{self.DISCLAIMER}"
    
    def _evaluate_answer(
        self, 
        query: str, 
        answer: str, 
        context: str,
        retrieval_context: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of the generated answer.
        
        Uses DeepEval if available, otherwise falls back to custom LLM evaluation.
        """
        # Use DeepEval if available
        if DEEPEVAL_AVAILABLE and self.use_deepeval:
            return self._evaluate_with_deepeval(query, answer, retrieval_context or [context])
        
        # Fallback to custom LLM-based evaluation
        return self._evaluate_with_llm(query, answer, context)
    
    def _evaluate_with_deepeval(
        self,
        query: str,
        answer: str,
        retrieval_context: List[str]
    ) -> Dict[str, Any]:
        """Evaluate using DeepEval RAG metrics."""
        try:
            # Strip the disclaimer from answer before evaluation
            # (disclaimer is added programmatically and shouldn't affect relevancy scores)
            answer_for_eval = answer.split("\n⚠️ **IMPORTANT DISCLAIMER**")[0].strip()
            
            # Create test case
            # Note: HallucinationMetric requires 'context', others use 'retrieval_context'
            test_case = LLMTestCase(
                input=query,
                actual_output=answer_for_eval,
                retrieval_context=retrieval_context,
                context=retrieval_context  # Required for HallucinationMetric
            )
            
            # Initialize metrics with thresholds
            faithfulness = FaithfulnessMetric(threshold=0.7)
            answer_relevancy = AnswerRelevancyMetric(threshold=0.7)
            contextual_relevancy = ContextualRelevancyMetric(threshold=0.7)
            hallucination = HallucinationMetric(threshold=0.5)
            
            metrics = [faithfulness, answer_relevancy, contextual_relevancy, hallucination]
            
            # Run evaluation
            results = {}
            for metric in metrics:
                try:
                    metric.measure(test_case)
                    results[metric.__class__.__name__] = {
                        "score": metric.score,
                        "passed": metric.is_successful(),
                        "reason": metric.reason if hasattr(metric, 'reason') else None
                    }
                except Exception as e:
                    logger.warning(f"Metric {metric.__class__.__name__} failed: {e}")
                    results[metric.__class__.__name__] = {"error": str(e)}
            
            # Calculate overall score
            scores = [r["score"] for r in results.values() if "score" in r and r["score"] is not None]
            overall_score = sum(scores) / len(scores) if scores else 0
            
            return {
                "evaluation_method": "deepeval",
                "faithfulness": results.get("FaithfulnessMetric", {}).get("score", 0) * 5,
                "relevance": results.get("AnswerRelevancyMetric", {}).get("score", 0) * 5,
                "contextual_relevancy": results.get("ContextualRelevancyMetric", {}).get("score", 0) * 5,
                "hallucination_score": results.get("HallucinationMetric", {}).get("score", 0),
                "overall_score": overall_score * 5,
                "detailed_results": results,
                "all_passed": all(r.get("passed", False) for r in results.values() if "passed" in r)
            }
            
        except Exception as e:
            logger.error(f"DeepEval evaluation failed: {e}")
            return {"error": str(e), "evaluation_method": "deepeval"}
    
    def _evaluate_with_llm(
        self,
        query: str,
        answer: str,
        context: str
    ) -> Dict[str, Any]:
        """Fallback: Evaluate using custom LLM-based evaluation."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.EVALUATION_PROMPT},
                    {"role": "user", "content": f"""USER QUESTION:
{query}

CONTEXT PROVIDED:
{context}

ASSISTANT'S ANSWER:
{answer}

Evaluate this response:"""}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            evaluation = json.loads(response.choices[0].message.content)
            evaluation["evaluation_method"] = "llm"
            return evaluation
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return {"error": str(e), "evaluation_method": "llm"}
    
    def chat(
        self, 
        query: str, 
        evaluate: bool = True,
        top_k: Optional[int] = None,
        use_memory: bool = True
    ) -> ChatResponse:
        """
        Process a user query and return an answer with citations.
        
        Args:
            query: The user's question
            evaluate: Whether to evaluate the answer quality
            top_k: Number of chunks to retrieve (default: self.top_k)
            use_memory: Whether to use conversation memory for context
        
        Returns:
            ChatResponse with answer, citations, sources, and optional evaluation
        """
        original_query = query
        
        # Expand query if it contains references to previous conversation
        if use_memory and self.memory:
            query = self._expand_query(query)
        
        # Retrieve relevant chunks using expanded query with optional reranking
        chunks = self._retrieve(query, top_k, use_rerank=self.use_rerank)
        
        if not chunks:
            answer = f"I couldn't find any relevant information in my knowledge base for your question.\n{self.DISCLAIMER}"
            if use_memory:
                self.memory.add_user_message(original_query)
                self.memory.add_assistant_message(answer)
            return ChatResponse(
                answer=answer,
                citations=[],
                sources=[]
            )
        
        # Build context and generate answer
        context = self._build_context(chunks)
        answer = self._generate_answer(query, context, original_query)
        
        # Extract citations
        citations = [chunk.citation for chunk in chunks]
        
        # Store in memory
        if use_memory:
            sources_summary = ", ".join([c.section_heading for c in chunks[:3]])
            self.memory.add_user_message(original_query)
            self.memory.add_assistant_message(answer, sources_summary)
        
        # Evaluate if requested
        evaluation = None
        if evaluate:
            # Extract retrieval context as list of texts for DeepEval
            retrieval_context = [chunk.text for chunk in chunks]
            evaluation = self._evaluate_answer(original_query, answer, context, retrieval_context)
        
        return ChatResponse(
            answer=answer,
            citations=citations,
            sources=chunks,
            evaluation=evaluation
        )
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
    
    def get_memory_status(self) -> str:
        """Get current memory status."""
        return f"💾 Memory: {len(self.memory)} turns stored"


def print_response(response: ChatResponse):
    """Pretty print a chat response."""
    print("\n" + "=" * 80)
    print("🤖 VISAWISE RESPONSE")
    print("=" * 80)
    print(f"\n{response.answer}\n")
    
    print("-" * 80)
    print("📚 SOURCES USED:")
    print("-" * 80)
    for i, source in enumerate(response.sources, 1):
        print(f"[{i}] {source.section_heading}")
        # Show rerank score if it was applied
        if source.rerank_score > 0:
            print(f"    Score: {source.score:.4f} | Rerank: {source.rerank_score:.0f}/10")
        else:
            print(f"    Score: {source.score:.4f}")
        print(f"    URL: {source.url}")
        if source.freshness_info:
            print(f"    📅 {source.freshness_info}")
    
    if response.evaluation:
        print("\n" + "-" * 80)
        print("📊 ANSWER EVALUATION:")
        print("-" * 80)
        eval_data = response.evaluation
        eval_method = eval_data.get('evaluation_method', 'unknown')
        print(f"  Method: {eval_method.upper()}")
        print()
        
        if "error" not in eval_data:
            if eval_method == "deepeval":
                # DeepEval metrics display
                print(f"  Faithfulness:         {eval_data.get('faithfulness', 0):.2f}/5")
                print(f"  Answer Relevancy:     {eval_data.get('relevance', 0):.2f}/5")
                print(f"  Contextual Relevancy: {eval_data.get('contextual_relevancy', 0):.2f}/5")
                print(f"  Hallucination Score:  {eval_data.get('hallucination_score', 0):.2f} (lower is better)")
                print(f"  ─────────────────────────────")
                print(f"  OVERALL SCORE:        {eval_data.get('overall_score', 0):.2f}/5")
                print(f"  All Checks Passed:    {'✅ Yes' if eval_data.get('all_passed') else '❌ No'}")
                
                # Show detailed reasons if available
                detailed = eval_data.get('detailed_results', {})
                for metric_name, result in detailed.items():
                    if result.get('reason'):
                        print(f"\n  💡 {metric_name}: {result['reason'][:200]}")
            else:
                # LLM-based evaluation display
                print(f"  Faithfulness:      {eval_data.get('faithfulness', 'N/A')}/5")
                print(f"  Relevance:         {eval_data.get('relevance', 'N/A')}/5")
                print(f"  Completeness:      {eval_data.get('completeness', 'N/A')}/5")
                print(f"  Citation Accuracy: {eval_data.get('citation_accuracy', 'N/A')}/5")
                print(f"  Clarity:           {eval_data.get('clarity', 'N/A')}/5")
                print(f"  ─────────────────────────")
                print(f"  OVERALL SCORE:     {eval_data.get('overall_score', 'N/A')}/5")
                
                if eval_data.get('issues'):
                    print(f"\n  ⚠️  Issues:")
                    for issue in eval_data['issues']:
                        print(f"      - {issue}")
                
                if eval_data.get('suggestions'):
                    print(f"\n  💡 Suggestions:")
                    for suggestion in eval_data['suggestions']:
                        print(f"      - {suggestion}")
        else:
            print(f"  Error: {eval_data['error']}")
    
    print("\n" + "=" * 80)


def interactive_mode(chatbot: VisaWiseChatbot):
    """Run the chatbot in interactive mode with conversation memory."""
    print("\n" + "=" * 80)
    print("💬 VISAWISE IMMIGRATION ASSISTANT (with Memory)")
    print("=" * 80)
    print("Ask questions about U.S. immigration. I remember our conversation!")
    print()
    print("Commands:")
    print("  /clear    - Clear conversation memory")
    print("  /history  - Show conversation history")
    print("  /memory   - Show memory status")
    print("  --no-eval - Skip answer evaluation (add to any question)")
    print("  quit      - Exit the chatbot")
    print("=" * 80 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            # Exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            if not user_input:
                continue
            
            # Memory commands
            if user_input.lower() == '/clear':
                chatbot.clear_memory()
                print("✅ Conversation memory cleared. Starting fresh!\n")
                continue
            
            if user_input.lower() == '/history':
                print("\n" + "-" * 40)
                print("📜 CONVERSATION HISTORY:")
                print("-" * 40)
                if not chatbot.memory.history:
                    print("  (empty)")
                else:
                    for i, turn in enumerate(chatbot.memory.history, 1):
                        role = "👤 You" if turn.role == "user" else "🤖 Bot"
                        content = turn.content[:100] + "..." if len(turn.content) > 100 else turn.content
                        print(f"  {i}. {role}: {content}")
                print("-" * 40 + "\n")
                continue
            
            if user_input.lower() == '/memory':
                print(f"\n{chatbot.get_memory_status()}\n")
                continue
            
            # Check for --no-eval flag
            evaluate = True
            if '--no-eval' in user_input:
                evaluate = False
                user_input = user_input.replace('--no-eval', '').strip()
            
            print("\n🔄 Searching knowledge base...")
            response = chatbot.chat(user_input, evaluate=evaluate)
            print_response(response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="VisaWise Immigration Chatbot")
    parser.add_argument(
        "query",
        nargs="*",
        help="Question to ask (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--qdrant-path",
        default="./qdrant_db",
        help="Path to Qdrant database"
    )
    parser.add_argument(
        "--collection",
        default="visawise_immigration",
        help="Qdrant collection name"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model for generation"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve"
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip answer evaluation"
    )
    parser.add_argument(
        "--memory-turns",
        type=int,
        default=10,
        help="Number of conversation turns to remember (default: 10)"
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable conversation memory"
    )
    parser.add_argument(
        "--use-deepeval",
        action="store_true",
        default=True,
        help="Use DeepEval for evaluation (enabled by default)"
    )
    parser.add_argument(
        "--no-deepeval",
        action="store_true",
        help="Disable DeepEval, use LLM-based evaluation instead"
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable LLM reranking of search results"
    )
    
    args = parser.parse_args()
    
    # Initialize chatbot
    chatbot = VisaWiseChatbot(
        qdrant_path=args.qdrant_path,
        collection_name=args.collection,
        model=args.model,
        top_k=args.top_k,
        memory_turns=args.memory_turns if not args.no_memory else 0,
        use_deepeval=not args.no_deepeval,
        use_rerank=not args.no_rerank
    )
    
    if args.query:
        # Single query mode
        query = " ".join(args.query)
        response = chatbot.chat(query, evaluate=not args.no_eval)
        print_response(response)
    else:
        # Interactive mode
        interactive_mode(chatbot)


if __name__ == "__main__":
    main()

