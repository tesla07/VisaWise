"""
Embed the scraped USCIS corpus using recommended embedding models.

Supports multiple embedding providers:
- OpenAI (text-embedding-3-small, text-embedding-3-large)
- Cohere (embed-english-v3.0)
- Local (nomic-embed-text-v1.5, sentence-transformers)
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_chunks(data_dir: Path) -> List[Dict[str, Any]]:
    """Load all chunks from processed JSON files."""
    all_chunks = []
    json_files = list(data_dir.glob("*.json"))
    
    logger.info(f"Loading chunks from {len(json_files)} files...")
    
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)
    
    logger.info(f"Loaded {len(all_chunks)} chunks")
    return all_chunks


def embed_with_openai(chunks: List[Dict[str, Any]], model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Embed chunks using OpenAI API with best practices.
    
    Best practices implemented:
    - Batch processing (up to 2048 texts per request)
    - Progress tracking with time estimates
    - Error handling and retries
    - Cost tracking
    - Input validation
    
    Recommended: text-embedding-3-small (best cost/quality)
    Cost: ~$0.02 per 1M tokens
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install with: pip install openai")
    
    import time
    
    # Validate API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it with: $env:OPENAI_API_KEY='sk-...'"
        )
    
    client = OpenAI(api_key=api_key)
    
    # Extract texts
    texts = [chunk['text'] for chunk in chunks]
    
    # Validate and truncate oversized chunks
    # text-embedding-3-small has max 8192 tokens per text
    # Use ULTRA conservative limit due to variable tokenization (especially for non-English)
    # For safety: target 4000 tokens max (Pashto/Arabic can be 2 chars/token!)
    # Better to truncate more than fail on API calls
    max_tokens_per_text = 4000  # ULTRA conservative to handle all scripts safely
    max_chars_per_text = max_tokens_per_text * 2  # ~8,000 chars (assume worst case: 2 chars/token)
    
    truncated_count = 0
    validated_texts = []
    
    for i, text in enumerate(texts):
        if len(text) > max_chars_per_text:
            truncated_count += 1
            truncated_text = text[:max_chars_per_text]
            validated_texts.append(truncated_text)
            if truncated_count <= 5:  # Log first 5
                logger.warning(
                    f"Chunk {i+1} truncated: {len(text)} → {len(truncated_text)} chars "
                    f"(~{len(text)/4:.0f} → {len(truncated_text)/4:.0f} tokens estimated)"
                )
        else:
            validated_texts.append(text)
    
    if truncated_count > 0:
        logger.warning(f"⚠️  Truncated {truncated_count} oversized chunks to fit token limit (very conservative for all scripts)")
    
    texts = validated_texts
    
    # Estimate tokens and cost
    total_chars = sum(len(text) for text in texts)
    estimated_tokens = total_chars / 4  # Rough estimate: 1 token ≈ 4 chars
    estimated_cost = (estimated_tokens / 1_000_000) * 0.02  # $0.02 per 1M tokens
    
    logger.info(f"Embedding {len(texts)} chunks with OpenAI {model}")
    logger.info(f"Estimated tokens: {estimated_tokens:,.0f}")
    logger.info(f"Estimated cost: ${estimated_cost:.4f}")
    
    # Batch processing with TOKEN-BASED batching (not count-based)
    # OpenAI limit: 300,000 tokens per request
    # Use conservative limit to account for encoding overhead
    max_tokens_per_batch = 200000  # Conservative: 200K tokens per batch
    all_embeddings = []
    start_time = time.time()
    
    # Create batches dynamically based on estimated tokens
    batches = []
    current_batch = []
    current_tokens = 0
    
    for text in texts:
        text_tokens = len(text) / 4  # Estimate tokens
        
        # If adding this text would exceed limit, start new batch
        if current_tokens + text_tokens > max_tokens_per_batch and current_batch:
            batches.append(current_batch)
            current_batch = [text]
            current_tokens = text_tokens
        else:
            current_batch.append(text)
            current_tokens += text_tokens
    
    # Don't forget the last batch
    if current_batch:
        batches.append(current_batch)
    
    total_batches = len(batches)
    logger.info(f"Created {total_batches} batches (max ~250K tokens each)")
    
    # Process each batch
    for batch_num, batch in enumerate(batches, 1):
        batch_size = len(batch)
        batch_tokens = sum(len(t) / 4 for t in batch)
        
        # Progress
        elapsed = time.time() - start_time
        if batch_num > 1:
            time_per_batch = elapsed / (batch_num - 1)
            remaining_batches = total_batches - batch_num
            eta_seconds = time_per_batch * remaining_batches
            eta_str = f"{eta_seconds:.0f}s" if eta_seconds < 60 else f"{eta_seconds/60:.1f}min"
            logger.info(
                f"Processing batch {batch_num}/{total_batches} "
                f"({len(all_embeddings)}/{len(texts)} texts, "
                f"~{batch_tokens:,.0f} tokens) - ETA: {eta_str}"
            )
        else:
            logger.info(
                f"Processing batch {batch_num}/{total_batches} "
                f"({batch_size} texts, ~{batch_tokens:,.0f} tokens)"
            )
        
        # API call with retry logic
        max_retries = 3
        for retry in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch,
                    encoding_format="float"
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                break  # Success
                
            except Exception as e:
                if retry < max_retries - 1:
                    wait_time = 2 ** retry  # Exponential backoff
                    logger.warning(f"Error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {max_retries} retries: {e}")
                    raise
    
    # Convert to numpy array
    embeddings = np.array(all_embeddings, dtype=np.float32)
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info(f"✅ Embedding complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Total chunks embedded: {len(embeddings)}")
    logger.info(f"Embedding dimensions: {embeddings.shape[1]}")
    logger.info(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")
    logger.info(f"Average: {elapsed_time/len(embeddings):.3f}s per chunk")
    
    return embeddings


def embed_with_cohere(chunks: List[Dict[str, Any]]) -> np.ndarray:
    """
    Embed chunks using Cohere API.
    
    Model: embed-english-v3.0
    Cost: ~$0.17 for entire corpus
    """
    try:
        import cohere
    except ImportError:
        raise ImportError("Install with: pip install cohere")
    
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    
    texts = [chunk['text'] for chunk in chunks]
    batch_size = 96  # Cohere limit
    all_embeddings = []
    
    logger.info(f"Embedding {len(texts)} chunks with Cohere embed-english-v3.0...")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        response = co.embed(
            texts=batch,
            model='embed-english-v3.0',
            input_type='search_document'
        )
        
        all_embeddings.extend(response.embeddings)
    
    embeddings = np.array(all_embeddings)
    logger.info(f"✅ Created embeddings with shape: {embeddings.shape}")
    return embeddings


def embed_with_local(chunks: List[Dict[str, Any]], model_name: str = "nomic-ai/nomic-embed-text-v1.5") -> np.ndarray:
    """
    Embed chunks using local sentence-transformers model.
    
    Recommended: nomic-ai/nomic-embed-text-v1.5
    Cost: $0.00 (free, local)
    
    Alternatives:
    - BAAI/bge-large-en-v1.5
    - sentence-transformers/all-MiniLM-L6-v2 (fastest, lower quality)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Install with: pip install sentence-transformers")
    
    logger.info(f"Loading local model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    
    texts = [chunk['text'] for chunk in chunks]
    
    logger.info(f"Embedding {len(texts)} chunks with {model_name}...")
    
    # Use prompt_name for nomic models
    if "nomic" in model_name.lower():
        embeddings = model.encode(
            texts,
            prompt_name='search_document',
            show_progress_bar=True,
            batch_size=32
        )
    else:
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
    
    logger.info(f"✅ Created embeddings with shape: {embeddings.shape}")
    return embeddings


def save_embeddings(embeddings: np.ndarray, chunks: List[Dict[str, Any]], output_path: Path):
    """
    Save embeddings and metadata with best practices.
    
    Best practices:
    - Atomic writes (temp file + rename)
    - Float32 for efficiency
    - Rich metadata for filtering
    - Validation before save
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate
    if len(embeddings) != len(chunks):
        raise ValueError(f"Mismatch: {len(embeddings)} embeddings but {len(chunks)} chunks")
    
    # Save embeddings as numpy array (float32 for efficiency)
    embeddings_file = output_path.with_suffix('.npy')
    embeddings_float32 = embeddings.astype(np.float32)
    
    # Save directly (np.save adds .npy if needed, so we use the explicit path)
    np.save(embeddings_file, embeddings_float32)
    
    logger.info(f"✅ Saved embeddings to {embeddings_file}")
    logger.info(f"   Size: {embeddings_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Save metadata with rich information for filtering and citations
    metadata = []
    for chunk in chunks:
        chunk_metadata = chunk.get('metadata', {})
        citation_data = chunk.get('citation', {})
        
        meta = {
            'chunk_id': chunk['chunk_id'],
            'task_name': chunk['task_name'],
            'section_heading': chunk['section_heading'],
            'text': chunk['text'],
            'url': chunk_metadata.get('url', ''),
            'page_title': chunk_metadata.get('page_title', ''),
            'source_type': chunk_metadata.get('source_type', ''),
            'section': chunk_metadata.get('section', ''),
            # Timestamps for source freshness awareness
            'retrieved_at': chunk_metadata.get('retrieved_at', ''),  # When USCIS page was scraped
            'last_updated': chunk_metadata.get('last_updated', ''),  # When USCIS page was last modified
            # Citation info for user-facing references
            'citation_url': citation_data.get('full_url', chunk_metadata.get('url', '')),
            'accessed_date': citation_data.get('accessed_date', ''),  # When chunk was processed
            'formatted_citation': citation_data.get('formatted_citation', ''),
            'markdown_citation': citation_data.get('markdown_citation', ''),
        }
        metadata.append(meta)
    
    # Atomic write for metadata
    metadata_file = output_path.with_suffix('.json')
    temp_metadata = metadata_file.with_suffix('.json.tmp')
    
    with open(temp_metadata, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    temp_metadata.replace(metadata_file)
    
    logger.info(f"✅ Saved metadata to {metadata_file}")
    logger.info(f"   Size: {metadata_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"FILES CREATED:")
    logger.info(f"{'='*70}")
    logger.info(f"Embeddings: {embeddings_file}")
    logger.info(f"  - {len(embeddings)} vectors")
    logger.info(f"  - {embeddings.shape[1]} dimensions")
    logger.info(f"  - {embeddings_file.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"\nMetadata: {metadata_file}")
    logger.info(f"  - {len(metadata)} records")
    logger.info(f"  - {metadata_file.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"{'='*70}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Embed USCIS corpus")
    parser.add_argument(
        "--provider",
        choices=["openai", "cohere", "local"],
        default="openai",
        help="Embedding provider (default: openai)"
    )
    parser.add_argument(
        "--model",
        help="Specific model name (optional)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Input directory with JSON chunks"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/embeddings/embeddings"),
        help="Output path (without extension)"
    )
    
    args = parser.parse_args()
    
    # Load chunks
    chunks = load_chunks(args.data_dir)
    
    # Embed based on provider
    if args.provider == "openai":
        model = args.model or "text-embedding-3-small"
        embeddings = embed_with_openai(chunks, model=model)
    elif args.provider == "cohere":
        embeddings = embed_with_cohere(chunks)
    elif args.provider == "local":
        model = args.model or "nomic-ai/nomic-embed-text-v1.5"
        embeddings = embed_with_local(chunks, model_name=model)
    
    # Save results
    save_embeddings(embeddings, chunks, args.output)
    
    logger.info("✅ Embedding complete!")
    logger.info(f"   Chunks: {len(chunks)}")
    logger.info(f"   Dimensions: {embeddings.shape[1]}")
    logger.info(f"   Output: {args.output}")


if __name__ == "__main__":
    main()

