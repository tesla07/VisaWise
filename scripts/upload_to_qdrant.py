"""
Upload embeddings to Qdrant vector database.

This script:
1. Loads embeddings from data/embeddings/
2. Creates a Qdrant collection
3. Uploads vectors with metadata
4. Verifies the upload
"""

import json
import logging
import argparse
from pathlib import Path
import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    print("\n❌ ERROR: qdrant-client not installed")
    print("Install with: pip install qdrant-client")
    print()
    exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_embeddings(embeddings_dir: Path):
    """Load embeddings and metadata."""
    embeddings_file = embeddings_dir / "embeddings.npy"
    metadata_file = embeddings_dir / "embeddings.json"
    
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_file}")
    
    logger.info(f"Loading embeddings from {embeddings_file}")
    embeddings = np.load(embeddings_file)
    
    logger.info(f"Loading metadata from {metadata_file}")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")
    logger.info(f"Loaded {len(metadata)} metadata records")
    
    if len(embeddings) != len(metadata):
        raise ValueError(f"Mismatch: {len(embeddings)} embeddings but {len(metadata)} metadata")
    
    return embeddings, metadata


def create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """Create or recreate Qdrant collection."""
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if collection_name in collection_names:
        logger.warning(f"Collection '{collection_name}' already exists")
        
        # Check if force flag is set
        import sys
        force = '--force' in sys.argv
        
        if force:
            logger.info(f"Force flag set - deleting existing collection '{collection_name}'")
            client.delete_collection(collection_name=collection_name)
        else:
            print(f"\n⚠️  Collection '{collection_name}' already exists!")
            print("Do you want to delete and recreate it? (yes/no): ", end='')
            response = input().strip().lower()
            
            if response in ['yes', 'y']:
                logger.info(f"Deleting existing collection '{collection_name}'")
                client.delete_collection(collection_name=collection_name)
            else:
                logger.info("Keeping existing collection. Exiting.")
                return False
    
    logger.info(f"Creating collection '{collection_name}' with vector size {vector_size}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE  # Best for semantic similarity
        )
    )
    logger.info("✅ Collection created successfully")
    return True


def upload_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    embeddings: np.ndarray,
    metadata: list,
    batch_size: int = 50  # Reduced for cloud reliability
):
    """Upload embeddings and metadata to Qdrant with retry logic."""
    import time
    
    total = len(embeddings)
    logger.info(f"Uploading {total} vectors to Qdrant in batches of {batch_size}...")
    
    # Upload in batches with retry
    for i in range(0, total, batch_size):
        batch_end = min(i + batch_size, total)
        batch_embeddings = embeddings[i:batch_end]
        batch_metadata = metadata[i:batch_end]
        
        # Create points for this batch
        points = []
        for j, (embedding, meta) in enumerate(zip(batch_embeddings, batch_metadata)):
            # Build payload with citation support
            payload = {
                "chunk_id": meta["chunk_id"],
                "task_name": meta["task_name"],
                "section_heading": meta["section_heading"],
                "url": meta["url"],
                "text": meta["text"],
                "page_title": meta.get("page_title", ""),
                "source_type": meta.get("source_type", ""),
                "section": meta.get("section", ""),
            }
            
            # Add timestamp fields for source freshness awareness
            if meta.get("retrieved_at"):
                payload["retrieved_at"] = meta["retrieved_at"]
            if meta.get("last_updated"):
                payload["last_updated"] = meta["last_updated"]
            if meta.get("accessed_date"):
                payload["accessed_date"] = meta["accessed_date"]
            
            # Add citation fields if available (from agentic chunking - flattened format)
            if meta.get("citation_url"):
                payload["citation_url"] = meta["citation_url"]
                payload["citation_formatted"] = meta.get("formatted_citation", "")
                payload["citation_markdown"] = meta.get("markdown_citation", "")
            # Legacy format: citation as nested object
            elif "citation" in meta:
                citation = meta["citation"]
                payload["citation_url"] = citation.get("full_url", meta["url"])
                payload["citation_formatted"] = citation.get("formatted_citation", "")
                payload["citation_markdown"] = citation.get("markdown_citation", "")
            else:
                # Generate basic citation for legacy chunks
                payload["citation_url"] = meta["url"]
                payload["citation_formatted"] = f"{meta.get('page_title', 'USCIS')} - {meta['section_heading']}. USCIS. {meta['url']}"
                payload["citation_markdown"] = f"[{meta.get('page_title', 'USCIS')} - {meta['section_heading']}]({meta['url']})"
            
            # Add topic summary if available
            if "topic_summary" in meta:
                payload["topic_summary"] = meta["topic_summary"]
            
            point = PointStruct(
                id=i + j,
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)
        
        # Upload batch with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
                    logger.warning(f"Upload failed (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Upload failed after {max_retries} attempts")
                    raise
        
        # Progress
        logger.info(f"Uploaded {batch_end}/{total} vectors ({batch_end/total*100:.1f}%)")
    
    logger.info("✅ Upload complete!")


def verify_upload(client: QdrantClient, collection_name: str, expected_count: int):
    """Verify the upload was successful."""
    
    logger.info("Verifying upload...")
    
    # Get collection info
    collection_info = client.get_collection(collection_name=collection_name)
    actual_count = collection_info.points_count
    
    logger.info(f"Expected vectors: {expected_count}")
    logger.info(f"Actual vectors: {actual_count}")
    
    if actual_count == expected_count:
        logger.info("✅ Verification passed!")
        return True
    else:
        logger.error(f"❌ Verification failed: count mismatch")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload embeddings to Qdrant vector database"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("data/embeddings"),
        help="Directory containing embeddings.npy and embeddings.json"
    )
    parser.add_argument(
        "--collection-name",
        default="visawise_immigration",
        help="Name of the Qdrant collection"
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant server URL (default: http://localhost:6333)"
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        help="Path for embedded Qdrant (alternative to --qdrant-url)"
    )
    parser.add_argument(
        "--api-key",
        help="Qdrant API key (for cloud)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for upload (default: 100)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreate collection without prompting"
    )
    
    args = parser.parse_args()
    
    # Load embeddings
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Loading Embeddings")
    logger.info("="*70)
    embeddings, metadata = load_embeddings(args.embeddings_dir)
    vector_size = embeddings.shape[1]
    
    # Initialize Qdrant client
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Connecting to Qdrant")
    logger.info("="*70)
    
    if args.qdrant_path:
        # Embedded mode
        logger.info(f"Using embedded Qdrant at {args.qdrant_path}")
        client = QdrantClient(path=str(args.qdrant_path))
    else:
        # Server mode
        logger.info(f"Connecting to Qdrant server at {args.qdrant_url}")
        try:
            client = QdrantClient(
                url=args.qdrant_url,
                api_key=args.api_key,
                timeout=120  # Increase timeout for cloud uploads
            )
            # Test connection
            client.get_collections()
        except Exception as e:
            logger.error(f"\n❌ Failed to connect to Qdrant at {args.qdrant_url}")
            logger.error(f"Error: {e}")
            logger.error("\nMake sure Qdrant is running:")
            logger.error("  Docker: docker run -d -p 6333:6333 qdrant/qdrant")
            logger.error("  Or use embedded mode: --qdrant-path ./qdrant_db")
            return
    
    logger.info("✅ Connected to Qdrant")
    
    # Create collection
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Creating Collection")
    logger.info("="*70)
    
    if not create_qdrant_collection(client, args.collection_name, vector_size):
        return
    
    # Upload vectors
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Uploading Vectors")
    logger.info("="*70)
    
    upload_to_qdrant(
        client,
        args.collection_name,
        embeddings,
        metadata,
        batch_size=args.batch_size
    )
    
    # Verify
    logger.info("\n" + "="*70)
    logger.info("STEP 5: Verification")
    logger.info("="*70)
    
    verify_upload(client, args.collection_name, len(embeddings))
    
    # Summary
    print("\n" + "="*70)
    print("🎉 UPLOAD COMPLETE!")
    print("="*70)
    print(f"Collection: {args.collection_name}")
    print(f"Vectors uploaded: {len(embeddings):,}")
    print(f"Vector dimensions: {vector_size}")
    
    if not args.qdrant_path:
        print(f"Qdrant URL: {args.qdrant_url}")
        print(f"\n📊 Dashboard: {args.qdrant_url}/dashboard")
    
    print(f"\n🔍 Test queries:")
    print(f"  python scripts/query_qdrant.py \"How do I apply for an H-1B visa?\"")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
