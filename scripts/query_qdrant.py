"""
Query Qdrant vector database with natural language questions.

This script demonstrates semantic search over the immigration knowledge base.
"""

import argparse
import logging
import os
from typing import List, Dict, Any

try:
    from qdrant_client import QdrantClient
except ImportError:
    print("\n❌ ERROR: qdrant-client not installed")
    print("Install with: pip install qdrant-client")
    print()
    exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("\n❌ ERROR: openai not installed")
    print("Install with: pip install openai")
    print()
    exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def embed_query(query: str, model: str = "text-embedding-3-small") -> List[float]:
    """Embed a query using OpenAI."""
    client = OpenAI()
    
    response = client.embeddings.create(
        model=model,
        input=[query]
    )
    
    return response.data[0].embedding


def search_qdrant(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    limit: int = 5
) -> List[Any]:
    """Search Qdrant for similar vectors."""
    
    # Qdrant 1.16.x uses query_points method
    from qdrant_client.models import QueryRequest
    
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit
    )
    
    return results.points


def display_results(results: List[Any], show_text: bool = True, show_citations: bool = True):
    """Display search results in a readable format with citation support."""
    
    print("\n" + "=" * 80)
    print("🔍 SEARCH RESULTS")
    print("=" * 80)
    
    if not results:
        print("\nNo results found.")
        return
    
    for i, result in enumerate(results, 1):
        score = result.score if hasattr(result, 'score') else 0.0
        payload = result.payload if hasattr(result, 'payload') else {}
        
        print(f"\n{i}. Relevance Score: {score:.4f}")
        print(f"   📄 Section: {payload.get('section_heading', 'N/A')}")
        print(f"   📂 Source: {payload.get('section', 'N/A')}")
        print(f"   🔗 URL: {payload.get('citation_url', payload.get('url', 'N/A'))}")
        
        # Show topic summary if available (from agentic chunking)
        if payload.get('topic_summary'):
            print(f"   💡 Topic: {payload.get('topic_summary')}")
        
        if show_text:
            text = payload.get('text', '')
            # Show first 300 characters
            preview = text[:300] + "..." if len(text) > 300 else text
            print(f"\n   📝 Content Preview:")
            print(f"   {preview}")
        
        # Show citation for downstream use
        if show_citations:
            citation = payload.get('citation_formatted') or payload.get('citation_markdown')
            if citation:
                print(f"\n   📚 Citation: {citation}")
        
        print("-" * 80)
    
    # Print all citations at the end for easy copy-paste
    if show_citations and results:
        print("\n📚 ALL CITATIONS (for copy-paste):")
        print("-" * 40)
        for i, result in enumerate(results, 1):
            payload = result.payload if hasattr(result, 'payload') else {}
            citation = payload.get('citation_markdown') or payload.get('citation_formatted')
            if citation:
                print(f"[{i}] {citation}")
        print()
    
    print()


def interactive_mode(client: QdrantClient, collection_name: str, top_k: int):
    """Interactive query mode."""
    
    print("\n" + "=" * 80)
    print("💬 INTERACTIVE QUERY MODE - Immigration AI Assistant")
    print("=" * 80)
    print("Ask questions about US immigration. Type 'quit' or 'exit' to stop.")
    print("=" * 80 + "\n")
    
    while True:
        try:
            query = input("Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            if not query:
                continue
            
            # Embed query
            print("🔄 Searching...")
            query_vector = embed_query(query)
            
            # Search
            results = search_qdrant(client, collection_name, query_vector, limit=top_k)
            
            # Display
            display_results(results, show_text=True)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Query immigration knowledge base in Qdrant"
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Immigration question (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--collection-name",
        default="visawise_immigration",
        help="Name of the Qdrant collection"
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant server URL"
    )
    parser.add_argument(
        "--qdrant-path",
        help="Path for embedded Qdrant (alternative to --qdrant-url)"
    )
    parser.add_argument(
        "--api-key",
        help="Qdrant API key (for cloud)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Don't show text preview in results"
    )
    
    args = parser.parse_args()
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("\n❌ OPENAI_API_KEY environment variable not set")
        logger.error("It should be in your .env file")
        logger.error("Or set it with: $env:OPENAI_API_KEY='sk-...'\n")
        exit(1)
    
    # Connect to Qdrant
    if args.qdrant_path:
        logger.info(f"Connecting to embedded Qdrant at {args.qdrant_path}...")
        client = QdrantClient(path=args.qdrant_path)
    else:
        logger.info(f"Connecting to Qdrant at {args.qdrant_url}...")
        client = QdrantClient(
            url=args.qdrant_url,
            api_key=args.api_key
        )
    
    try:
        
        # Check collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if args.collection_name not in collection_names:
            logger.error(f"\n❌ Collection '{args.collection_name}' not found")
            logger.error(f"Available collections: {collection_names}")
            logger.error("Run 'python scripts/upload_to_qdrant.py' first\n")
            exit(1)
        
        # Get collection info
        collection_info = client.get_collection(collection_name=args.collection_name)
        logger.info(f"✅ Connected to collection: {args.collection_name}")
        logger.info(f"Vectors in database: {collection_info.points_count:,}\n")
        
    except Exception as e:
        logger.error(f"\n❌ Failed to connect to Qdrant: {e}")
        logger.error("Make sure Qdrant is running:")
        logger.error("  docker run -d -p 6333:6333 qdrant/qdrant\n")
        exit(1)
    
    # Interactive or single query mode
    if not args.query:
        # Interactive mode
        interactive_mode(client, args.collection_name, args.top_k)
    else:
        # Single query mode
        query = " ".join(args.query)
        
        print(f"\n📝 Query: {query}")
        
        # Embed query
        logger.info("Embedding query...")
        query_vector = embed_query(query)
        
        # Search
        logger.info("Searching Qdrant...")
        results = search_qdrant(client, args.collection_name, query_vector, limit=args.top_k)
        
        # Display
        display_results(results, show_text=not args.no_text)


if __name__ == "__main__":
    main()
