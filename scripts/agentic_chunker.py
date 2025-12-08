"""
Agentic Chunking for VisaWise Immigration Data

Uses an LLM to intelligently split content into semantically coherent chunks.
More accurate than fixed-size or basic semantic chunking.

CITATION SUPPORT:
- Each chunk preserves source URL, page title, and section heading
- Adds formatted citation field for easy downstream use
- Supports anchor links for specific sections when available

PERFORMANCE:
- Parallel processing with configurable concurrency
- Resume capability (skips already processed files)
"""

import json
import os
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# You can use OpenAI or Anthropic
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Citation information for a chunk."""
    source_url: str
    page_title: str
    section_heading: str
    anchor_id: Optional[str] = None  # For direct section links
    accessed_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    
    @property
    def full_url(self) -> str:
        """URL with anchor if available."""
        if self.anchor_id:
            return f"{self.source_url}#{self.anchor_id}"
        return self.source_url
    
    @property
    def formatted_citation(self) -> str:
        """Ready-to-use citation string."""
        return f"{self.page_title} - {self.section_heading}. USCIS. {self.full_url}"
    
    @property
    def markdown_citation(self) -> str:
        """Citation with markdown link."""
        return f"[{self.page_title} - {self.section_heading}]({self.full_url})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_url": self.source_url,
            "full_url": self.full_url,
            "page_title": self.page_title,
            "section_heading": self.section_heading,
            "anchor_id": self.anchor_id,
            "accessed_date": self.accessed_date,
            "formatted_citation": self.formatted_citation,
            "markdown_citation": self.markdown_citation
        }


@dataclass
class AgenticChunk:
    """A semantically coherent chunk created by the LLM."""
    chunk_id: str
    section_heading: str
    text: str
    topic_summary: str  # LLM-generated topic description
    citation: Citation  # Citation information for this chunk
    metadata: Dict[str, Any]


CHUNKING_PROMPT = """You are an expert at organizing immigration law content for a retrieval system.

Your task is to split the following content into semantically coherent chunks. Each chunk should:
1. Cover ONE specific topic or concept completely
2. Be self-contained (understandable without other chunks)
3. Be between 200-800 words (not too short, not too long)
4. Include all related information about that topic

For immigration content, good chunk boundaries are:
- Eligibility requirements (keep all requirements together)
- Application process steps
- Required documents
- Fees and costs
- Timeline information
- Specific visa category details

Return a JSON object with a "chunks" array where each element has:
- "section_heading": A clear, descriptive title for this chunk (used for citations)
- "text": The actual content (preserve original wording EXACTLY)
- "topic_summary": A 1-sentence summary of what this chunk covers
- "anchor_id": A URL-safe slug for this section (e.g., "eligibility-requirements", "application-process")

IMPORTANT: 
- Do NOT modify the original text, only organize it
- Do NOT add information that isn't in the source
- Keep related sentences together even if it makes chunks larger
- Prefer fewer, complete chunks over many fragmented ones
- The section_heading will be used in citations, so make it clear and descriptive
- The anchor_id should be lowercase with hyphens, suitable for URL fragments

Content to chunk:
---
{content}
---

Return ONLY valid JSON object with "chunks" array, no other text."""


class AgenticChunker:
    """Uses an LLM to create semantically coherent chunks with citation support."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",  # Cost-effective, good for chunking
        max_input_chars: int = 12000,  # Stay within context limits
        api_key: Optional[str] = None
    ):
        self.model = model
        self.max_input_chars = max_input_chars
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    @staticmethod
    def _generate_anchor(heading: str) -> str:
        """Generate a URL-safe anchor ID from a heading."""
        import re
        # Convert to lowercase, replace spaces with hyphens, remove special chars
        anchor = heading.lower().strip()
        anchor = re.sub(r'[^a-z0-9\s-]', '', anchor)
        anchor = re.sub(r'\s+', '-', anchor)
        anchor = re.sub(r'-+', '-', anchor)
        return anchor.strip('-')[:50]  # Limit length
        
    def chunk_content(self, content: str, metadata: Dict[str, Any], _depth: int = 0) -> List[AgenticChunk]:
        """Split content into semantically coherent chunks using LLM."""
        
        # Prevent infinite recursion
        if _depth > 3:
            logger.warning(f"Max recursion depth reached, using fallback chunking")
            return self._fallback_chunk(content[:self.max_input_chars], metadata)
        
        # If content is too long, pre-split by sections first
        if len(content) > self.max_input_chars:
            return self._chunk_long_content(content, metadata, _depth)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise content organizer. Return only valid JSON."
                    },
                    {
                        "role": "user", 
                        "content": CHUNKING_PROMPT.format(content=content)
                    }
                ],
                temperature=0.1,  # Low temperature for consistency
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Handle both {"chunks": [...]} and direct array formats
            chunks_data = result.get("chunks", result) if isinstance(result, dict) else result
            if isinstance(chunks_data, dict):
                chunks_data = [chunks_data]
            
            chunks = []
            base_id = metadata.get("task_name", "chunk")
            source_url = metadata.get("url", "")
            page_title = metadata.get("page_title", "USCIS")
            
            for i, chunk_data in enumerate(chunks_data, start=1):
                section_heading = chunk_data.get("section_heading", "General Information")
                anchor_id = chunk_data.get("anchor_id", self._generate_anchor(section_heading))
                
                # Create citation for this chunk
                citation = Citation(
                    source_url=source_url,
                    page_title=page_title,
                    section_heading=section_heading,
                    anchor_id=anchor_id
                )
                
                chunk = AgenticChunk(
                    chunk_id=f"{base_id}-{i:03d}",
                    section_heading=section_heading,
                    text=chunk_data.get("text", ""),
                    topic_summary=chunk_data.get("topic_summary", ""),
                    citation=citation,
                    metadata={
                        **metadata,
                        "chunk_index": str(i),
                        "chunk_total": str(len(chunks_data)),
                        "chunking_method": "agentic",
                        "topic_summary": chunk_data.get("topic_summary", ""),
                        "content_section": section_heading,
                        "anchor_id": anchor_id
                    }
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"LLM chunking failed: {e}")
            # Fallback to simple chunking
            return self._fallback_chunk(content, metadata)
    
    def _chunk_long_content(self, content: str, metadata: Dict[str, Any], _depth: int = 0) -> List[AgenticChunk]:
        """Handle content that exceeds context limits by pre-splitting."""
        # If still too long after max depth, truncate and use fallback
        if _depth > 2:
            logger.warning(f"Content too long even after splitting, truncating to {self.max_input_chars} chars")
            return self._fallback_chunk(content[:self.max_input_chars], metadata)
        
        # Split by double newlines or headers first
        sections = content.split("\n\n")
        
        # If no good split points, force split at max_input_chars boundary
        if len(sections) == 1 and len(content) > self.max_input_chars:
            sections = [content[i:i+self.max_input_chars] for i in range(0, len(content), self.max_input_chars)]
        
        current_batch = ""
        all_chunks = []
        batch_num = 0
        
        for section in sections:
            # If a single section is too long, handle it separately
            if len(section) > self.max_input_chars:
                # Process current batch first
                if current_batch:
                    batch_num += 1
                    batch_metadata = {**metadata, "batch": batch_num}
                    chunks = self.chunk_content(current_batch, batch_metadata, _depth + 1)
                    all_chunks.extend(chunks)
                    current_batch = ""
                
                # Force chunk the oversized section
                batch_num += 1
                batch_metadata = {**metadata, "batch": batch_num}
                chunks = self._fallback_chunk(section[:self.max_input_chars], batch_metadata)
                all_chunks.extend(chunks)
                continue
            
            if len(current_batch) + len(section) > self.max_input_chars:
                if current_batch:
                    batch_num += 1
                    batch_metadata = {**metadata, "batch": batch_num}
                    chunks = self.chunk_content(current_batch, batch_metadata, _depth + 1)
                    all_chunks.extend(chunks)
                current_batch = section
            else:
                current_batch += "\n\n" + section if current_batch else section
        
        # Process final batch
        if current_batch:
            batch_num += 1
            batch_metadata = {**metadata, "batch": batch_num}
            chunks = self.chunk_content(current_batch, batch_metadata, _depth + 1)
            all_chunks.extend(chunks)
        
        # Re-number chunks
        for i, chunk in enumerate(all_chunks, start=1):
            chunk.chunk_id = f"{metadata.get('task_name', 'chunk')}-{i:03d}"
            chunk.metadata["chunk_index"] = str(i)
            chunk.metadata["chunk_total"] = str(len(all_chunks))
        
        return all_chunks
    
    def _fallback_chunk(self, content: str, metadata: Dict[str, Any]) -> List[AgenticChunk]:
        """Simple fallback if LLM fails."""
        section_heading = metadata.get("page_title", "Content")
        citation = Citation(
            source_url=metadata.get("url", ""),
            page_title=metadata.get("page_title", "USCIS"),
            section_heading=section_heading,
            anchor_id=self._generate_anchor(section_heading)
        )
        return [AgenticChunk(
            chunk_id=f"{metadata.get('task_name', 'chunk')}-001",
            section_heading=section_heading,
            text=content,
            topic_summary="Full page content",
            citation=citation,
            metadata={**metadata, "chunk_index": "1", "chunk_total": "1", "chunking_method": "fallback"}
        )]


def process_json_file(input_path: Path, output_path: Path, chunker: AgenticChunker) -> int:
    """Re-chunk a single JSON file using agentic chunking with citation support."""
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        return 0
    
    # Combine all existing chunks into full content
    full_text = "\n\n".join(chunk.get("text", "") for chunk in data)
    
    # Get metadata from first chunk
    base_metadata = data[0].get("metadata", {})
    base_metadata["task_name"] = data[0].get("task_name", input_path.stem)
    
    # Re-chunk with agentic approach
    new_chunks = chunker.chunk_content(full_text, base_metadata)
    
    # Convert to output format with citation support
    output_data = []
    for chunk in new_chunks:
        output_data.append({
            "task_name": base_metadata.get("task_name"),
            "chunk_id": chunk.chunk_id,
            "section_heading": chunk.section_heading,
            "text": chunk.text,
            "topic_summary": chunk.topic_summary,
            # Citation fields for downstream applications
            "citation": chunk.citation.to_dict(),
            "metadata": chunk.metadata
        })
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return len(new_chunks)


def process_file_wrapper(args_tuple):
    """Wrapper for parallel processing."""
    json_file, output_file, model = args_tuple
    chunker = AgenticChunker(model=model)
    try:
        chunk_count = process_json_file(json_file, output_file, chunker)
        return {"file": json_file.name, "chunks": chunk_count, "error": None}
    except Exception as e:
        return {"file": json_file.name, "chunks": 0, "error": str(e)}


def main():
    """Re-chunk all processed files using agentic chunking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Re-chunk immigration data using agentic chunking")
    parser.add_argument("--input-dir", default="data/processed", help="Input directory with JSON files")
    parser.add_argument("--output-dir", default="data/agentic_chunked", help="Output directory")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers (default: 5)")
    parser.add_argument("--resume", action="store_true", help="Skip already processed files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without doing it")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    json_files = list(input_dir.glob("*.json"))
    total_files = len(json_files)
    logger.info(f"Found {total_files} JSON files")
    
    # Resume: skip already processed files
    if args.resume:
        existing = set(f.name for f in output_dir.glob("*.json"))
        json_files = [f for f in json_files if f.name not in existing]
        skipped = total_files - len(json_files)
        if skipped > 0:
            logger.info(f"Resuming: skipping {skipped} already processed files")
    
    if args.limit:
        json_files = json_files[:args.limit]
        logger.info(f"Limited to {args.limit} files")
    
    if not json_files:
        logger.info("No files to process!")
        return
    
    if args.dry_run:
        logger.info(f"DRY RUN - would process {len(json_files)} files with {args.workers} workers:")
        for f in json_files[:10]:
            logger.info(f"  - {f.name}")
        if len(json_files) > 10:
            logger.info(f"  ... and {len(json_files) - 10} more")
        
        # Time estimate
        est_time = len(json_files) * 3 / args.workers  # ~3 sec per file
        logger.info(f"\n⏱️  Estimated time: {est_time/60:.1f} minutes with {args.workers} workers")
        return
    
    # Prepare work items
    work_items = [
        (json_file, output_dir / json_file.name, args.model)
        for json_file in json_files
    ]
    
    logger.info(f"Processing {len(json_files)} files with {args.workers} parallel workers...")
    start_time = time.time()
    
    total_chunks = 0
    processed = 0
    errors = 0
    
    # Parallel processing
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file_wrapper, item): item for item in work_items}
        
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            
            if result["error"]:
                logger.error(f"[{i}/{len(json_files)}] ✗ {result['file']}: {result['error']}")
                errors += 1
            else:
                logger.info(f"[{i}/{len(json_files)}] ✓ {result['file']} → {result['chunks']} chunks")
                total_chunks += result["chunks"]
                processed += 1
            
            # Progress every 10 files
            if i % 10 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (len(json_files) - i) / rate
                logger.info(f"    📊 Progress: {i}/{len(json_files)} ({i/len(json_files)*100:.1f}%) - ETA: {remaining/60:.1f} min")
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("🎉 AGENTIC CHUNKING COMPLETE")
    print("=" * 60)
    print(f"Files processed: {processed}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Errors: {errors}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Output directory: {output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

