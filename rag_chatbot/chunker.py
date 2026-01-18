"""
Text Chunking Module for RAG Pipeline
Chunks extracted text with metadata preservation.
"""

import json
from pathlib import Path
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# Configuration
INPUT_FILE = Path("rag_chatbot/output/extracted_documents.json")
OUTPUT_FILE = Path("rag_chatbot/output/chunks.json")

# Chunking parameters
CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 150  # characters


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    Chunk documents while preserving metadata.
    Uses recursive character splitting for better semantic boundaries.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )

    chunks = []
    chunk_id = 0

    for doc in documents:
        content = doc['content']
        metadata = doc['metadata']

        # Split the document
        splits = text_splitter.split_text(content)

        for i, split in enumerate(splits):
            # Create chunk with enriched metadata
            chunk = {
                "id": f"chunk_{chunk_id}",
                "content": split,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks_in_page": len(splits),
                    "char_count": len(split),
                    "token_count": count_tokens(split)
                }
            }
            chunks.append(chunk)
            chunk_id += 1

    return chunks


def load_documents(input_path: Path) -> List[Dict]:
    """Load documents from JSON file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_chunks(chunks: List[Dict], output_path: Path):
    """Save chunks to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def main():
    """Main chunking function."""
    print("=" * 60)
    print("Text Chunking")
    print("=" * 60)

    # Load documents
    print(f"\nLoading documents from {INPUT_FILE}...")
    documents = load_documents(INPUT_FILE)
    print(f"Loaded {len(documents)} documents")

    # Chunk documents
    print(f"\nChunking with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Calculate statistics
    total_tokens = sum(c['metadata']['token_count'] for c in chunks)
    avg_tokens = total_tokens / len(chunks)

    print(f"\nChunk statistics:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average tokens per chunk: {avg_tokens:.1f}")

    # Section distribution
    sections = {}
    for chunk in chunks:
        section = chunk['metadata']['section']
        sections[section] = sections.get(section, 0) + 1

    print(f"\nChunks by section:")
    for section, count in sorted(sections.items(), key=lambda x: -x[1]):
        print(f"  {section}: {count}")

    # Save chunks
    save_chunks(chunks, OUTPUT_FILE)
    print(f"\nSaved to: {OUTPUT_FILE}")

    return chunks


if __name__ == "__main__":
    main()
