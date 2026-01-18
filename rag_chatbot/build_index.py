"""
Build Index Script
Extracts text, chunks it, and uploads to Supabase vector store.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_chatbot.extract_text import extract_narrative_text, save_documents
from rag_chatbot.chunker import chunk_documents, save_chunks
from rag_chatbot.supabase_store import SupabaseVectorStore

# Load environment variables
load_dotenv()

# Configuration
PDF_PATH = Path("prudential-plc-ar-2022.pdf")
OUTPUT_DIR = Path("rag_chatbot/output")


def main():
    """Main function to build the vector index."""
    print("=" * 60)
    print("Building Vector Index")
    print("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check environment variables
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "OPENAI_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print(f"\nError: Missing environment variables: {', '.join(missing)}")
        print("Please set them in your .env file")
        print("\nExample .env file:")
        print("  SUPABASE_URL=https://your-project.supabase.co")
        print("  SUPABASE_KEY=your-anon-key")
        print("  OPENAI_API_KEY=your-openai-key")
        return

    # Step 1: Extract text
    print("\n[Step 1/4] Extracting narrative text...")
    documents = extract_narrative_text(PDF_PATH)
    save_documents(documents, OUTPUT_DIR / "extracted_documents.json")
    print(f"  Extracted {len(documents)} pages")

    # Step 2: Chunk documents
    print("\n[Step 2/4] Chunking documents...")
    chunks = chunk_documents(documents)
    save_chunks(chunks, OUTPUT_DIR / "chunks.json")
    print(f"  Created {len(chunks)} chunks")

    # Step 3: Initialize vector store
    print("\n[Step 3/4] Connecting to Supabase...")
    try:
        store = SupabaseVectorStore()
        current_count = store.count_documents()
        print(f"  Connected! Current documents: {current_count}")
    except Exception as e:
        print(f"  Error connecting to Supabase: {e}")
        print("\n  Make sure you've run the setup SQL in Supabase SQL Editor.")
        print("  See rag_chatbot/supabase_store.py for the schema.")
        return

    # Step 4: Upload to vector store
    print("\n[Step 4/4] Uploading to vector store...")
    if current_count > 0:
        print(f"  Store already has {current_count} documents.")
        response = input("  Delete existing and re-upload? (y/N): ")
        if response.lower() == 'y':
            store.delete_all()
            store.insert_documents(chunks)
        else:
            print("  Skipping upload.")
    else:
        store.insert_documents(chunks)

    # Verify
    final_count = store.count_documents()
    print(f"\n{'=' * 60}")
    print("Index Build Complete!")
    print(f"{'=' * 60}")
    print(f"  Documents in store: {final_count}")

    # Test search
    print("\nTesting search...")
    results = store.search("What are Prudential's main strategic priorities?", match_count=2)
    if results:
        print("  Search working!")
        print(f"  Top result similarity: {results[0]['similarity']:.3f}")
    else:
        print("  Warning: Search returned no results")


if __name__ == "__main__":
    main()
