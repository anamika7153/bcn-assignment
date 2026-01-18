"""
Supabase Vector Store Module
Handles document storage and retrieval using Supabase with pgvector.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


class SupabaseVectorStore:
    """Vector store using Supabase with pgvector."""

    def __init__(self):
        """Initialize Supabase client and OpenAI client."""
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in environment")

        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.openai = OpenAI(api_key=OPENAI_API_KEY)

    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using OpenAI."""
        response = self.openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    def create_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Create embeddings for multiple texts in batches."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)

            print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")

        return all_embeddings

    def insert_documents(self, chunks: List[Dict]):
        """Insert documents with embeddings into Supabase."""
        print(f"Creating embeddings for {len(chunks)} chunks...")

        # Extract texts
        texts = [chunk['content'] for chunk in chunks]

        # Create embeddings in batches
        embeddings = self.create_embeddings_batch(texts)

        print("Inserting into Supabase...")

        # Insert in batches
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            records = []
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                records.append({
                    "content": chunk['content'],
                    "embedding": embedding,
                    "metadata": chunk['metadata']
                })

            self.supabase.table("documents").insert(records).execute()
            print(f"  Inserted {min(i + batch_size, len(chunks))}/{len(chunks)} documents")

        print("Done!")

    def search(
        self,
        query: str,
        match_count: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents using vector similarity.

        Args:
            query: Search query text
            match_count: Number of results to return
            filter_metadata: Optional metadata filter (e.g., {"section": "Risk Management"})

        Returns:
            List of matching documents with similarity scores
        """
        # Create query embedding
        query_embedding = self.create_embedding(query)

        # Call the match_documents function
        response = self.supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": match_count,
                "filter": filter_metadata or {}
            }
        ).execute()

        return response.data

    def delete_all(self):
        """Delete all documents from the store."""
        self.supabase.table("documents").delete().neq("id", 0).execute()
        print("All documents deleted.")

    def count_documents(self) -> int:
        """Count total documents in store."""
        response = self.supabase.table("documents").select("id", count="exact").execute()
        return response.count or 0


def setup_supabase_schema():
    """
    Print SQL commands needed to set up Supabase schema.
    Run these in the Supabase SQL Editor.
    """
    schema = """
-- Run this SQL in Supabase SQL Editor to set up the schema

-- 1. Enable pgvector extension
create extension if not exists vector;

-- 2. Create documents table
create table if not exists documents (
  id bigserial primary key,
  content text not null,
  embedding vector(1536),
  metadata jsonb,
  created_at timestamp with time zone default now()
);

-- 3. Create index for fast similarity search
create index if not exists documents_embedding_idx
  on documents using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- 4. Create search function
create or replace function match_documents (
  query_embedding vector(1536),
  match_count int default 5,
  filter jsonb default '{}'
)
returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where
    case
      when filter = '{}'::jsonb then true
      else metadata @> filter
    end
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- 5. Verify setup
select count(*) from documents;
"""
    print("=" * 60)
    print("SUPABASE SETUP INSTRUCTIONS")
    print("=" * 60)
    print("\n1. Go to your Supabase project dashboard")
    print("2. Navigate to SQL Editor")
    print("3. Run the following SQL:\n")
    print(schema)


def main():
    """Main function to demonstrate vector store usage."""
    print("=" * 60)
    print("Supabase Vector Store")
    print("=" * 60)

    # Check environment
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("\nSupabase credentials not found!")
        print("Please set SUPABASE_URL and SUPABASE_KEY in your .env file")
        print("\nPrinting setup instructions instead...\n")
        setup_supabase_schema()
        return

    if not OPENAI_API_KEY:
        print("\nOpenAI API key not found!")
        print("Please set OPENAI_API_KEY in your .env file")
        return

    # Initialize store
    store = SupabaseVectorStore()

    # Count existing documents
    count = store.count_documents()
    print(f"\nCurrent documents in store: {count}")

    # Example search
    if count > 0:
        print("\nExample search: 'What are Prudential's strategic priorities?'")
        results = store.search("What are Prudential's strategic priorities?", match_count=3)
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} (similarity: {result['similarity']:.3f}) ---")
            print(f"Section: {result['metadata'].get('section', 'N/A')}")
            print(f"Page: {result['metadata'].get('page', 'N/A')}")
            print(f"Content: {result['content'][:200]}...")


if __name__ == "__main__":
    main()
