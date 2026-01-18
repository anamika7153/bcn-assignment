-- =============================================================================
-- Prudential RAG Chatbot - Supabase Migration
-- =============================================================================
-- This migration creates the necessary database schema for the RAG system.
--
-- How to run:
--   1. Go to your Supabase project dashboard
--   2. Navigate to SQL Editor
--   3. Copy and paste this entire file
--   4. Click "Run" or press Cmd+Enter
--
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Step 1: Enable pgvector extension
-- -----------------------------------------------------------------------------
-- pgvector enables vector similarity search for embeddings
-- This is required for semantic search functionality

create extension if not exists vector;

-- -----------------------------------------------------------------------------
-- Step 2: Create documents table
-- -----------------------------------------------------------------------------
-- This table stores document chunks with their embeddings and metadata
--
-- Columns:
--   id         : Unique identifier for each chunk
--   content    : The actual text content of the chunk
--   embedding  : 1536-dimensional vector from OpenAI text-embedding-3-small
--   metadata   : JSON object containing page number, section, source, etc.
--   created_at : Timestamp when the record was created

create table if not exists documents (
  id bigserial primary key,
  content text not null,
  embedding vector(1536),
  metadata jsonb,
  created_at timestamp with time zone default now()
);

-- Add comments for documentation
comment on table documents is 'Stores document chunks with vector embeddings for RAG retrieval';
comment on column documents.content is 'Text content of the document chunk';
comment on column documents.embedding is '1536-dim vector from OpenAI text-embedding-3-small';
comment on column documents.metadata is 'JSON metadata: page, section, source, token_count, etc.';

-- -----------------------------------------------------------------------------
-- Step 3: Create vector similarity search index
-- -----------------------------------------------------------------------------
-- IVFFlat index for fast approximate nearest neighbor search
-- lists = 100 is a good balance between speed and accuracy for ~2000 documents

create index if not exists documents_embedding_idx
  on documents using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- Additional index on metadata for filtered queries
create index if not exists documents_metadata_idx
  on documents using gin (metadata);

-- -----------------------------------------------------------------------------
-- Step 4: Create similarity search function
-- -----------------------------------------------------------------------------
-- This function performs vector similarity search with optional metadata filtering
--
-- Parameters:
--   query_embedding : The embedding vector of the search query
--   match_count     : Number of results to return (default: 5)
--   filter          : Optional JSONB filter for metadata (default: {})
--
-- Returns:
--   Table with id, content, metadata, and similarity score

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
      else documents.metadata @> filter
    end
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;

comment on function match_documents is 'Performs vector similarity search with optional metadata filtering';

-- -----------------------------------------------------------------------------
-- Step 5: Create helper functions (optional but useful)
-- -----------------------------------------------------------------------------

-- Function to get document count
create or replace function get_document_count()
returns bigint
language sql
as $$
  select count(*) from documents;
$$;

-- Function to get documents by section
create or replace function get_documents_by_section(section_name text)
returns table (
  id bigint,
  content text,
  metadata jsonb
)
language plpgsql
as $$
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata
  from documents
  where documents.metadata->>'section' = section_name
  order by documents.id;
end;
$$;

-- Function to clear all documents (useful for re-indexing)
create or replace function clear_documents()
returns void
language sql
as $$
  truncate table documents;
$$;

comment on function clear_documents is 'Removes all documents from the table. Use with caution!';

-- -----------------------------------------------------------------------------
-- Step 6: Set up Row Level Security (optional but recommended)
-- -----------------------------------------------------------------------------
-- Uncomment if you want to enable RLS for additional security

-- alter table documents enable row level security;

-- -- Allow anonymous read access
-- create policy "Allow anonymous read access"
--   on documents for select
--   to anon
--   using (true);

-- -- Allow authenticated users to insert
-- create policy "Allow authenticated insert"
--   on documents for insert
--   to authenticated
--   with check (true);

-- =============================================================================
-- Migration Complete!
-- =============================================================================
--
-- After running this migration, you can:
--   1. Run `python part_b/build_index.py` to upload documents
--   2. Run `streamlit run part_b/app.py` to start the chatbot
--
-- To verify the setup, run this query:
--   SELECT get_document_count();
--
-- =============================================================================
