# Prudential 2022 Annual Report - GenAI Analysis

A comprehensive solution for extracting, analyzing, and querying Prudential's 2022 Annual Report using modern AI/ML techniques.

## Overview

This project implements three main components:

1. **Part A: Data Extraction & Insights (ETL)**
   - Extract financial tables from PDF
   - Clean and structure data into CSVs
   - Generate insights and visualizations
   - LLM-powered automated analysis

2. **Part B: RAG-based Q&A Chatbot**
   - Extract narrative text from report
   - Build vector embeddings with Supabase pgvector
   - Streamlit interface for interactive Q&A

3. **Part C: Fine-tuned Text Classifier**
   - Create labeled dataset from report chunks
   - Fine-tune DistilBERT for section classification
   - Enhance RAG with category-filtered retrieval

## Project Structure

```
bcn-assignment/
├── prudential-plc-ar-2022.pdf      # Source PDF
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
├── README.md                        # This file
│
├── etl_pipeline/                          # ETL & Insights
│   ├── extract_tables.py           # Table extraction
│   ├── generate_insights.py        # Insight generation
│   ├── llm_insights.py             # LLM-based insights
│   └── output/
│       ├── tables/                 # Extracted CSVs
│       └── visualizations/         # Charts
│
├── rag_chatbot/                          # RAG Chatbot
│   ├── extract_text.py             # Text extraction
│   ├── chunker.py                  # Text chunking
│   ├── supabase_store.py           # Vector store
│   ├── build_index.py              # Index builder
│   ├── qa_chain.py                 # Q&A chain
│   ├── app.py                      # Streamlit app
│   └── example_questions.md        # Q&A examples
│
└── text_classifier/                          # Fine-tuning
    ├── create_dataset.py           # Dataset creation
    ├── fine_tune.py                # Training script
    └── evaluate.py                 # Evaluation
```

## Quick Start

### 1. Environment Setup

```bash
# Clone/download the project
cd bcn-assignment

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template and add your keys
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` with your credentials:

```
OPENAI_API_KEY=your-openai-api-key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
```

### 3. Set Up Supabase

1. Create a free account at [supabase.com](https://supabase.com)
2. Create a new project
3. Go to SQL Editor and run:

```sql
-- Enable pgvector
create extension if not exists vector;

-- Create documents table
create table if not exists documents (
  id bigserial primary key,
  content text not null,
  embedding vector(1536),
  metadata jsonb,
  created_at timestamp with time zone default now()
);

-- Create index for fast search
create index if not exists documents_embedding_idx
  on documents using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- Create search function
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
```

---

## Part A: Data Extraction & Insights

### Extract Tables

```bash
python etl_pipeline/extract_tables.py
```

**Output:** 7 CSV files in `etl_pipeline/output/tables/`:
- `01_eev_new_business_profit_by_region.csv`
- `02_profit_margin_analysis.csv`
- `03_asset_management_performance.csv`
- `04_free_surplus_movement.csv`
- `05_greater_china_performance.csv`
- `06_key_performance_indicators.csv`
- `07_earnings_per_share.csv`

### Generate Insights & Visualizations

```bash
python etl_pipeline/generate_insights.py
```

**Output:** 5 visualizations in `etl_pipeline/output/visualizations/`:
- Regional performance analysis
- Profitability drivers
- Capital generation waterfall
- Greater China concentration
- Solvency position

### LLM-Based Insights (Bonus)

```bash
python etl_pipeline/llm_insights.py
```

Requires `OPENAI_API_KEY` to be set.

---

## Part B: RAG Q&A Chatbot

### Build the Index

```bash
# Step 1: Extract text and chunk
python rag_chatbot/extract_text.py
python rag_chatbot/chunker.py

# Step 2: Upload to Supabase
python rag_chatbot/build_index.py
```

### Launch the Chatbot

```bash
streamlit run rag_chatbot/app.py
```

Open http://localhost:8501 in your browser.

### Example Questions

1. What are Prudential's strategic priorities for the next 5 years?
2. How does Prudential manage climate-related risks?
3. What was the CEO's main message about company performance?
4. How did the Asia segment perform in 2022?
5. What are the key risks facing Prudential?
6. What is Prudential's approach to ESG?
7. How is Prudential positioned in the African market?
8. What are the main challenges in the current market environment?

See `rag_chatbot/example_questions.md` for detailed expected outputs.

---

## Part C: Fine-tuned Text Classifier (Optional)

This section demonstrates fine-tuning a small transformer model for text classification, which can enhance the RAG system by enabling category-filtered retrieval.

### Dataset Description

The dataset was created by extracting and labeling text chunks from the Prudential 2022 Annual Report:

| Category | Description | Train Samples | Val Samples |
|----------|-------------|---------------|-------------|
| **Strategy** | Business strategy, priorities, growth plans | ~17 | ~6 |
| **Financial** | Financial metrics, profits, revenue data | ~17 | ~2 |
| **Risk** | Risk management, mitigation strategies | ~17 | ~4 |
| **ESG** | Environmental, social, governance topics | ~17 | ~4 |
| **Market** | Market conditions, competition, trends | ~17 | ~4 |
| **Operations** | Operational performance, processes | ~17 | ~5 |
| **Leadership** | CEO/Chair statements, executive messaging | ~15 | ~5 |

**Total:** 117 training samples, 30 validation samples across 7 categories.

### Create Dataset

```bash
python text_classifier/create_dataset.py
```

This script:
1. Loads the chunked text from Part B
2. Uses keyword matching and heuristics to assign labels
3. Splits into 80% train / 20% validation
4. Outputs JSON and CSV files in `text_classifier/data/`

### Train the Model

```bash
python text_classifier/fine_tune.py
```

**Training Configuration:**
- Base Model: `distilbert-base-uncased` (66M parameters)
- Max Sequence Length: 256 tokens
- Batch Size: 8
- Epochs: 5
- Learning Rate: 2e-5
- Early Stopping: Patience of 2 epochs

Training completes in ~30-60 seconds on CPU.

### Model Performance

After training, the model achieves:

```
              precision    recall  f1-score   support

    strategy      0.500     0.500     0.500         6
   financial      0.200     1.000     0.333         2
        risk      0.800     1.000     0.889         4
         esg      0.800     1.000     0.889         4
      market      0.000     0.000     0.000         4
  operations      0.000     0.000     0.000         5
  leadership      1.000     0.600     0.750         5

    accuracy                          0.533        30
```

**Note:** The 53% accuracy is reasonable given:
- Small dataset (117 training samples for 7 classes)
- Class imbalance in validation set
- Some categories have semantic overlap

### Evaluate & Demo

```bash
python text_classifier/evaluate.py
```

Interactive demo to test the classifier on custom text.

**Example Outputs:**

| Input Text | Predicted Category | Confidence |
|------------|-------------------|------------|
| "Our strategic priority is to expand in Southeast Asia" | strategy | 0.85 |
| "The board is committed to strong governance practices" | esg | 0.72 |
| "We achieved operating profit growth of 8%" | financial | 0.91 |
| "Climate change poses significant transition risks" | risk | 0.78 |

### How Fine-Tuning Improves the System

The fine-tuned classifier enhances the RAG pipeline in several ways:

1. **Category-Filtered Retrieval**: Users can filter search results by category (e.g., "show me only risk-related content")

2. **Metadata Enrichment**: Automatically tag new documents with categories for better organization

3. **Query Classification**: Classify user questions to route to the most relevant content sections

4. **Improved Precision**: Reduce noise in search results by focusing on topically relevant chunks

**Integration Example:**
```python
from text_classifier.evaluate import load_model, predict

# Classify a user query
model, tokenizer = load_model()
query = "What are the key risks facing the company?"
category = predict(query, model, tokenizer)
# category = "risk"

# Use category to filter vector search
results = vector_store.search(query, filter={"category": category})
```

### Future Improvements

With more labeled data, the model could achieve higher accuracy:
- Manually label 500+ samples for better coverage
- Use active learning to identify uncertain predictions
- Try larger models (BERT-base, RoBERTa) for better performance
- Implement multi-label classification for chunks spanning multiple topics

---

## Key Insights Summary

### 1. Regional Performance
- Growth markets (+20% NBP) outperforming
- Hong Kong facing margin compression (-47% NBP)
- Geographic diversification progressing well

### 2. Profitability
- Insurance margin driving growth (+15%)
- Business model resilient to market volatility
- Focus on health & protection products paying off

### 3. Capital Generation
- Net free surplus up 21% to $1,374m
- Strong in-force cash generation ($2,753m)
- Supports dividends and reinvestment

### 4. Greater China Exposure
- Concentration declining (47% → 42% of NBP)
- Diversification reducing single-market risk
- Growth markets becoming more significant

### 5. Solvency Strength
- 307% cover ratio (well above requirements)
- $15.6bn surplus provides volatility buffer
- Operating profit growth demonstrates resilience

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| PDF Processing | pdfplumber, PyMuPDF |
| Data Analysis | pandas, matplotlib, seaborn |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | Supabase with pgvector |
| LLM | OpenAI GPT-4o-mini |
| Web UI | Streamlit |
| Fine-tuning | HuggingFace Transformers |
| Classification | DistilBERT |

---

## Cost Estimates

- **Embeddings**: ~$0.01 per 1M tokens
- **Q&A (GPT-4o-mini)**: ~$0.001 per query
- **Supabase**: Free tier sufficient
- **Fine-tuning**: Local/free

---

## Troubleshooting

### "OPENAI_API_KEY not found"
Make sure you've created `.env` file with your API key.

### "Supabase connection failed"
1. Check SUPABASE_URL and SUPABASE_KEY in `.env`
2. Verify the SQL schema was run in Supabase SQL Editor
3. Check that pgvector extension is enabled

### "No documents found in search"
Run `python rag_chatbot/build_index.py` to upload documents to Supabase.

### Streamlit won't start
```bash
# Try with explicit path
/Users/anamikayadav/Library/Python/3.9/bin/streamlit run rag_chatbot/app.py
```

---

## License

This project is for educational/assessment purposes only.

---

## Author

Built as part of the GenAI Engineer Case Study assignment.
