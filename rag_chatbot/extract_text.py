"""
Text Extraction Module for RAG Pipeline
Extracts narrative text (non-table content) from Prudential 2022 Annual Report.
"""

import pdfplumber
import re
from pathlib import Path
from typing import List, Dict
import json

# Configuration
PDF_PATH = Path("prudential-plc-ar-2022.pdf")
OUTPUT_DIR = Path("rag_chatbot/output")

# Section mappings based on Table of Contents
SECTIONS = {
    "Chair's Statement": (6, 7),
    "CEO Statement": (None, None),  # Will be detected
    "Strategic Report": (10, 50),
    "Risk Management": (54, 70),
    "Governance": (71, 130),
    "Financial Statements": (131, 300),
    "Additional Information": (301, 456),
}

# Pages to skip (mostly financial tables, charts, etc.)
SKIP_PAGES = set(range(130, 300))  # Financial statements section with complex tables


def clean_text(text: str) -> str:
    """Clean extracted text by removing noise and normalizing whitespace."""
    if not text:
        return ""

    # Remove page numbers and headers/footers
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip page numbers
        if re.match(r'^\d+$', line):
            continue

        # Skip common headers/footers
        skip_patterns = [
            r'^Prudential plc Annual Report 2022',
            r'^www\.prudentialplc\.com',
            r'^\d+\s*Prudential',
            r'^Strategic report',
            r'^Governance',
            r'^Directors',
        ]
        if any(re.match(p, line, re.IGNORECASE) for p in skip_patterns):
            continue

        cleaned_lines.append(line)

    text = ' '.join(cleaned_lines)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,;:!?\'\"()\-–—$%£€&/]', '', text)

    return text.strip()


def detect_section(text: str, page_num: int) -> str:
    """Detect which section the page belongs to based on content."""
    text_lower = text.lower()

    # Check for section indicators
    if page_num <= 10:
        if 'chair' in text_lower and ('statement' in text_lower or 'letter' in text_lower):
            return "Leadership - Chair's Statement"
        if 'chief executive' in text_lower or 'ceo' in text_lower:
            return "Leadership - CEO Statement"
        if 'investment case' in text_lower:
            return "Investment Case"

    if 10 < page_num <= 35:
        if 'strategy' in text_lower:
            return "Strategic Report - Strategy"
        if 'business model' in text_lower:
            return "Strategic Report - Business Model"
        if 'key performance' in text_lower or 'kpi' in text_lower:
            return "Strategic Report - KPIs"

    if 35 < page_num <= 55:
        if 'financial review' in text_lower:
            return "Financial Review"
        if 'segment' in text_lower:
            return "Segment Performance"

    if 55 < page_num <= 75:
        if 'risk' in text_lower:
            return "Risk Management"

    if 75 < page_num <= 90:
        if 'sustainability' in text_lower or 'esg' in text_lower or 'climate' in text_lower:
            return "Sustainability & ESG"

    if page_num > 90 and page_num <= 130:
        return "Corporate Governance"

    return "General"


def extract_narrative_text(pdf_path: Path) -> List[Dict]:
    """Extract narrative text from PDF, page by page with metadata."""
    documents = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Processing {total_pages} pages...")

        for i, page in enumerate(pdf.pages):
            page_num = i + 1

            # Skip certain pages
            if page_num in SKIP_PAGES:
                continue

            # Extract text
            text = page.extract_text()
            if not text:
                continue

            # Clean text
            cleaned_text = clean_text(text)

            # Skip very short pages (likely just images/charts)
            if len(cleaned_text) < 100:
                continue

            # Detect section
            section = detect_section(cleaned_text, page_num)

            # Create document
            doc = {
                "content": cleaned_text,
                "metadata": {
                    "page": page_num,
                    "section": section,
                    "source": "Prudential AR 2022",
                    "char_count": len(cleaned_text)
                }
            }
            documents.append(doc)

            if page_num % 50 == 0:
                print(f"  Processed page {page_num}/{total_pages}")

    return documents


def save_documents(documents: List[Dict], output_path: Path):
    """Save extracted documents to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)


def main():
    """Main extraction function."""
    print("=" * 60)
    print("Narrative Text Extraction")
    print("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Extract text
    documents = extract_narrative_text(PDF_PATH)

    print(f"\nExtracted {len(documents)} pages of narrative content")

    # Print section distribution
    sections = {}
    for doc in documents:
        section = doc['metadata']['section']
        sections[section] = sections.get(section, 0) + 1

    print("\nSection distribution:")
    for section, count in sorted(sections.items(), key=lambda x: -x[1]):
        print(f"  {section}: {count} pages")

    # Calculate total characters
    total_chars = sum(doc['metadata']['char_count'] for doc in documents)
    print(f"\nTotal characters: {total_chars:,}")

    # Save to file
    output_file = OUTPUT_DIR / "extracted_documents.json"
    save_documents(documents, output_file)
    print(f"\nSaved to: {output_file}")

    return documents


if __name__ == "__main__":
    main()
