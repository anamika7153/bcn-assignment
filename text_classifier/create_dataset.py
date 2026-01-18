"""
Dataset Creation for Text Classification Fine-tuning
Creates a labeled dataset from extracted chunks for category classification.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

# Configuration
CHUNKS_FILE = Path("rag_chatbot/output/chunks.json")
OUTPUT_DIR = Path("text_classifier/data")

# Category definitions
CATEGORIES = {
    "strategy": {
        "keywords": ["strategic", "strategy", "priorities", "growth", "ambition", "goals", "vision", "mission", "competitive"],
        "description": "Strategic direction, business priorities, and long-term goals"
    },
    "financial": {
        "keywords": ["profit", "revenue", "earnings", "margin", "capital", "surplus", "operating", "eev", "nbp", "ape", "premium"],
        "description": "Financial performance, metrics, and results"
    },
    "risk": {
        "keywords": ["risk", "risks", "uncertainty", "exposure", "mitigation", "stress", "scenario", "adverse", "volatility"],
        "description": "Risk management, risk factors, and mitigation strategies"
    },
    "esg": {
        "keywords": ["climate", "sustainability", "environmental", "social", "governance", "carbon", "esg", "diversity", "inclusion"],
        "description": "Environmental, social, and governance matters"
    },
    "market": {
        "keywords": ["market", "competition", "economic", "industry", "trends", "customers", "digital", "technology", "distribution"],
        "description": "Market conditions, competition, and industry dynamics"
    },
    "operations": {
        "keywords": ["operations", "business", "segment", "asia", "africa", "hong kong", "singapore", "malaysia", "china", "region"],
        "description": "Business operations and segment performance"
    },
    "leadership": {
        "keywords": ["chairman", "ceo", "board", "directors", "executive", "leadership", "management", "team"],
        "description": "Leadership messages and governance"
    }
}


def classify_chunk(content: str, metadata: Dict) -> Tuple[str, float]:
    """
    Classify a chunk based on keyword matching and section metadata.
    Returns category and confidence score.
    """
    content_lower = content.lower()
    section = metadata.get('section', '').lower()

    # Count keyword matches for each category
    scores = {}
    for category, info in CATEGORIES.items():
        keyword_count = sum(1 for kw in info['keywords'] if kw in content_lower)
        scores[category] = keyword_count

    # Boost based on section
    if 'strategy' in section or 'strategic' in section:
        scores['strategy'] = scores.get('strategy', 0) + 3
    if 'risk' in section:
        scores['risk'] = scores.get('risk', 0) + 3
    if 'financial' in section or 'review' in section:
        scores['financial'] = scores.get('financial', 0) + 3
    if 'sustainability' in section or 'esg' in section:
        scores['esg'] = scores.get('esg', 0) + 3
    if 'governance' in section:
        scores['leadership'] = scores.get('leadership', 0) + 2
    if 'ceo' in section or 'chair' in section:
        scores['leadership'] = scores.get('leadership', 0) + 3

    # Get best category
    if not any(scores.values()):
        return 'operations', 0.3  # Default to operations

    best_category = max(scores, key=scores.get)
    max_score = max(scores.values())

    # Calculate confidence (normalized)
    confidence = min(max_score / 5, 1.0)

    return best_category, confidence


def load_chunks() -> List[Dict]:
    """Load chunks from JSON file."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

    with open(CHUNKS_FILE, 'r') as f:
        return json.load(f)


def create_labeled_dataset(chunks: List[Dict], target_size: int = 150) -> List[Dict]:
    """Create a labeled dataset from chunks."""
    labeled_data = []

    for chunk in chunks:
        category, confidence = classify_chunk(chunk['content'], chunk['metadata'])

        labeled_data.append({
            'id': chunk['id'],
            'text': chunk['content'],
            'label': category,
            'confidence': confidence,
            'page': chunk['metadata'].get('page'),
            'section': chunk['metadata'].get('section')
        })

    # Sort by confidence and select diverse samples
    labeled_data.sort(key=lambda x: -x['confidence'])

    # Ensure category balance
    final_dataset = []
    category_counts = {cat: 0 for cat in CATEGORIES}
    target_per_category = target_size // len(CATEGORIES)

    # First pass: high confidence samples
    for item in labeled_data:
        if item['confidence'] >= 0.5:
            if category_counts[item['label']] < target_per_category:
                final_dataset.append(item)
                category_counts[item['label']] += 1

    # Second pass: fill remaining with lower confidence
    for item in labeled_data:
        if item not in final_dataset:
            if category_counts[item['label']] < target_per_category:
                final_dataset.append(item)
                category_counts[item['label']] += 1

        if len(final_dataset) >= target_size:
            break

    return final_dataset


def split_dataset(data: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """Split dataset into train and validation sets."""
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def save_dataset(train_data: List[Dict], val_data: List[Dict], output_dir: Path):
    """Save datasets to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    with open(output_dir / 'train.json', 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(output_dir / 'val.json', 'w') as f:
        json.dump(val_data, f, indent=2)

    # Save as CSV for easy viewing
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)

    # Save label mapping
    label_map = {cat: i for i, cat in enumerate(CATEGORIES.keys())}
    with open(output_dir / 'label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)


def main():
    """Main function to create the labeled dataset."""
    print("=" * 60)
    print("Creating Labeled Dataset for Text Classification")
    print("=" * 60)

    # Load chunks
    print("\nLoading chunks...")
    try:
        chunks = load_chunks()
        print(f"Loaded {len(chunks)} chunks")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the text extraction and chunking scripts first:")
        print("  python part_b/extract_text.py")
        print("  python part_b/chunker.py")
        return

    # Create labeled dataset
    print("\nCreating labeled dataset...")
    labeled_data = create_labeled_dataset(chunks, target_size=150)
    print(f"Created {len(labeled_data)} labeled samples")

    # Print category distribution
    print("\nCategory distribution:")
    category_counts = {}
    for item in labeled_data:
        label = item['label']
        category_counts[label] = category_counts.get(label, 0) + 1

    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    # Split dataset
    print("\nSplitting into train/val sets (80/20)...")
    train_data, val_data = split_dataset(labeled_data)
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")

    # Save
    print(f"\nSaving to {OUTPUT_DIR}...")
    save_dataset(train_data, val_data, OUTPUT_DIR)

    # Print sample
    print("\nSample from training data:")
    sample = random.choice(train_data)
    print(f"  Label: {sample['label']}")
    print(f"  Confidence: {sample['confidence']:.2f}")
    print(f"  Text: {sample['text'][:200]}...")

    print(f"\n{'=' * 60}")
    print("Dataset creation complete!")
    print(f"{'=' * 60}")
    print(f"\nFiles created:")
    print(f"  - {OUTPUT_DIR}/train.json ({len(train_data)} samples)")
    print(f"  - {OUTPUT_DIR}/val.json ({len(val_data)} samples)")
    print(f"  - {OUTPUT_DIR}/train.csv")
    print(f"  - {OUTPUT_DIR}/val.csv")
    print(f"  - {OUTPUT_DIR}/label_map.json")


if __name__ == "__main__":
    main()
