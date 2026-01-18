"""
Model Evaluation Script
Evaluates the fine-tuned classifier and demonstrates its usage.
"""

import json
from pathlib import Path
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Configuration
MODEL_DIR = Path("text_classifier/model/final")
DATA_DIR = Path("text_classifier/data")


class TextClassifier:
    """Text classifier using fine-tuned DistilBERT."""

    def __init__(self, model_path=MODEL_DIR):
        """Load the fine-tuned model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Load label mapping from model config
        self.id_to_label = self.model.config.id2label

    def predict(self, text: str) -> dict:
        """
        Predict the category of a text.

        Returns:
            dict with predicted label, confidence, and all probabilities
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get probabilities
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

        # Helper to get label (handles both int and string keys)
        def get_label(idx):
            return self.id_to_label.get(idx) or self.id_to_label.get(str(idx))

        # All category probabilities
        all_probs = {
            get_label(i): round(p.item(), 4)
            for i, p in enumerate(probs)
        }

        return {
            'label': get_label(pred_idx),
            'confidence': round(confidence, 4),
            'probabilities': all_probs
        }

    def predict_batch(self, texts: list) -> list:
        """Predict categories for multiple texts."""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


def evaluate_on_examples():
    """Evaluate the model on example texts."""
    print("=" * 60)
    print("Text Classification Model Evaluation")
    print("=" * 60)

    # Check if model exists
    if not MODEL_DIR.exists():
        print("\nError: Model not found!")
        print(f"Expected model at: {MODEL_DIR}")
        print("\nPlease train the model first:")
        print("  python text_classifier/create_dataset.py")
        print("  python text_classifier/fine_tune.py")
        return

    # Load classifier
    print("\nLoading model...")
    classifier = TextClassifier()
    print("Model loaded!")

    # Example texts for each category
    test_examples = [
        {
            "text": "Our strategic priorities focus on delivering sustainable growth through customer-centric innovation and digital transformation across our Asian markets.",
            "expected": "strategy"
        },
        {
            "text": "The Group's adjusted operating profit increased by 8% to $3,375 million, with EEV new business profit of $2,184 million representing strong underlying performance.",
            "expected": "financial"
        },
        {
            "text": "Climate change presents both physical and transition risks to our business. We have integrated climate risk assessment into our investment decisions and underwriting processes.",
            "expected": "risk"
        },
        {
            "text": "Our commitment to sustainability includes reducing the carbon intensity of our investment portfolio by 43% compared to our 2019 baseline.",
            "expected": "esg"
        },
        {
            "text": "The insurance market in Asia continues to offer significant growth opportunities, with low penetration rates and rising middle-class populations driving demand.",
            "expected": "market"
        },
        {
            "text": "Our Singapore operation delivered strong results with APE sales growth of 6% and new business profit of $499 million.",
            "expected": "operations"
        },
        {
            "text": "As Chairman, I am pleased to report that the Board has maintained strong oversight of the Group's strategic direction and risk management framework.",
            "expected": "leadership"
        }
    ]

    print("\nEvaluating on test examples:")
    print("-" * 60)

    correct = 0
    for i, example in enumerate(test_examples, 1):
        result = classifier.predict(example['text'])

        is_correct = result['label'] == example['expected']
        correct += is_correct
        status = "✓" if is_correct else "✗"

        print(f"\n[{i}] {status}")
        print(f"    Expected: {example['expected']}")
        print(f"    Predicted: {result['label']} (confidence: {result['confidence']:.3f})")
        print(f"    Text: {example['text'][:80]}...")

    accuracy = correct / len(test_examples)
    print(f"\n{'=' * 60}")
    print(f"Accuracy on examples: {accuracy:.1%} ({correct}/{len(test_examples)})")
    print(f"{'=' * 60}")

    # Interactive demo
    print("\n\nInteractive Demo")
    print("-" * 40)
    print("Enter text to classify (or 'quit' to exit):\n")

    while True:
        try:
            user_text = input("> ").strip()
            if user_text.lower() in ['quit', 'exit', 'q']:
                break
            if not user_text:
                continue

            result = classifier.predict(user_text)
            print(f"\n  Category: {result['label']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  All probabilities:")
            for cat, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
                bar = "█" * int(prob * 20)
                print(f"    {cat:12} {prob:.3f} {bar}")
            print()

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    print("\nDone!")


def main():
    """Main evaluation function."""
    evaluate_on_examples()


if __name__ == "__main__":
    main()
