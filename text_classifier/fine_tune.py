"""
Fine-tuning Script for Text Classification
Fine-tunes DistilBERT for classifying report sections.
"""

import json
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Configuration
DATA_DIR = Path("text_classifier/data")
MODEL_DIR = Path("text_classifier/model")
MODEL_NAME = "distilbert-base-uncased"

# Training parameters
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5


class TextClassificationDataset(Dataset):
    """PyTorch Dataset for text classification."""

    def __init__(self, data, tokenizer, label_map, max_length=MAX_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = self.label_map[item['label']]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data():
    """Load training and validation data."""
    with open(DATA_DIR / 'train.json', 'r') as f:
        train_data = json.load(f)

    with open(DATA_DIR / 'val.json', 'r') as f:
        val_data = json.load(f)

    with open(DATA_DIR / 'label_map.json', 'r') as f:
        label_map = json.load(f)

    return train_data, val_data, label_map


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}


def train_model():
    """Train the text classification model."""
    print("=" * 60)
    print("Fine-tuning DistilBERT for Text Classification")
    print("=" * 60)

    # Check for data
    if not (DATA_DIR / 'train.json').exists():
        print("\nError: Training data not found!")
        print("Please run: python text_classifier/create_dataset.py first")
        return

    # Load data
    print("\nLoading data...")
    train_data, val_data, label_map = load_data()
    num_labels = len(label_map)
    id_to_label = {v: k for k, v in label_map.items()}

    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Number of labels: {num_labels}")
    print(f"  Labels: {list(label_map.keys())}")

    # Initialize tokenizer and model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_map
    )

    # Create datasets
    print("\nPreparing datasets...")
    train_dataset = TextClassificationDataset(train_data, tokenizer, label_map)
    val_dataset = TextClassificationDataset(val_data, tokenizer, label_map)

    # Training arguments
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=str(MODEL_DIR / 'logs'),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        learning_rate=LEARNING_RATE,
        report_to="none",  # Disable wandb
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train
    print("\nStarting training...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")

    trainer.train()

    # Save model
    print("\nSaving model...")
    model.save_pretrained(MODEL_DIR / 'final')
    tokenizer.save_pretrained(MODEL_DIR / 'final')

    # Final evaluation
    print("\nFinal evaluation on validation set:")
    eval_results = trainer.evaluate()
    print(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")

    # Detailed classification report
    print("\nGenerating detailed classification report...")
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    true_labels = [label_map[item['label']] for item in val_data]

    report = classification_report(
        true_labels,
        preds,
        target_names=list(label_map.keys()),
        digits=3
    )
    print("\nClassification Report:")
    print(report)

    # Save report
    with open(MODEL_DIR / 'classification_report.txt', 'w') as f:
        f.write(report)

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"{'=' * 60}")
    print(f"\nModel saved to: {MODEL_DIR / 'final'}")
    print(f"Classification report: {MODEL_DIR / 'classification_report.txt'}")


if __name__ == "__main__":
    train_model()
