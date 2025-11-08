import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from typing import List, Dict
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

PURPOSE_LABELS = [
    "passport application",
    "visa application",
    "bank account opening",
    "education loan",
    "scholarship",
    "internship",
    "document verification",
    "hostel admission",
    "company verification",
    "official verification",
    "other"
]

label2id = {label: i for i, label in enumerate(PURPOSE_LABELS)}
id2label = {i: label for i, label in enumerate(PURPOSE_LABELS)}


class PurposeDataset(Dataset):
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path) as f:
            self.data = json.load(f)
        
        self.data = [item for item in self.data if item['labels'].get('purpose')]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        purpose = item['labels']['purpose']
        
        label_id = label2id.get(purpose, label2id['other'])
        
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_id)
        }


class BERTPurposeClassifier:    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path and Path(model_path).exists():
            self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path)
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=len(PURPOSE_LABELS),
                id2label=id2label,
                label2id=label2id
            )
        
        mapping_path = Path(model_path) / "label_mapping.json" if model_path else None
        if mapping_path and mapping_path.exists():
            with open(mapping_path, "r") as f:
                label_map = json.load(f)
            self.id2label = {int(v): k for k, v in label_map.items()}
        else:
            self.id2label = {
                0: "passport application",
                1: "visa application",
                2: "bank account opening",
                3: "loan application",
                4: "scholarship",
                5: "internship verification",
                6: "document verification",
                7: "hostel requirement",
                8: "company verification",
                9: "official verification",
                10: "other"
            }

        self.model.to(self.device)
        self.model.eval()

    
    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            conf = conf.item()
            label = self.id2label[pred.item()]

        if conf < 0.6:
            return None
        return label
    
    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, predicted_idx = torch.max(probs, dim=-1)
        
        predicted_label = id2label[predicted_idx.item()]
        return predicted_label, confidence.item()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }


def train_purpose_classifier(
    train_data_path: str,
    output_dir: str = 'models/bert_classifier',
    epochs: int = 3,
    batch_size: int = 16
):
    
    print("Starting BERT purpose classifier training...")
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(PURPOSE_LABELS),
        id2label=id2label,
        label2id=label2id
    )
    
    print("Loading training data...")
    train_dataset = PurposeDataset(train_data_path, tokenizer)
    print(f"   Loaded {len(train_dataset)} samples")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        save_strategy='epoch',
        eval_strategy='no',
        learning_rate=2e-5,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Training complete!")


def evaluate_classifier(model_path: str, test_data_path: str):
    print("\nEvaluating classifier...")
    
    classifier = BERTPurposeClassifier(model_path)
    
    with open(test_data_path) as f:
        test_data = json.load(f)
    
    correct = 0
    total = 0
    
    for item in test_data:
        if not item['labels'].get('purpose'):
            continue
        
        text = item['text']
        true_purpose = item['labels']['purpose']
        
        predicted_purpose, confidence = classifier.predict_with_confidence(text)
        
        if predicted_purpose == true_purpose:
            correct += 1
        
        total += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return accuracy


if __name__ == "__main__":
    train_purpose_classifier(
        train_data_path='data/train_requests.json',
        output_dir='models/bert_classifier',
        epochs=3,
        batch_size=16
    )
    
    evaluate_classifier(
        model_path='models/bert_classifier',
        test_data_path='data/val_requests.json'
    )
    
    print("\nTesting predictions...")
    classifier = BERTPurposeClassifier(model_path='models/bert_classifier')
    
    test_texts = [
        "I need bonafide for my passport application",
        "Please issue certificate for bank account opening",
        "Required for visa application to USA",
        "Need it for scholarship application"
    ]
    
    for text in test_texts:
        purpose, confidence = classifier.predict_with_confidence(text)
        print(f"\nText: {text}")
        print(f"Purpose: {purpose} (confidence: {confidence:.2%})")