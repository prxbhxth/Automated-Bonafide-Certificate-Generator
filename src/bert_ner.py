import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from typing import List, Dict, Tuple
import json
from pathlib import Path
import re

LABEL_LIST = [
    "O",           # Outside any entity
    "B-NAME",      # Beginning of name
    "I-NAME",      # Inside name
    "B-ROLL",      # Beginning of roll number
    "I-ROLL",      # Inside roll number
    "B-COURSE",    # Beginning of course
    "I-COURSE",    # Inside course
    "B-YEAR",      # Beginning of year
    "I-YEAR",      # Inside year
    "B-PURPOSE",   # Beginning of purpose
    "I-PURPOSE",   # Inside purpose
]

label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label = {i: label for i, label in enumerate(LABEL_LIST)}

class BonafideNERDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path) as f:
            self.data = json.load(f)
        
        self.samples = []
        for item in self.data:
            tokens, labels = self._create_labels(item)
            self.samples.append((tokens, labels))
    
    def _clean_token(self, token: str) -> str:
        return token.rstrip('.,;:!?()[]{}"\'-')
    
    def _create_labels(self, item: Dict) -> Tuple[List[str], List[str]]:
        text = item['text']
        labels_dict = item['labels']
        
        words = text.split()
        labels = ['O'] * len(words)
        
        type_map = {
            'name': 'NAME',
            'roll_number': 'ROLL',
            'course': 'COURSE',
            'year': 'YEAR',
            'purpose': 'PURPOSE'
        }
        
        for entity_type, entity_value in labels_dict.items():
            if not entity_value:
                continue
            
            label_prefix = type_map.get(entity_type)
            if not label_prefix:
                continue
            
            entity_str = str(entity_value).strip()
            entity_words = entity_str.split()
            entity_len = len(entity_words)
            
            if entity_len == 0:
                continue
            
            found = False
            
            # For single-word entities
            if entity_len == 1:
                entity_clean = entity_words[0].lower()
                for i, word in enumerate(words):
                    word_clean = self._clean_token(word).lower()
                    if word_clean == entity_clean:
                        labels[i] = f'B-{label_prefix}'
                        found = True
                        break
            
            # For multi-word entities
            else:
                for i in range(len(words) - entity_len + 1):
                    text_segment = ' '.join(
                        self._clean_token(w).lower() for w in words[i:i+entity_len]
                    )
                    entity_clean = ' '.join(w.lower() for w in entity_words)
                    
                    if text_segment == entity_clean:
                        labels[i] = f'B-{label_prefix}'
                        for j in range(1, entity_len):
                            labels[i + j] = f'I-{label_prefix}'
                        found = True
                        break
        
        return words, labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        words, labels = self.samples[idx]
        
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                aligned_labels.append(label2id[labels[word_id]])
            else:
                aligned_labels.append(-100)
            previous_word_id = word_id
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(aligned_labels)
        }


class BERTNERExtractor:
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path and Path(model_path).exists():
            self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
            self.model = BertForTokenClassification.from_pretrained(model_path)
            
            # FIX: Use model's own label mappings
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.model = BertForTokenClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=len(LABEL_LIST),
                id2label=id2label,
                label2id=label2id
            )
            self.id2label = id2label
            self.label2id = label2id
        
        self.model.to(self.device)
        self.model.eval()
    
    def extract_entities(self, text: str) -> Dict[str, str]:
        tokens = text.split()
        
        if not tokens:
            return {
                'name': None,
                'roll_number': None,
                'course': None,
                'year': None,
                'purpose': None
            }
        
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        word_ids = encoding.word_ids(batch_index=0)
        encoding_tensors = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoding_tensors)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # FIX: Use model's id2label mapping
        predicted_labels = []
        for pred in predictions[0]:
            pred_id = pred.item()
            # Handle unknown label IDs gracefully
            label = self.id2label.get(pred_id, 'O')
            predicted_labels.append(label)
        
        entities = {
            'name': None,
            'roll_number': None,
            'course': None,
            'year': None,
            'purpose': None
        }
        
        current_entity = None
        current_tokens = []
        previous_word_id = None
        
        for word_id, label in zip(word_ids, predicted_labels):
            if word_id is None:
                continue
            
            # Only process first subword of each word
            if word_id == previous_word_id:
                continue
            
            previous_word_id = word_id
            
            if label.startswith('B-'):
                # Save previous entity
                if current_entity and current_tokens:
                    self._save_entity(entities, current_entity, current_tokens)
                
                # Start new entity
                current_entity = label.split('-')[1]
                current_tokens = [tokens[word_id]]
            
            elif label.startswith('I-') and current_entity:
                entity_type = label.split('-')[1]
                if entity_type == current_entity:
                    current_tokens.append(tokens[word_id])
            
            else:
                # Save current entity if exists
                if current_entity and current_tokens:
                    self._save_entity(entities, current_entity, current_tokens)
                
                current_entity = None
                current_tokens = []
        
        # Save last entity if exists
        if current_entity and current_tokens:
            self._save_entity(entities, current_entity, current_tokens)
        
        # Clean name
        if entities.get("name"):
            entities["name"] = self._clean_name(entities["name"])
        
        return entities
    
    def _save_entity(self, entities: Dict, entity_type: str, tokens: List[str]):
        entity_text = ' '.join(tokens)
        type_map = {
            'NAME': 'name',
            'ROLL': 'roll_number',
            'COURSE': 'course',
            'YEAR': 'year',
            'PURPOSE': 'purpose'
        }
        entity_key = type_map.get(entity_type)
        if entity_key:
            entities[entity_key] = entity_text
    
    def _clean_name(self, text: str) -> str:
        if not text:
            return ""
        
        parts = text.split()
        cleaned = []
        for p in parts:
            if not cleaned or p != cleaned[-1]:
                cleaned.append(p)
        return " ".join(cleaned)


def train_ner_model(
    train_data_path: str,
    val_data_path: str = None,
    output_dir: str = "models/bert_ner",
    epochs: int = 7,
    batch_size: int = 8,
    learning_rate: float = 3e-5
):
    print("="*70)
    print("Starting BERT NER Training")
    print("="*70)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForTokenClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id
    )

    print("\nLoading datasets...")
    train_dataset = BonafideNERDataset(train_data_path, tokenizer)
    val_dataset = BonafideNERDataset(val_data_path, tokenizer) if val_data_path else None
    
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch" if val_data_path else "no",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        load_best_model_at_end=True if val_data_path else False,
        metric_for_best_model="eval_loss" if val_data_path else None,
        greater_is_better=False,
        save_total_limit=3,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print(f"\nTraining for {epochs} epochs...")
    trainer.train()

    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete!")


if __name__ == "__main__":
    train_ner_model(
        train_data_path='data/train_requests.json',
        val_data_path='data/val_requests.json',
        output_dir='models/bert_ner',
        epochs=7,
        batch_size=8
    )
    
    print("\n" + "="*70)
    print("Testing NER Model")
    print("="*70)
    
    extractor = BERTNERExtractor(model_path='models/bert_ner')
    
    test_cases = [
        "Hi, I'm Rahul Sharma, roll number 22BCE0145. Need bonafide for passport.",
        "N R Krishna, 25BEE0174, need bonafide for visa",
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {text}")
        entities = extractor.extract_entities(text)
        for key, value in entities.items():
            status = "✓" if value else "❌"
            print(f"  {status} {key}: {value}")