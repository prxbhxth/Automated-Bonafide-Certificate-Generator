# Automated Bonafide Certificate Generator

An NLP-powered system that extracts student information from natural language and generates professional bonafide certificates instantly.

## Problem Statement

Manual bonafide certificate processing is:
- Time-consuming (2-3 days per request)
- Error-prone due to manual data entry
- Requires students to fill multiple forms
- Creates administrative overhead

## Solution

**Hybrid NLP approach** combining fine-tuned BERT models with rule-based extraction:
- Students type natural language requests
- System extracts all required fields automatically
- Professional PDF certificate generated in <500ms

## Example

**Input:**
```
Hi, I'm Prabhath Avadhanam, registration number 22BCE3005. 
I need a bonafide for visa application.
```

**Extracted:**
- Name: Prabhath Avadhanam
- Roll: 22BCE3005
- Course: BCE
- Year: Final year
- Purpose: visa application

**Output:** PDF Certificate

## Architecture

1. **BERT NER Model**: Extracts name, roll number, course, year
2. **BERT Classifier**: Identifies purpose (passport, visa, loan, etc.)
3. **Rule-based Engine**: Validates and enhances extracted data
4. **PDF Generator**: Creates certificates

## Performance

| Metric | Score |
|--------|-------|
| Overall Accuracy | 94.6% |
| Processing Time | <500ms |
| Name Extraction | 92% |
| Roll Number | 99% |
| Purpose Classification | 89% |

## Installation

```bash
# Clone repository
git clone https://github.com/prxbhxth/Automated-Bonafide-Certificate-Generator.git
cd automated-bonafide-certificate-generator

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

Access at: `http://localhost:5000`

## Tech Stack

- **Backend**: Flask, Python
- **NLP**: HuggingFace Transformers, BERT (bert-base-uncased)
- **ML Framework**: PyTorch
- **PDF**: ReportLab


## API Endpoints

### Extract Entities
```http
POST /api/extract
Content-Type: application/json

{
  "text": "Student request in natural language"
}
```

### Generate PDF
```http
POST /api/generate-pdf
Content-Type: application/json

{
  "name": "Student Name",
  "roll_number": "22BCE0001",
  "course": "BCE",
  "year": "2nd year",
  "purpose": "passport application"
}
```

## Why Hybrid Approach?

**BERT Strengths:**
- Understands context and natural language
- Handles variations in phrasing
- Learns from training data

**Rule-based Strengths:**
- 99% accuracy for roll number patterns
- Domain-specific validation
- Inference capabilities (year from roll)

**Result:** Best of both worlds with 94.6% accuracy

## Training Models

```bash
# Train NER model
python src/bert_ner.py

# Train purpose classifier
python src/bert_classifier.py
```

**NER Training:**
- Model: bert-base-uncased
- Labels: 11 BIO tags
- Epochs: 7
- Learning Rate: 3e-5

**Classifier Training:**
- Model: bert-base-uncased
- Categories: 11 purposes
- Epochs: 3
- Learning Rate: 2e-5

## Acknowledgments

- HuggingFace Transformers
- BERT paper authors
- Flask framework

---