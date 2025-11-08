import re
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

try:
    from .bert_ner import BERTNERExtractor
    from .bert_classifier import BERTPurposeClassifier
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("BERT models not available, using rule-based fallback")

from .extractor import BonafideExtractor

class HybridExtractor:
    def __init__(self, use_bert: bool = True):
        self.use_bert = use_bert and BERT_AVAILABLE
        self.rule_extractor = BonafideExtractor()

        if self.use_bert:
            ner_model_path = "models/bert_ner"
            classifier_model_path = "models/bert_classifier"
            if Path(ner_model_path).exists() and Path(classifier_model_path).exists():
                print("Loading BERT models...")
                self.ner_model = BERTNERExtractor(model_path=ner_model_path)
                self.purpose_classifier = BERTPurposeClassifier(model_path=classifier_model_path)
                print("BERT models loaded")
            else:
                print("BERT models not found, using rule-based extraction")
                self.use_bert = False

        self.roll_pattern = re.compile(r"\b\d{2}[A-Z]{3}\d{4,}\b", re.IGNORECASE)

        self.course_abbrevs = {
            "BCE", "BCS", "BIT", "BEC", "BEE", "BME", 
            "BCV", "BCH", "BCB", "BAI", "BDA"
        }

    def extract_all(self, text: str) -> Dict[str, Optional[str]]:
        if self.use_bert:
            entities = self.ner_model.extract_entities(text)
            entities["purpose"] = self.purpose_classifier.predict(text)
            entities = self._enhance_with_rules(text, entities)
        else:
            entities = self.rule_extractor.extract_all(text)
        return entities

    def _enhance_with_rules(self, text: str, entities: Dict) -> Dict:
        roll_match = self.roll_pattern.search(text)
        if roll_match:
            roll_number = roll_match.group(0).upper().rstrip('.,;:')
            entities["roll_number"] = roll_number
            entities = self._infer_from_roll(entities, roll_number)
        else:
            entities["roll_number"] = self.rule_extractor.extract_roll_number(text)

        entities["name"] = self._validate_name(text, entities.get("name"))

        if entities.get("year"):
            entities["year"] = self._normalize_year(entities["year"])
        elif not entities.get("year") and entities.get("roll_number"):
            entities = self._infer_from_roll(entities, entities["roll_number"])

        if entities.get("course"):
            entities["course"] = self._extract_course_abbreviation(text, entities["course"])
        elif not entities.get("course") and entities.get("roll_number"):
            entities = self._infer_from_roll(entities, entities["roll_number"])

        if not entities.get("purpose"):
            entities["purpose"] = self.rule_extractor.extract_purpose(text)

        return entities

    def _validate_name(self, text: str, name: Optional[str]) -> Optional[str]:
        if not name:
            rule_name = self.rule_extractor.extract_name(text)
            return rule_name

        name_clean = re.sub(r"[^A-Za-z\s]", "", name).strip()
        parts = name_clean.lower().split()

        invalid_starts = {"hi", "hello", "dear", "respected", "sir", "madam"}
        invalid_words = {"sir", "madam", "please", "bonafide", "certificate", "request", "issue", "needed", "from"}

        if parts and (parts[0] in invalid_starts or any(w in invalid_words for w in parts)):
            return self.rule_extractor.extract_name(text)

        if len(parts) < 1 or len(parts) > 5:
            return self.rule_extractor.extract_name(text)

        return name_clean.title()

    def _infer_from_roll(self, entities: Dict, roll_number: str) -> Dict:
        course_code = re.search(r"\d{2}([A-Z]{3})\d+", roll_number)
        if course_code:
            abbrev = course_code.group(1).upper()
            if abbrev in self.course_abbrevs and not entities.get("course"):
                entities["course"] = abbrev

        current_year = datetime.now().year
        batch_prefix = int(roll_number[:2])
        batch_year = 2000 + batch_prefix
        year_diff = current_year - batch_year

        if year_diff <= 0:
            entities["year"] = "1st year"
        elif year_diff == 1:
            entities["year"] = "2nd year"
        elif year_diff == 2:
            entities["year"] = "3rd year"
        elif year_diff == 3:
            entities["year"] = "Final year"
        else:
            entities["year"] = "Alumni"

        return entities

    def _extract_course_abbreviation(self, text: str, course: str) -> str:  
        if course.upper() in self.course_abbrevs:
            return course.upper()
        
        for abbrev in self.course_abbrevs:
            if re.search(r'\b' + abbrev + r'\b', text, re.IGNORECASE):
                return abbrev
        
        full_to_abbrev = {
            "B.Tech Computer Science and Engineering": "BCE",
            "B.Tech Computer Science and Engineering (Data Science)": "BCS",
            "B.Tech Information Technology": "BIT",
            "B.Tech Electronics and Communication Engineering": "BEC",
            "B.Tech Electrical and Electronics Engineering": "BEE",
            "B.Tech Mechanical Engineering": "BME",
            "B.Tech Civil Engineering": "BCV",
            "B.Tech Chemical Engineering": "BCH",
            "B.Tech Biotechnology": "BCB",
            "B.Tech Artificial Intelligence and Data Science": "BAI",
            "B.Tech Data Science": "BDA"
        }
        
        return full_to_abbrev.get(course, course)

    def _normalize_year(self, year: str) -> str:
        year_lower = year.lower()
        mapping = {
            "1st": "1st year",
            "first": "1st year",
            "2nd": "2nd year",
            "second": "2nd year",
            "3rd": "3rd year",
            "third": "3rd year",
            "4th": "Final year",
            "fourth": "Final year",
            "final": "Final year",
        }
        for k, v in mapping.items():
            if k in year_lower:
                return v
        return year

    def validate_extraction(self, extraction: Dict) -> tuple[bool, list]:
        required = ["name", "roll_number", "course", "year", "purpose"]
        missing = [f for f in required if not extraction.get(f)]
        return len(missing) == 0, missing

    def get_extraction_method(self) -> str:
        return "BERT + Rules (Hybrid)" if self.use_bert else "Rule-based only"


def create_extractor(use_bert: bool = True) -> HybridExtractor:
    return HybridExtractor(use_bert=use_bert)

if __name__ == "__main__":
    extractor = HybridExtractor(use_bert=True)
    text = "Hi Sir, I'm Prabhath Avadhanam, registration number 22BCE3005, from VIT Vellore. I need a bonafide for visa."
    result = extractor.extract_all(text)
    print(result)