import re
from typing import Dict, Optional

class BonafideExtractor:
    def __init__(self):    
        self.roll_pattern = re.compile(r'\b\d{2}[A-Z]{3}\d{4,}\b', re.IGNORECASE)
            
        self.name_pattern = re.compile(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b'
        )
            
        self.course_keywords = [
            "B.Tech Computer Science", "B.Tech Electronics", "B.Tech Mechanical",
            "B.Tech Civil", "B.Tech Electrical", "M.Tech CSE", "M.Sc Physics",
            "M.Sc Mathematics", "MBA", "B.Sc Computer Science", "BBA",
            "CSE", "ECE", "Mechanical", "Civil", "EEE"
        ]
            
        self.year_pattern = re.compile(
            r'\b(1st|2nd|3rd|4th|first|second|third|fourth|final)\s*year\b',
            re.IGNORECASE
        )
        
        self.purpose_keywords = {
            'passport': ['passport'],
            'visa': ['visa'],
            'bank loan': ['bank', 'loan', 'bank loan'],
            'education loan': ['education loan', 'educational loan'],
            'scholarship': ['scholarship'],
            'internship': ['internship'],
            'higher studies': ['higher studies', 'further studies'],
            'verification': ['verification', 'document verification'],
        }
    
    def extract_roll_number(self, text: str) -> Optional[str]:
        match = self.roll_pattern.search(text)
        return match.group(0).upper() if match else None
    
    def extract_name(self, text: str) -> Optional[str]:
    
        name_indicators = [
            r"I(?:'m| am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
            r"name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
            r"This\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
            r"Name:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
            r"Name\s*-\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
        
            r"(?:I(?:'m| am)|name is|this is)\s+([a-z]+(?:\s+[a-z]+){0,2})",
        ]
        
        for pattern in name_indicators:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1)
            
                if name.lower() in {'studying', 'currently', 'please', 'needed', 'required'}:
                    continue
            
                if name.islower():
                    name = ' '.join(word.capitalize() for word in name.split())
                return name
        
    
    
        paren_pattern = r"This\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s*\("
        match = re.search(paren_pattern, text)
        if match:
            return match.group(1)
        
    
        matches = self.name_pattern.findall(text)
        if matches:
        
            common_words = {'Sir', 'Madam', 'Dear', 'Hello', 'Hi', 'Request', 'Details', 'Good', 'Evening', 'Morning', 'Please', 'This'}
            for match in matches:
                if match not in common_words:
                    return match
        
        return None
    
    def extract_course(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        
    
        for course in self.course_keywords:
            if course.lower() in text_lower:
                return course
        
    
        abbrev_map = {
            'cse': 'B.Tech Computer Science',
            'ece': 'B.Tech Electronics',
            'mee': 'B.Tech Mechanical',
            'mechanical': 'B.Tech Mechanical',
            'cee': 'B.Tech Civil',
            'civil': 'B.Tech Civil',
            'eee': 'B.Tech Electrical',
            'electrical': 'B.Tech Electrical',
            'bce': 'B.Tech Biotechnology',
            'biotechnology': 'B.Tech Biotechnology',
            'che': 'B.Tech Chemical',
            'chemical': 'B.Tech Chemical',
        }
        
        for abbrev, full_name in abbrev_map.items():
            if abbrev in text_lower:
                return full_name
        
        return None
    
    def extract_year(self, text: str) -> Optional[str]:
        match = self.year_pattern.search(text)
        if match:
            year_text = match.group(1).lower()
        
            year_map = {
                '1st': '1st year', 'first': '1st year',
                '2nd': '2nd year', 'second': '2nd year',
                '3rd': '3rd year', 'third': '3rd year',
                '4th': '4th year', 'fourth': '4th year',
                'final': 'Final year'
            }
            return year_map.get(year_text, match.group(0))
        return None
    
    def extract_purpose(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        
    
        for purpose, keywords in self.purpose_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return purpose
        
        return None
    
    def extract_all(self, text: str) -> Dict[str, Optional[str]]:
        return {
            'name': self.extract_name(text),
            'roll_number': self.extract_roll_number(text),
            'course': self.extract_course(text),
            'year': self.extract_year(text),
            'purpose': self.extract_purpose(text)
        }
    
    def validate_extraction(self, extraction: Dict) -> tuple[bool, list]:
        required_fields = ['name', 'roll_number', 'course', 'year', 'purpose']
        missing = [field for field in required_fields if not extraction.get(field)]
        
        is_valid = len(missing) == 0
        return is_valid, missing

if __name__ == "__main__":
    extractor = BonafideExtractor()
    
    test_cases = [
        "Hi, I'm Rahul Sharma. My roll number is 21CS045. I need a bonafide certificate for passport application. I'm in 3rd year B.Tech Computer Science.",
        "This is Priya Kumar (20EC023). Please issue bonafide for bank loan. I'm studying B.Tech Electronics, currently in 2nd year.",
        "Bonafide needed for visa. Name: Amit Patel, Roll: 19ME067, Course: Mechanical, Year: final year.",
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}:")
        print(f"Input: {text}")
        print(f"\nExtracted:")
        result = extractor.extract_all(text)
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        is_valid, missing = extractor.validate_extraction(result)
        print(f"\nValid: {is_valid}")
        if missing:
            print(f"Missing: {', '.join(missing)}")