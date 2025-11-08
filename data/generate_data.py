import json
import random
import argparse
from datetime import datetime
from pathlib import Path

NAMES = [
    # Common Indian names
    "Rahul Sharma", "Nisha Patel", "Amit Verma", "Sneha Reddy", "Arjun Mehta",
    "Priya Nair", "Rohit Das", "Meera Iyer", "Karan Gupta", "Divya Menon",
    "Krithika M", "Raghav Raj", "Hari Charan", "Rahul Menon", "Karthik Raj",
    "Priya Kumar", "Amit Patel", "Meenakshi Devi", "Radhika Rao",
    # Single names (edge case)
    "Nisha", "Arjun", "Priya", "Rohan", "Kavya",
    # Three-part names
    "Sai Kiran Reddy", "Lakshmi Priya Sharma", "Venkata Sai Kumar", "Anjali Devi Patel",
    "Mohammed Ali Khan", "Preethi Lakshmi Iyer", "Suresh Kumar Gupta",
    # Names with initials
    "S Ramesh", "K Vijay", "R Srinivasan", "M Deepika", "A Lakshmi",
    "T N Srinivas", "K S Raghavan", "P V Sindhu", "N R Krishna",
    # Longer names
    "Srinivasa Ramanujan Iyengar", "Venkata Naga Sai Prakash", "Bhavana Sri Lakshmi",
    # Names with various casings (edge case for name extraction)
    "arun kumar", "PRIYA SHARMA", "Divya RAO", "karthik MENON",
    # Western names (international students)
    "John Smith", "Emily Chen", "David Lee", "Sarah Johnson", "Michael Brown"
]

COURSE_MAP = {
    "BCE": "B.Tech Computer Science and Engineering",
    "BCS": "B.Tech Computer Science and Engineering (Data Science)",
    "BIT": "B.Tech Information Technology",
    "BEC": "B.Tech Electronics and Communication Engineering",
    "BEE": "B.Tech Electrical and Electronics Engineering",
    "BME": "B.Tech Mechanical Engineering",
    "BCV": "B.Tech Civil Engineering",
    "BCH": "B.Tech Chemical Engineering",
    "BCB": "B.Tech Biotechnology",
    "BAI": "B.Tech Artificial Intelligence and Data Science",
    "BDA": "B.Tech Data Science"
}

# Expanded purpose variations
PURPOSE_LABELS = [
    "passport application", "visa application", "bank account opening",
    "education loan", "scholarship", "internship", "document verification",
    "hostel admission", "company verification", "official verification", "other"
]

PURPOSE_VARIATIONS = {
    "passport application": [
        "for passport application", "for applying passport", "to get my passport",
        "for passport verification", "for passport renewal", "passport purposes",
        "for new passport", "for passport reissue", "passport related work",
        "for tatkal passport", "for passport at RPO"
    ],
    "visa application": [
        "for visa application", "for visa process", "for my visa procedure",
        "for visa verification", "for embassy visa request", "for US visa",
        "for UK visa", "for student visa", "for tourist visa", "visa purposes",
        "for Schengen visa", "for visa interview", "for consulate submission"
    ],
    "bank account opening": [
        "for bank account opening", "to open bank account", "for bank verification",
        "for salary account creation", "for student bank account", "for SBI account",
        "for ICICI account", "bank account purposes", "for opening savings account",
        "for zero balance account", "for current account"
    ],
    "education loan": [
        "for education loan", "to apply for education loan", "for study loan",
        "for higher education loan", "for educational financial assistance",
        "for bank loan", "for loan verification", "for SBI scholar loan",
        "for abroad education loan", "for master's education loan"
    ],
    "scholarship": [
        "for scholarship", "for applying scholarship", "to get scholarship",
        "for renewal of scholarship", "for merit scholarship", "for NSP scholarship",
        "for govt scholarship", "for minority scholarship", "scholarship verification",
        "for continuation of scholarship"
    ],
    "internship": [
        "for internship application", "to verify my internship", "for internship verification",
        "for internship process", "for internship requirement", "for summer internship",
        "for industrial training", "for company internship", "for 6-month internship"
    ],
    "document verification": [
        "for document verification", "for certificate verification", "for record verification",
        "for identity verification", "for document check", "for official records",
        "for academic verification", "for credential verification"
    ],
    "hostel admission": [
        "for hostel admission", "to get hostel", "for hostel allocation",
        "for hostel verification", "for accommodation approval", "for hostel seat",
        "for room allotment", "for girls hostel", "for boys hostel"
    ],
    "company verification": [
        "for company verification", "for employment verification",
        "for background check", "for HR verification", "for job application process",
        "for placement verification", "for offer letter process", "for onboarding"
    ],
    "official verification": [
        "for official verification", "for government verification",
        "for police verification", "for embassy verification", "for administrative approval",
        "for legal purposes", "for court submission", "for official records"
    ],
    "other": [
        "for personal use", "for address proof", "for general purpose",
        "for ID card reissue", "for updating college records", "for library membership",
        "for gym membership", "for phone connection", "for driving license"
    ]
}

# More varied templates with edge cases
TEMPLATES = [
    # Standard formats
    "Hi, I am {name}, {year} {course_short} ({roll}). I need a bonafide certificate {purpose_text}.",
    "Respected Sir, This is {name} from {campus}, roll {roll}, {year} {course_short}. Requesting bonafide {purpose_text}.",
    "Dear Sir/Madam, I'm {name} (Reg no {roll}), studying {year} {course_short}. Please issue a bonafide {purpose_text}.",
    
    # Informal/casual
    "Hey, need bonafide certificate. {name} here, {roll}, {course_short} {year}. Purpose: {purpose_text}.",
    "Hi sir pls issue bonafide {purpose_text}. I'm {name} {roll} from {course_short}",
    "bonafide needed urgently {purpose_text}. {name}, {year} {course_short}, roll {roll}",
    
    # Very formal
    "To Whom It May Concern, I, {name}, registration number {roll}, a student of {year} {course_short} at {campus}, respectfully request the issuance of a bonafide certificate {purpose_text}.",
    "Subject: Request for Bonafide Certificate. Respected Sir/Madam, I am {name}, currently enrolled in {year} {course_short} with registration number {roll}. I require a bonafide certificate {purpose_text}. Kindly consider my request.",
    
    # Missing punctuation (edge case)
    "hi im {name} roll no {roll} from {course_short} need bonafide {purpose_text}",
    "request for bonafide certificate {name} {roll} {year} {course_short} {purpose_text}",
    
    # Extra spacing/formatting issues (edge case)
    "Hi  ,  I'm {name}  ( {roll} ) from  {course_short}  . Need bonafide  {purpose_text}  .",
    "Hello,this is {name},roll number {roll}.I need bonafide{purpose_text}.",
    
    # Different word orders
    "{purpose_text} - need bonafide certificate. Details: {name}, {roll}, {year} {course_short}, {campus}",
    "Bonafide certificate required. Student details - Name: {name}, Registration: {roll}, Course: {course_short}, Year: {year}, Purpose: {purpose_text}",
    
    # Short/terse
    "{name} {roll} {course_short} - bonafide {purpose_text}",
    "Need bonafide. {name}, {roll}, {purpose_text}",
    
    # Very long/detailed
    "Good morning/afternoon Sir/Madam, I hope this message finds you well. My name is {name} and I am currently pursuing {year} of {course_short} at {campus}. My registration number is {roll}. I am writing to request a bonafide certificate as I need it {purpose_text}. I would be grateful if you could process this request at your earliest convenience.",
    
    # With typos (edge case)
    "Helo sir, I am {name} from {campus}. My rol number is {roll}. I ned bonafide sertificate {purpose_text}.",
    "Respected sir, this is {name}, registraion no {roll}. Plz issue bonafide {purpose_text}.",
    
    # Name at different positions
    "Roll number {roll}, {year} {course_short} student {name} requesting bonafide {purpose_text}.",
    "From {campus}, student {name} ({roll}) needs bonafide certificate {purpose_text}. Currently in {year} {course_short}.",
    
    # Multiple sentences
    "Hello. My name is {name}. I study {course_short} at {campus}. I am in {year}. My roll number is {roll}. I need a bonafide certificate. The purpose is {purpose_text}. Please issue it.",
    "I am {name}. Roll {roll}. {year} {course_short}. Need bonafide {purpose_text}. Please help.",
    
    # With greetings at different positions
    "Requesting bonafide certificate {purpose_text}. Regards, {name} ({roll}), {year} {course_short}",
    "Thank you sir. I'm {name}, {roll}, {course_short}. Need bonafide {purpose_text}. Thanks in advance.",
    
    # Questions format (edge case)
    "Can I get a bonafide certificate {purpose_text}? I'm {name}, roll no {roll}, {year} {course_short}.",
    "Is it possible to issue bonafide {purpose_text}? Details: {name}, {roll}, {course_short}, {year}",
    
    # With department mentions
    "Dear {course_short} department, I am {name} ({roll}), {year} student. Kindly issue bonafide {purpose_text}.",
    "To the administration office, This is {name} from {course_short}, roll {roll}. Request for bonafide {purpose_text}.",
    
    # Emergency/urgent tone
    "URGENT: Need bonafide certificate immediately {purpose_text}. {name}, {roll}, {year} {course_short}",
    "Sir please help, urgent requirement for bonafide {purpose_text}. I'm {name}, {roll} from {course_short}",
    
    # With dates/timelines
    "Requesting bonafide certificate by tomorrow {purpose_text}. Student: {name}, Roll: {roll}, Course: {course_short}, Year: {year}",
    "Need bonafide urgently within 2 days {purpose_text}. I am {name} ({roll}), {year} {course_short}",
    
    # With additional context
    "I have an appointment next week {purpose_text}, so I need a bonafide certificate. I'm {name}, roll number {roll}, studying {year} {course_short} at {campus}.",
    "As I need to submit documents {purpose_text}, kindly issue bonafide. Details: {name}, {roll}, {course_short}, {year}",
    
    # Mixed language style (Hinglish edge case)
    "Sir pls bonafide chahiye {purpose_text}. Mera naam {name} hai, roll {roll}, {course_short} {year}",
    "Bonafide certificate chahiye urgent {purpose_text}. I am {name}, registration {roll}, {course_short}",
    
    # With course full name instead of abbreviation
    "Hello, I'm {name} ({roll}), studying {year} {course_full}. Need bonafide {purpose_text}.",
    "Respected Sir, I am {name}, reg no {roll}, a {year} student of {course_full}. Requesting bonafide {purpose_text}.",
    
    # Minimal information (stress test)
    "{name}, {roll}, need bonafide {purpose_text}",
    "Bonafide for {name} {roll} {purpose_text}",
    
    # Overly polite/verbose
    "Most Respected Sir/Madam, I humbly request you to kindly issue a bonafide certificate for me. I am {name}, a sincere student of {year} {course_short} with registration number {roll} at {campus}. I desperately need this certificate {purpose_text}. I would be highly obliged and grateful for your kind consideration and prompt action on this matter.",
]

CAMPUSES = ["VIT Vellore", "VIT Chennai", "VIT AP", "VIT Bhopal"]
YEARS = ["1st year", "2nd year", "3rd year", "final year"]

# Edge case roll numbers
def gen_roll(batch_start=None, prefix=None, edge_case=False):
    if batch_start is None:
        batch_start = str(random.randint(18, 25))
    batch_start = batch_start[-2:]
    
    if prefix is None:
        prefix = random.choice(list(COURSE_MAP.keys()))
    
    if edge_case and random.random() < 0.1:
        # Sometimes generate unusual roll numbers
        suffix = random.randint(1, 9999)
    else:
        suffix = random.randint(100, 999)
    
    return f"{batch_start}{prefix}{suffix:04d}"

def map_year_from_batch(batch_prefix):
    current_year = datetime.now().year
    admission_year = 2000 + int(batch_prefix)
    diff = current_year - admission_year
    if diff <= 0:
        return "1st year"
    if diff == 1:
        return "2nd year"
    if diff == 2:
        return "3rd year"
    if diff == 3:
        return "final year"
    return "Alumni"

def add_noise(text):
    """Add realistic typos and variations"""
    if random.random() < 0.15:  # 15% chance of noise
        noise_type = random.choice(['spacing', 'case', 'typo', 'abbreviation'])
        
        if noise_type == 'spacing':
            # Add or remove spaces randomly
            text = text.replace(', ', ',').replace(' .', '.') if random.random() < 0.5 else text.replace(',', ', ')
        
        elif noise_type == 'case':
            # Random case changes
            if random.random() < 0.3:
                text = text.lower()
            elif random.random() < 0.3:
                text = text.upper()
        
        elif noise_type == 'typo':
            # Common typos
            typos = {
                'need': 'ned', 'please': 'pls', 'certificate': 'certificate',
                'bonafide': 'bonafied', 'roll': 'rol', 'number': 'no',
                'student': 'studnt', 'request': 'reqst'
            }
            for correct, typo in typos.items():
                if random.random() < 0.3 and correct in text.lower():
                    text = text.replace(correct, typo)
        
        elif noise_type == 'abbreviation':
            # Abbreviate common words
            abbrevs = {
                'please': 'pls', 'certificate': 'cert', 'number': 'no',
                'registration': 'reg', 'bonafide': 'bonafide'
            }
            for word, abbrev in abbrevs.items():
                if random.random() < 0.5 and word in text.lower():
                    text = text.replace(word, abbrev)
    
    return text

def make_example():
    name = random.choice(NAMES)
    campus = random.choice(CAMPUSES)
    prefix = random.choice(list(COURSE_MAP.keys()))
    batch = random.choice([str(y) for y in range(18, 26)])
    roll = gen_roll(batch_start=batch, prefix=prefix, edge_case=True)
    course_short = prefix
    course_full = COURSE_MAP[prefix]
    purpose = random.choice(PURPOSE_LABELS)
    purpose_text = random.choice(PURPOSE_VARIATIONS[purpose])
    year = map_year_from_batch(batch)

    template = random.choice(TEMPLATES)
    text = template.format(
        name=name, year=year, course_short=course_short, course_full=course_full,
        roll=roll, purpose_text=purpose_text, campus=campus
    )
    
    # Add noise to some examples
    text = add_noise(text)

    labels = {
        "name": name,
        "roll_number": roll,
        "course": course_short,
        "year": year,
        "purpose": purpose
    }
    return {"text": text, "labels": labels}

def generate(count=5000, val=1000, seed=42, out_dir="data"):
    """Generate enhanced dataset with more variation"""
    random.seed(seed)
    data = [make_example() for _ in range(count + val)]
    train = data[:count]
    val_set = data[count:count+val]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/train_requests.json", "w") as f:
        json.dump(train, f, indent=2)
    with open(f"{out_dir}/val_requests.json", "w") as f:
        json.dump(val_set, f, indent=2)

    print(f"✓ Generated {len(train)} training and {len(val_set)} validation examples")
    print(f"✓ Saved to {out_dir}/")
    print(f"\nDataset features:")
    print(f"  - {len(NAMES)} name variations (including edge cases)")
    print(f"  - {len(TEMPLATES)} diverse templates")
    print(f"  - {len(PURPOSE_LABELS)} purpose categories")
    print(f"  - Includes: typos, spacing issues, mixed cases, informal language")
    print(f"  - Edge cases: single names, long names, unusual formatting")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate enhanced training data with edge cases")
    parser.add_argument("--count", type=int, default=5000, help="Number of training examples")
    parser.add_argument("--val", type=int, default=1000, help="Number of validation examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="data", help="Output directory")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Enhanced Bonafide Data Generator")
    print("="*70 + "\n")
    
    generate(count=args.count, val=args.val, seed=args.seed, out_dir=args.out)
    
    print("\n✓ Data generation complete!")
    print("\nNext steps:")
    print("  1. Train models: python train_models.py")
    print("  2. Or customize epochs: python train_models.py --ner-epochs 3 --classifier-epochs 2")
    print("  3. Test models: python train_models.py --test")