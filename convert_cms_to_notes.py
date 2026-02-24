"""
Convert CMS DE-SynPUF Data to Medical Note Format

This script converts synthetic CMS claims data into realistic medical notes
for testing the forgery detection pipeline.

Generates:
- Discharge summaries (from inpatient claims)
- Office visit notes (from carrier claims)
- Outpatient procedure notes (from outpatient claims)
- Medication lists (from prescription drug events)
"""

import csv
import random
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# Paths
BASE_DIR = Path(__file__).parent
SYNPUF_DIR = BASE_DIR / "note_data" / "cms_samples" / "synpuf"
OUTPUT_DIR = BASE_DIR / "note_data" / "cms_notes"

# ICD-9 Code Descriptions (common codes)
ICD9_DESCRIPTIONS = {
    # Cardiovascular
    "4019": "Hypertension, unspecified",
    "4280": "Congestive heart failure, unspecified",
    "41401": "Coronary atherosclerosis of native coronary artery",
    "42731": "Atrial fibrillation",
    "4139": "Other and unspecified angina pectoris",
    "4111": "Intermediate coronary syndrome",
    "4148": "Other forms of chronic ischemic heart disease",
    "4241": "Aortic valve disorders",
    "4240": "Mitral valve disorders",
    "4273": "Atrial fibrillation and flutter",

    # Diabetes
    "25000": "Diabetes mellitus type II without complication",
    "25002": "Diabetes mellitus type II, uncontrolled",
    "2720": "Pure hypercholesterolemia",
    "2724": "Other and unspecified hyperlipidemia",

    # Respiratory
    "4660": "Acute bronchitis",
    "486": "Pneumonia, organism unspecified",
    "4928": "Other emphysema",
    "49121": "Obstructive chronic bronchitis with acute exacerbation",
    "496": "Chronic airway obstruction, not elsewhere classified",

    # Musculoskeletal
    "7242": "Lumbago",
    "7244": "Thoracic or lumbosacral neuritis, unspecified",
    "7245": "Backache, unspecified",
    "71536": "Osteoarthrosis, localized, lower leg",
    "73300": "Osteoporosis, unspecified",

    # Mental Health
    "311": "Depressive disorder, not elsewhere classified",
    "30000": "Anxiety state, unspecified",

    # Renal
    "5859": "Chronic kidney disease, unspecified",
    "5849": "Acute kidney failure, unspecified",

    # Other
    "7802": "Syncope and collapse",
    "78820": "Retention of urine, unspecified",
    "4580": "Orthostatic hypotension",
    "V4501": "Cardiac pacemaker status",
    "V4502": "Automatic implantable cardiac defibrillator status",
    "V5841": "Encounter for fitting/adjustment of spectacles",
    "V5883": "Encounter for other screening for eye disorders",
    "E9330": "Fall from wheelchair",
}

# HCPCS/CPT Code Descriptions
HCPCS_DESCRIPTIONS = {
    "99213": "Office visit, established patient, low complexity",
    "99214": "Office visit, established patient, moderate complexity",
    "99215": "Office visit, established patient, high complexity",
    "99203": "Office visit, new patient, low complexity",
    "99204": "Office visit, new patient, moderate complexity",
    "99223": "Initial hospital care, high complexity",
    "99232": "Subsequent hospital care, moderate complexity",
    "99238": "Hospital discharge day management",
    "97001": "Physical therapy evaluation",
    "85610": "Prothrombin time",
    "84153": "PSA (prostate specific antigen)",
    "93000": "Electrocardiogram, routine ECG",
    "71046": "Chest X-ray, 2 views",
    "80053": "Comprehensive metabolic panel",
    "80061": "Lipid panel",
}

# Common drug names (for prescription data - NDC codes are synthetic)
DRUG_NAMES = {
    "metformin": ["Metformin 500mg", "Metformin 850mg", "Metformin 1000mg"],
    "lisinopril": ["Lisinopril 10mg", "Lisinopril 20mg", "Lisinopril 40mg"],
    "atorvastatin": ["Atorvastatin 10mg", "Atorvastatin 20mg", "Atorvastatin 40mg"],
    "amlodipine": ["Amlodipine 5mg", "Amlodipine 10mg"],
    "omeprazole": ["Omeprazole 20mg", "Omeprazole 40mg"],
    "levothyroxine": ["Levothyroxine 25mcg", "Levothyroxine 50mcg", "Levothyroxine 100mcg"],
    "metoprolol": ["Metoprolol 25mg", "Metoprolol 50mg", "Metoprolol 100mg"],
    "gabapentin": ["Gabapentin 100mg", "Gabapentin 300mg", "Gabapentin 600mg"],
    "hydrochlorothiazide": ["Hydrochlorothiazide 12.5mg", "Hydrochlorothiazide 25mg"],
    "furosemide": ["Furosemide 20mg", "Furosemide 40mg"],
}

# Provider name generator
FIRST_NAMES = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
               "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Sarah"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas", "Moore", "Jackson", "Lee"]

@dataclass
class Beneficiary:
    """Patient demographic information."""
    id: str
    birth_date: str
    death_date: Optional[str]
    sex: str  # 1=Male, 2=Female
    race: str
    state_code: str
    chronic_conditions: Dict[str, bool]


@dataclass
class InpatientClaim:
    """Inpatient hospital claim."""
    patient_id: str
    claim_id: str
    admission_date: str
    discharge_date: str
    provider_num: str
    attending_npi: str
    admitting_diagnosis: str
    diagnoses: List[str]
    procedures: List[str]
    drg_code: str
    payment_amount: float


@dataclass
class CarrierClaim:
    """Carrier (physician/supplier) claim."""
    patient_id: str
    claim_id: str
    service_date: str
    provider_npi: str
    diagnoses: List[str]
    hcpcs_codes: List[str]
    payment_amounts: List[float]


def generate_provider_name() -> str:
    """Generate a random provider name."""
    return f"Dr. {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}, MD"


def generate_npi() -> str:
    """Generate a valid-looking NPI (10 digits starting with 1 or 2)."""
    return f"{random.choice(['1', '2'])}{random.randint(100000000, 999999999)}"


def format_date(date_str: str) -> str:
    """Convert YYYYMMDD to MM/DD/YYYY."""
    if not date_str or len(date_str) != 8:
        return "Unknown"
    try:
        return f"{date_str[4:6]}/{date_str[6:8]}/{date_str[0:4]}"
    except:
        return "Unknown"


def get_icd9_description(code: str) -> str:
    """Get description for ICD-9 code."""
    return ICD9_DESCRIPTIONS.get(code, f"Diagnosis code {code}")


def get_hcpcs_description(code: str) -> str:
    """Get description for HCPCS/CPT code."""
    return HCPCS_DESCRIPTIONS.get(code, f"Procedure {code}")


def calculate_age(birth_date: str, reference_date: str) -> int:
    """Calculate age from birth date to reference date."""
    try:
        birth = datetime.strptime(birth_date, "%Y%m%d")
        ref = datetime.strptime(reference_date, "%Y%m%d")
        age = (ref - birth).days // 365
        return max(0, age)
    except:
        return 65  # Default age


def get_sex_string(sex_code: str) -> str:
    """Convert sex code to string."""
    return "male" if sex_code == "1" else "female"


def load_beneficiaries(limit: int = 1000) -> Dict[str, Beneficiary]:
    """Load beneficiary data from CSV."""
    beneficiaries = {}
    filepath = SYNPUF_DIR / "DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv"

    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return beneficiaries

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break

            chronic = {
                "alzheimers": row.get("SP_ALZHDMTA", "2") == "1",
                "chf": row.get("SP_CHF", "2") == "1",
                "ckd": row.get("SP_CHRNKIDN", "2") == "1",
                "cancer": row.get("SP_CNCR", "2") == "1",
                "copd": row.get("SP_COPD", "2") == "1",
                "depression": row.get("SP_DEPRESSN", "2") == "1",
                "diabetes": row.get("SP_DIABETES", "2") == "1",
                "ihd": row.get("SP_ISCHMCHT", "2") == "1",  # Ischemic heart disease
                "osteoporosis": row.get("SP_OSTEOPRS", "2") == "1",
                "arthritis": row.get("SP_RA_OA", "2") == "1",
                "stroke": row.get("SP_STRKETIA", "2") == "1",
            }

            beneficiaries[row["DESYNPUF_ID"]] = Beneficiary(
                id=row["DESYNPUF_ID"],
                birth_date=row.get("BENE_BIRTH_DT", ""),
                death_date=row.get("BENE_DEATH_DT", "") or None,
                sex=row.get("BENE_SEX_IDENT_CD", "1"),
                race=row.get("BENE_RACE_CD", "1"),
                state_code=row.get("SP_STATE_CODE", ""),
                chronic_conditions=chronic
            )

    print(f"Loaded {len(beneficiaries)} beneficiaries")
    return beneficiaries


def load_inpatient_claims(limit: int = 500) -> List[InpatientClaim]:
    """Load inpatient claims from CSV."""
    claims = []
    filepath = SYNPUF_DIR / "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv"

    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return claims

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break

            # Collect diagnoses
            diagnoses = []
            for j in range(1, 11):
                dx = row.get(f"ICD9_DGNS_CD_{j}", "")
                if dx:
                    diagnoses.append(dx)

            # Collect procedures
            procedures = []
            for j in range(1, 7):
                proc = row.get(f"ICD9_PRCDR_CD_{j}", "")
                if proc:
                    procedures.append(proc)

            claims.append(InpatientClaim(
                patient_id=row["DESYNPUF_ID"],
                claim_id=row["CLM_ID"],
                admission_date=row.get("CLM_ADMSN_DT", ""),
                discharge_date=row.get("NCH_BENE_DSCHRG_DT", ""),
                provider_num=row.get("PRVDR_NUM", ""),
                attending_npi=row.get("AT_PHYSN_NPI", ""),
                admitting_diagnosis=row.get("ADMTNG_ICD9_DGNS_CD", ""),
                diagnoses=diagnoses,
                procedures=procedures,
                drg_code=row.get("CLM_DRG_CD", ""),
                payment_amount=float(row.get("CLM_PMT_AMT", "0") or "0")
            ))

    print(f"Loaded {len(claims)} inpatient claims")
    return claims


def load_carrier_claims(limit: int = 1000) -> List[CarrierClaim]:
    """Load carrier claims from CSV."""
    claims = []
    filepath = SYNPUF_DIR / "DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv"

    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return claims

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break

            # Collect diagnoses
            diagnoses = []
            for j in range(1, 9):
                dx = row.get(f"ICD9_DGNS_CD_{j}", "")
                if dx:
                    diagnoses.append(dx)

            # Collect HCPCS codes and payments
            hcpcs = []
            payments = []
            for j in range(1, 14):
                code = row.get(f"HCPCS_CD_{j}", "")
                if code:
                    hcpcs.append(code)
                    amt = row.get(f"LINE_NCH_PMT_AMT_{j}", "0")
                    payments.append(float(amt or "0"))

            if diagnoses:  # Only include claims with diagnoses
                claims.append(CarrierClaim(
                    patient_id=row["DESYNPUF_ID"],
                    claim_id=row["CLM_ID"],
                    service_date=row.get("CLM_FROM_DT", ""),
                    provider_npi=row.get("PRF_PHYSN_NPI_1", ""),
                    diagnoses=diagnoses,
                    hcpcs_codes=hcpcs,
                    payment_amounts=payments
                ))

    print(f"Loaded {len(claims)} carrier claims")
    return claims


def generate_discharge_summary(claim: InpatientClaim, beneficiary: Optional[Beneficiary]) -> str:
    """Generate a discharge summary from inpatient claim data."""
    provider = generate_provider_name()
    npi = claim.attending_npi or generate_npi()

    # Calculate patient age
    age = 70
    sex = "male"
    if beneficiary:
        age = calculate_age(beneficiary.birth_date, claim.admission_date)
        sex = get_sex_string(beneficiary.sex)

    # Build chronic conditions list
    chronic_list = []
    if beneficiary:
        for condition, has_it in beneficiary.chronic_conditions.items():
            if has_it:
                chronic_list.append(condition.replace("_", " ").title())

    # Build diagnosis list
    dx_list = []
    for dx in claim.diagnoses[:5]:
        desc = get_icd9_description(dx)
        dx_list.append(f"  - {desc} (ICD-9: {dx})")

    # Build procedure list
    proc_list = []
    for proc in claim.procedures[:3]:
        proc_list.append(f"  - Procedure code {proc}")

    admission_dt = format_date(claim.admission_date)
    discharge_dt = format_date(claim.discharge_date)

    note = f"""DISCHARGE SUMMARY

Patient ID: {claim.patient_id[:8]}...
Admission Date: {admission_dt}
Discharge Date: {discharge_dt}
Attending Physician: {provider}
NPI: {npi}

PATIENT DEMOGRAPHICS:
{age}-year-old {sex}

ADMITTING DIAGNOSIS:
{get_icd9_description(claim.admitting_diagnosis)} (ICD-9: {claim.admitting_diagnosis})

HOSPITAL COURSE:
Patient was admitted for evaluation and management of the above condition.
Hospital course was {"uncomplicated" if len(claim.diagnoses) < 3 else "complicated by multiple comorbidities"}.
Patient received appropriate medical management and showed clinical improvement.

DISCHARGE DIAGNOSES:
{chr(10).join(dx_list) if dx_list else "  - See admitting diagnosis"}

PROCEDURES PERFORMED:
{chr(10).join(proc_list) if proc_list else "  - No surgical procedures"}

CHRONIC CONDITIONS:
{chr(10).join(f"  - {c}" for c in chronic_list) if chronic_list else "  - None documented"}

DISCHARGE MEDICATIONS:
  - Continue home medications
  - New prescriptions as indicated by diagnoses

FOLLOW-UP:
  - PCP follow-up in 1-2 weeks
  - Specialty follow-up as needed

DRG: {claim.drg_code}

Electronically signed by {provider}
{discharge_dt}
"""
    return note


def generate_office_visit_note(claim: CarrierClaim, beneficiary: Optional[Beneficiary]) -> str:
    """Generate an office visit note from carrier claim data."""
    provider = generate_provider_name()
    npi = claim.provider_npi or generate_npi()

    # Calculate patient age
    age = 65
    sex = "female"
    if beneficiary:
        age = calculate_age(beneficiary.birth_date, claim.service_date)
        sex = get_sex_string(beneficiary.sex)

    # Primary diagnosis
    primary_dx = claim.diagnoses[0] if claim.diagnoses else "V7000"
    primary_desc = get_icd9_description(primary_dx)

    # Additional diagnoses
    additional_dx = []
    for dx in claim.diagnoses[1:4]:
        additional_dx.append(f"  - {get_icd9_description(dx)} ({dx})")

    # Procedures/services
    services = []
    for code in claim.hcpcs_codes[:3]:
        services.append(f"  - {get_hcpcs_description(code)} ({code})")

    service_dt = format_date(claim.service_date)

    # Generate vitals
    bp_sys = random.randint(110, 150)
    bp_dia = random.randint(65, 95)
    hr = random.randint(60, 100)
    temp = round(random.uniform(97.5, 99.0), 1)

    note = f"""OFFICE VISIT NOTE

Date of Service: {service_dt}
Patient ID: {claim.patient_id[:8]}...
Provider: {provider}
NPI: {npi}

CHIEF COMPLAINT:
Follow-up for {primary_desc.lower()}

HISTORY OF PRESENT ILLNESS:
This is a {age}-year-old {sex} presenting for evaluation and management.
Patient reports {"stable symptoms" if len(claim.diagnoses) < 3 else "multiple concerns"}.
{"Compliant with current medication regimen." if random.random() > 0.3 else "Reports occasional missed doses."}

VITAL SIGNS:
  BP: {bp_sys}/{bp_dia} mmHg
  HR: {hr} bpm
  Temp: {temp}F
  SpO2: {random.randint(95, 100)}% on room air

PHYSICAL EXAMINATION:
General: Alert, no acute distress
{"HEENT: Normocephalic, atraumatic" if random.random() > 0.5 else ""}
Cardiovascular: Regular rate and rhythm, no murmurs
Lungs: Clear to auscultation bilaterally
Abdomen: Soft, non-tender
Extremities: No edema

ASSESSMENT:
1. {primary_desc} (ICD-9: {primary_dx})
{chr(10).join(additional_dx) if additional_dx else ""}

SERVICES PROVIDED:
{chr(10).join(services) if services else "  - Evaluation and management"}

PLAN:
  - Continue current management
  - {"Lab work ordered" if "80053" in claim.hcpcs_codes or "80061" in claim.hcpcs_codes else "Routine monitoring"}
  - Follow-up in {"4-6 weeks" if len(claim.diagnoses) > 2 else "3 months"}

Electronically signed by {provider}
{service_dt}
"""
    return note


def generate_notes(
    num_discharge: int = 20,
    num_office: int = 30,
    include_flawed: bool = True
) -> int:
    """Generate medical notes from CMS data."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading CMS data...")
    beneficiaries = load_beneficiaries(limit=2000)
    inpatient_claims = load_inpatient_claims(limit=num_discharge * 2)
    carrier_claims = load_carrier_claims(limit=num_office * 2)

    generated = 0

    # Generate discharge summaries
    print(f"\nGenerating {num_discharge} discharge summaries...")
    for i, claim in enumerate(inpatient_claims[:num_discharge]):
        beneficiary = beneficiaries.get(claim.patient_id)
        note = generate_discharge_summary(claim, beneficiary)

        filename = f"discharge_summary_{i+1:03d}.txt"
        filepath = OUTPUT_DIR / filename
        filepath.write_text(note)
        generated += 1

    # Generate office visit notes
    print(f"Generating {num_office} office visit notes...")
    for i, claim in enumerate(carrier_claims[:num_office]):
        beneficiary = beneficiaries.get(claim.patient_id)
        note = generate_office_visit_note(claim, beneficiary)

        filename = f"office_visit_{i+1:03d}.txt"
        filepath = OUTPUT_DIR / filename
        filepath.write_text(note)
        generated += 1

    # Generate some intentionally flawed notes for testing
    if include_flawed:
        print("Generating flawed notes for testing...")
        flawed_notes = generate_flawed_cms_notes(beneficiaries, carrier_claims)
        for filename, content in flawed_notes:
            filepath = OUTPUT_DIR / filename
            filepath.write_text(content)
            generated += 1

    print(f"\nGenerated {generated} notes in {OUTPUT_DIR}")
    return generated


def generate_flawed_cms_notes(
    beneficiaries: Dict[str, Beneficiary],
    claims: List[CarrierClaim]
) -> List[tuple]:
    """Generate intentionally flawed notes based on real CMS data structure."""

    flawed = []

    # 1. Note with impossible dates (future dates)
    if claims:
        claim = claims[0]
        future_date = (datetime.now() + timedelta(days=60)).strftime("%m/%d/%Y")
        note = f"""OFFICE VISIT NOTE

Date of Service: {future_date}
Patient ID: {claim.patient_id[:8]}...
Provider: Dr. Future Doctor, MD
NPI: 1234567890

CHIEF COMPLAINT: Routine follow-up

ASSESSMENT:
1. Hypertension (ICD-9: 4019)

Labs drawn on {(datetime.now() + timedelta(days=90)).strftime("%m/%d/%Y")} - pending

Signed: {future_date}
"""
        flawed.append(("flawed_future_dates.txt", note))

    # 2. Note with dangerous medication dosages
    note = """DISCHARGE SUMMARY

Patient ID: SYNTH12345
Admission Date: 01/15/2024
Discharge Date: 01/18/2024
Attending: Dr. Wrong Dose, MD

DISCHARGE MEDICATIONS:
1. Metformin 5000mg PO TID (DANGEROUS - max 2550mg/day)
2. Lisinopril 200mg PO daily (DANGEROUS - max 80mg)
3. Warfarin 50mg PO daily (DANGEROUS - typical 2-10mg)
4. Metoprolol 500mg PO BID (DANGEROUS - max 400mg/day)

DIAGNOSES:
- Diabetes (ICD-9: 25000)
- Hypertension (ICD-9: 4019)

Dr. Wrong Dose, MD
"""
    flawed.append(("flawed_dangerous_doses.txt", note))

    # 3. Note with drug interactions
    note = """MEDICATION RECONCILIATION

Patient: CMS Patient
Date: 02/01/2024
Provider: Dr. Interaction, MD

CURRENT MEDICATIONS:
1. Warfarin 5mg daily - for AFib
2. Aspirin 325mg daily - cardiac protection
3. Ibuprofen 800mg TID - for arthritis
4. Naproxen 500mg BID PRN - for pain
5. Clopidogrel 75mg daily - post-stent

DIAGNOSES:
- Atrial Fibrillation (ICD-9: 42731)
- Coronary Artery Disease (ICD-9: 41401)
- Osteoarthritis (ICD-9: 71536)

WARNING: Multiple blood thinners + NSAIDs = HIGH BLEEDING RISK

Dr. Interaction, MD
"""
    flawed.append(("flawed_drug_interactions.txt", note))

    # 4. Note with invalid codes
    note = """OFFICE VISIT NOTE

Date: 01/20/2024
Provider: Dr. Bad Coder, MD
NPI: 0000000000

ASSESSMENT:
1. Headache (ICD-9: ZZZ.999) - INVALID CODE
2. Back pain (ICD-9: ABC.123) - INVALID CODE
3. Anxiety (ICD-9: 999.99) - INVALID CODE

PROCEDURES:
- Office visit (CPT: 12345) - INVALID CODE
- Blood draw (CPT: XXXXX) - INVALID CODE

Dr. Bad Coder, MD
"""
    flawed.append(("flawed_invalid_codes.txt", note))

    # 5. AI-generated style note
    note = """COMPREHENSIVE MEDICAL EVALUATION

Date of Evaluation: 02/10/2024
Patient Identifier: CMS-SYNTHETIC-001

INTRODUCTION:
It is important to note that this comprehensive evaluation aims to provide
a thorough assessment of the patient's current health status. Furthermore,
this document will outline various aspects in a systematic manner.

DETAILED HISTORY:
The patient presents with a constellation of symptoms that warrant careful
consideration. Additionally, it should be noted that these symptoms have
been gradually progressive. Moreover, the patient reports compliance with
the current medication regimen.

COMPREHENSIVE ASSESSMENT:
Based on the comprehensive evaluation, it can be concluded that the patient
is experiencing symptoms consistent with the documented conditions.
Furthermore, it is essential to note that continued monitoring is warranted.

In conclusion, this comprehensive evaluation has provided valuable insights
into the patient's health status. Additionally, the recommended treatment
approach should be implemented in a timely manner.

Respectfully submitted,
Dr. AI Generator, MD
"""
    flawed.append(("flawed_ai_generated_style.txt", note))

    return flawed


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert CMS data to medical notes")
    parser.add_argument("--discharge", type=int, default=20,
                        help="Number of discharge summaries to generate")
    parser.add_argument("--office", type=int, default=30,
                        help="Number of office visit notes to generate")
    parser.add_argument("--no-flawed", action="store_true",
                        help="Don't generate intentionally flawed notes")
    parser.add_argument("--list", action="store_true",
                        help="List generated files and exit")

    args = parser.parse_args()

    if args.list:
        if OUTPUT_DIR.exists():
            files = sorted(OUTPUT_DIR.glob("*.txt"))
            print(f"Generated notes in {OUTPUT_DIR}:")
            for f in files:
                size = f.stat().st_size
                print(f"  {f.name}: {size:,} bytes")
            print(f"\nTotal: {len(files)} files")
        else:
            print("No notes generated yet.")
        return

    # Check for source data
    if not SYNPUF_DIR.exists():
        print(f"Error: CMS data not found at {SYNPUF_DIR}")
        print("Run: python download_cms_data.py first")
        return

    # Generate notes
    count = generate_notes(
        num_discharge=args.discharge,
        num_office=args.office,
        include_flawed=not args.no_flawed
    )

    print(f"\nDone! Generated {count} medical notes.")
    print(f"Notes saved to: {OUTPUT_DIR}")
    print("\nTo run detection on these notes:")
    print("  python run_detection.py")


if __name__ == "__main__":
    main()
