"""
Test Data Generator for Medical Document Forgery Detection
Generates synthetic medical notes with various characteristics for testing.
"""

import os
import random
from datetime import datetime, timedelta
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent / "note_data"


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


# ============================================================================
# LEGITIMATE-LOOKING MEDICAL NOTES
# ============================================================================

LEGITIMATE_NOTES = [
    """PATIENT: [REDACTED]
DATE OF SERVICE: {date}
PROVIDER: Dr. Sarah Mitchell, MD
NPI: 1234567893

CHIEF COMPLAINT: Persistent cough for 2 weeks

HISTORY OF PRESENT ILLNESS:
Patient is a 45-year-old male presenting with productive cough x 2 weeks.
Reports yellow-green sputum production, worse in the morning. Denies fever,
chills, or night sweats. No hemoptysis. Associated with mild shortness of
breath on exertion. No chest pain.

PAST MEDICAL HISTORY:
- Hypertension, well-controlled on lisinopril 10mg daily
- Type 2 Diabetes Mellitus, A1C 6.8% on metformin 500mg BID

MEDICATIONS:
1. Lisinopril 10mg PO daily
2. Metformin 500mg PO BID

PHYSICAL EXAMINATION:
Vitals: BP 128/82, HR 76, RR 16, Temp 98.6F, SpO2 97% RA
General: Alert, no acute distress
Lungs: Scattered rhonchi bilateral bases, no wheezes

ASSESSMENT AND PLAN:
1. Acute bronchitis (ICD-10: J20.9)
   - Supportive care, increase fluids
   - Benzonatate 100mg TID PRN cough
   - Return if worsening or fever develops

Electronically signed by Dr. Sarah Mitchell, MD
{sign_date}
""",

    """PROGRESS NOTE
Patient Name: [REDACTED]
DOB: [REDACTED]
Date: {date}
Provider: James Chen, PA-C
Supervising Physician: Dr. Robert Williams, MD
NPI: 1982735462

SUBJECTIVE:
72 y/o female here for follow-up of CHF. Patient reports improved exercise
tolerance since last visit. Can now walk to mailbox without stopping.
Sleeping flat with 2 pillows. Denies PND or orthopnea. Weight stable per
patient. Compliant with medications and low-sodium diet.

OBJECTIVE:
BP: 118/74  HR: 68  Weight: 156 lbs (down 3 lbs from last visit)
JVP: Normal
Heart: RRR, no murmurs, gallops
Lungs: Clear to auscultation bilaterally
Extremities: Trace pedal edema bilateral, improved from 1+ last visit

LABS (from {lab_date}):
BNP: 245 (down from 580)
BMP: Na 138, K 4.2, Cr 1.1, BUN 22

ASSESSMENT:
1. CHF with reduced EF (35%) - NYHA Class II, improving (ICD-10: I50.22)
2. Hypertension - controlled (ICD-10: I10)

PLAN:
- Continue current regimen: Carvedilol 12.5mg BID, Lisinopril 20mg daily,
  Furosemide 40mg daily
- Continue sodium restriction <2g/day
- Daily weights, call if gain >3 lbs
- Follow up 4 weeks
- Repeat echo in 3 months

James Chen, PA-C
Electronically signed {sign_date}
""",

    """EMERGENCY DEPARTMENT NOTE

Date/Time: {date} {time}
Patient: [REDACTED]
MRN: [REDACTED]
Attending: Dr. Amanda Foster, MD
NPI: 1629384756

CHIEF COMPLAINT: Right ankle injury

HISTORY:
28 y/o male presents after twisting right ankle playing basketball
approximately 2 hours ago. Reports immediate pain and swelling. Unable to
bear weight. Denies numbness or tingling. No prior ankle injuries. AMPLE:
No allergies, no medications, PMH unremarkable, last ate 4 hours ago,
playing basketball when injured.

PHYSICAL EXAM:
Right ankle: Significant swelling and ecchymosis over lateral malleolus
Tenderness to palpation over ATFL and lateral malleolus
Unable to bear weight
Neurovascularly intact distally
Ottawa ankle rules: Positive

IMAGING:
X-ray right ankle 3 views: No acute fracture. Soft tissue swelling noted
laterally.

DIAGNOSIS:
Right ankle sprain, Grade II (ICD-10: S93.401A)

TREATMENT:
- RICE protocol discussed
- Air stirrup ankle brace applied
- Ibuprofen 600mg PO q6h PRN pain x 7 days
- Crutches provided with instruction
- Weight bearing as tolerated
- Ortho follow-up if not improving in 1 week

DISPOSITION: Discharged home in stable condition

Dr. Amanda Foster, MD
{sign_date} {sign_time}
"""
]


# ============================================================================
# AI-STYLE GENERATED NOTES (with telltale AI patterns)
# ============================================================================

AI_GENERATED_NOTES = [
    """COMPREHENSIVE MEDICAL EVALUATION REPORT

Date of Evaluation: {date}
Patient Identifier: [REDACTED]
Healthcare Provider: Dr. Michael Thompson, MD

INTRODUCTION AND OVERVIEW:
It is important to note that this comprehensive medical evaluation aims to
provide a thorough assessment of the patient's current health status.
Furthermore, this document will outline the various aspects of the patient's
medical condition in a systematic and organized manner.

DETAILED HISTORY OF PRESENT ILLNESS:
The patient presents with a constellation of symptoms that warrant careful
consideration. Additionally, it should be noted that the patient has been
experiencing these symptoms for approximately two weeks. The symptoms include,
but are not limited to, the following:
- Persistent fatigue
- Mild headaches
- General malaise

It's worth mentioning that these symptoms have been gradually progressive in
nature. Furthermore, the patient reports that rest and over-the-counter
medications have provided minimal relief.

COMPREHENSIVE PHYSICAL EXAMINATION:
A thorough physical examination was conducted, revealing the following findings:
- Vital signs within normal limits
- General appearance: Well-developed, well-nourished individual
- Cardiovascular examination: Regular rate and rhythm
- Respiratory examination: Clear to auscultation bilaterally

ASSESSMENT AND CLINICAL IMPRESSION:
Based on the comprehensive evaluation, it can be concluded that the patient
is experiencing symptoms consistent with viral syndrome. It is essential to
note that further monitoring may be warranted.

DETAILED TREATMENT PLAN:
The following comprehensive approach is recommended:
1. Adequate rest and hydration
2. Over-the-counter analgesics as needed
3. Follow-up appointment in one week

In conclusion, this comprehensive evaluation has provided valuable insights
into the patient's current health status.

Respectfully submitted,
Dr. Michael Thompson, MD
{sign_date}
""",

    """PATIENT CONSULTATION NOTE

Date: {date}
RE: Comprehensive Health Assessment

Dear Colleague,

I am writing to provide you with a detailed summary of my consultation with
this patient. It is important to note that this evaluation was conducted in
a thorough and systematic manner.

BACKGROUND AND CONTEXT:
The patient, a 55-year-old individual, was referred for evaluation of
persistent lower back pain. Additionally, it should be mentioned that the
patient has a history of sedentary lifestyle and occasional physical activity.

DETAILED FINDINGS:
Upon careful examination, several noteworthy observations were made:

Firstly, the patient demonstrates limited range of motion in the lumbar spine.
Secondly, there is evidence of paraspinal muscle tenderness. Furthermore,
neurological examination reveals no focal deficits, which is reassuring.

It's worth noting that the patient's pain appears to be mechanical in nature.
Moreover, there are no red flag symptoms that would suggest a more serious
underlying condition.

COMPREHENSIVE RECOMMENDATIONS:
Based on my thorough evaluation, I would like to recommend the following
comprehensive treatment approach:

1. Physical therapy referral for core strengthening
2. NSAIDs for pain management as needed
3. Ergonomic workplace assessment
4. Gradual return to physical activity

In summary, this patient would benefit from a multimodal approach to treatment.
I hope this comprehensive evaluation proves helpful in guiding further
management.

Best regards,
Dr. Jennifer Walsh, MD
Orthopedic Consultation Service
{sign_date}
"""
]


# ============================================================================
# NOTES WITH INTENTIONAL ERRORS/RED FLAGS
# ============================================================================

def generate_flawed_notes():
    """Generate notes with various detectable flaws."""

    # Note with date inconsistencies
    future_date = (datetime.now() + timedelta(days=30)).strftime("%m/%d/%Y")
    past_date = "03/15/1952"

    date_error_note = f"""OFFICE VISIT NOTE

Date of Service: {future_date}
Patient: [REDACTED]
Provider: Dr. William Hayes, MD

CHIEF COMPLAINT: Annual physical examination

HISTORY:
Patient last seen on {past_date} for similar complaints. Labs drawn on
{(datetime.now() + timedelta(days=45)).strftime("%m/%d/%Y")} showed normal results.

ASSESSMENT:
1. Routine health maintenance
2. Hypertension - stable

PLAN:
- Continue current medications
- Follow up in 1 year

Dr. William Hayes, MD
Signed: {(datetime.now() + timedelta(days=60)).strftime("%m/%d/%Y")}
"""

    # Note with dangerous dosages
    dosage_error_note = """PRESCRIPTION NOTE

Date: {date}
Patient: [REDACTED]
Prescriber: Dr. Nancy Drew, MD
NPI: 1357924680

MEDICATIONS PRESCRIBED:

1. Metformin 5000mg PO TID
   #270 tablets, 3 refills

2. Lisinopril 200mg PO daily
   #90 tablets, 3 refills

3. Warfarin 50mg PO daily
   #30 tablets, 0 refills

4. Oxycodone 80mg PO q4h PRN pain
   #180 tablets, 5 refills

DIAGNOSIS: Type 2 Diabetes (E11.9), Hypertension (I10)

Electronically signed,
Dr. Nancy Drew, MD
""".format(date=datetime.now().strftime("%m/%d/%Y"))

    # Note with drug interaction issues
    interaction_note = """DISCHARGE MEDICATION LIST

Patient: [REDACTED]
Discharge Date: {date}
Attending: Dr. Robert Smith, MD

ACTIVE MEDICATIONS AT DISCHARGE:

1. Warfarin 5mg PO daily (for AFib)
2. Aspirin 325mg PO daily
3. Ibuprofen 800mg PO TID (for arthritis)
4. Naproxen 500mg PO BID PRN
5. Clopidogrel 75mg PO daily

DIAGNOSES:
- Atrial Fibrillation (I48.91)
- Osteoarthritis (M19.90)
- CAD s/p stent (Z95.5)

Patient educated on medications. Follow up with PCP in 1 week.

Dr. Robert Smith, MD
""".format(date=datetime.now().strftime("%m/%d/%Y"))

    # Note with invalid medical codes/NPI
    invalid_codes_note = """MEDICAL EVALUATION

Date: {date}
Provider: Dr. Fake Doctor, MD
NPI: 1234567890

DIAGNOSES:
1. Chronic back pain (ICD-10: ZZZ.999)
2. Anxiety disorder (ICD-10: ABC.123)
3. Hypertension (ICD-10: 999.99)

MEDICATIONS:
- Lisinopril 10mg daily
- Fakemedicationol 500mg BID
- Madeupitab 250mg TID

Follow up as needed.

Dr. Fake Doctor, MD
""".format(date=datetime.now().strftime("%m/%d/%Y"))

    # Note with mixed formatting (copy-paste indicators)
    mixed_format_note = """PATIENT ENCOUNTER NOTE

Date: {date}

SUBJECTIVE: Patient presents with chest pain x 2 days

---copied from external system---
The patient is a 67 year old male with PMH significant for
CORONARY ARTERY DISEASE, HYPERTENSION, AND HYPERLIPIDEMIA
who presents today for evaluation of chest discomfort
---end copy---

Physical Exam:
bp: 145/92  hr: 88  rr: 18
HEART - regular rate and rhythm
lungs - clear

A/P:
1.) CHEST PAIN - likely musculoskeletal
     >> continue current meds
     >> f/u prn

** NOTE: This section imported from EMR template v2.3 **
PATIENT EDUCATION PROVIDED:
- Discussed warning signs requiring immediate attention
- Reviewed medication compliance
- Diet and exercise counseling provided
** END IMPORTED SECTION **

Signed electronically,
Dr. Various Authors
{sign_date}
""".format(date=datetime.now().strftime("%m/%d/%Y"),
           sign_date=datetime.now().strftime("%m/%d/%Y"))

    # Note with terminology/spelling errors
    terminology_error_note = """CLINIC NOTE

Date: {date}
Patient: [REDACTED]

CC: Stomach problems

The pt came in cuz they been having tummy aches for like a week. They said
it hurts real bad after eating greasy stuff. No throwing up or nothing.
Been taking tums but it aint helping much.

Exam: Belly is soft, hurts a little when I push on the upper part

Assesment: Probably got some acid reflux or maybe gastitus

Plan:
- Gonna give em some prilosec
- Told em to stop eating junk food
- Come back if it dont get better

Doc Smith
""".format(date=datetime.now().strftime("%m/%d/%Y"))

    return [
        ("date_inconsistencies.txt", date_error_note),
        ("dangerous_dosages.txt", dosage_error_note),
        ("drug_interactions.txt", interaction_note),
        ("invalid_codes.txt", invalid_codes_note),
        ("mixed_formatting.txt", mixed_format_note),
        ("terminology_errors.txt", terminology_error_note),
    ]


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_test_data():
    """Generate all test data files."""
    ensure_output_dir()

    now = datetime.now()
    file_count = 0

    # Generate legitimate notes
    print("\nGenerating legitimate-looking medical notes...")
    for i, template in enumerate(LEGITIMATE_NOTES, 1):
        date = (now - timedelta(days=random.randint(1, 30))).strftime("%m/%d/%Y")
        lab_date = (now - timedelta(days=random.randint(5, 14))).strftime("%m/%d/%Y")
        sign_date = date
        time = f"{random.randint(8,17):02d}:{random.randint(0,59):02d}"
        sign_time = time

        content = template.format(
            date=date,
            lab_date=lab_date,
            sign_date=sign_date,
            time=time,
            sign_time=sign_time
        )

        filename = f"legitimate_note_{i}.txt"
        filepath = OUTPUT_DIR / filename
        filepath.write_text(content)
        print(f"  Created: {filename}")
        file_count += 1

    # Generate AI-style notes
    print("\nGenerating AI-pattern medical notes...")
    for i, template in enumerate(AI_GENERATED_NOTES, 1):
        date = (now - timedelta(days=random.randint(1, 30))).strftime("%m/%d/%Y")
        sign_date = date

        content = template.format(date=date, sign_date=sign_date)

        filename = f"ai_generated_note_{i}.txt"
        filepath = OUTPUT_DIR / filename
        filepath.write_text(content)
        print(f"  Created: {filename}")
        file_count += 1

    # Generate flawed notes
    print("\nGenerating notes with intentional flaws...")
    for filename, content in generate_flawed_notes():
        filepath = OUTPUT_DIR / filename
        filepath.write_text(content)
        print(f"  Created: {filename}")
        file_count += 1

    print(f"\n{'='*50}")
    print(f"Generated {file_count} test files in: {OUTPUT_DIR}")
    print(f"{'='*50}")

    # Print summary of what each file tests
    print("\nTEST FILE SUMMARY:")
    print("-" * 50)
    print("LEGITIMATE NOTES (should pass most checks):")
    print("  - legitimate_note_1.txt: Standard office visit")
    print("  - legitimate_note_2.txt: CHF follow-up with labs")
    print("  - legitimate_note_3.txt: ED ankle injury note")
    print()
    print("AI-GENERATED PATTERNS (should trigger AI detection):")
    print("  - ai_generated_note_1.txt: Overuse of 'comprehensive', 'furthermore'")
    print("  - ai_generated_note_2.txt: Formal transitions, verbose style")
    print()
    print("INTENTIONAL FLAWS (should trigger specific detectors):")
    print("  - date_inconsistencies.txt: Future dates, impossible timelines")
    print("  - dangerous_dosages.txt: Unsafe medication doses")
    print("  - drug_interactions.txt: Dangerous drug combinations")
    print("  - invalid_codes.txt: Fake ICD-10 codes, invalid NPI")
    print("  - mixed_formatting.txt: Copy-paste artifacts")
    print("  - terminology_errors.txt: Unprofessional language, misspellings")


if __name__ == "__main__":
    generate_all_test_data()
