"""
Medical Entity Validator Module
Validates medical entities like drug names, ICD codes, and procedures.
"""

import re
from typing import List, Set
from dataclasses import dataclass, field


@dataclass
class MedicalValidationResult:
    """Results from medical entity validation."""
    drugs_found: List[str] = field(default_factory=list)
    invalid_drugs: List[str] = field(default_factory=list)
    icd_codes_found: List[str] = field(default_factory=list)
    invalid_icd_codes: List[str] = field(default_factory=list)
    npi_numbers_found: List[str] = field(default_factory=list)
    invalid_npi_numbers: List[str] = field(default_factory=list)
    suspicious_combinations: List[str] = field(default_factory=list)
    all_issues: List[str] = field(default_factory=list)
    risk_score: float = 0.0


class MedicalEntityValidator:
    """Validate medical entities in documents."""

    # Common legitimate drug names (simplified list)
    VALID_DRUGS: Set[str] = {
        # Pain/Inflammation
        'acetaminophen', 'tylenol', 'ibuprofen', 'advil', 'motrin', 'aspirin',
        'naproxen', 'aleve', 'celecoxib', 'celebrex', 'meloxicam',
        # Antibiotics
        'amoxicillin', 'azithromycin', 'zithromax', 'ciprofloxacin', 'cipro',
        'doxycycline', 'metronidazole', 'flagyl', 'penicillin', 'cephalexin',
        'augmentin', 'bactrim', 'clindamycin',
        # Cardiovascular
        'lisinopril', 'amlodipine', 'norvasc', 'metoprolol', 'atenolol',
        'losartan', 'hydrochlorothiazide', 'hctz', 'furosemide', 'lasix',
        'warfarin', 'coumadin', 'aspirin', 'clopidogrel', 'plavix',
        # Diabetes
        'metformin', 'glucophage', 'glipizide', 'insulin', 'januvia',
        'jardiance', 'ozempic', 'trulicity',
        # Cholesterol
        'atorvastatin', 'lipitor', 'simvastatin', 'zocor', 'rosuvastatin',
        'crestor', 'pravastatin',
        # GI
        'omeprazole', 'prilosec', 'pantoprazole', 'protonix', 'famotidine',
        'pepcid', 'ondansetron', 'zofran',
        # Mental Health
        'sertraline', 'zoloft', 'fluoxetine', 'prozac', 'escitalopram',
        'lexapro', 'duloxetine', 'cymbalta', 'bupropion', 'wellbutrin',
        'alprazolam', 'xanax', 'lorazepam', 'ativan', 'clonazepam',
        'trazodone', 'quetiapine', 'seroquel',
        # Respiratory
        'albuterol', 'ventolin', 'fluticasone', 'montelukast', 'singulair',
        'prednisone', 'methylprednisolone',
        # Thyroid
        'levothyroxine', 'synthroid',
        # Other common
        'gabapentin', 'neurontin', 'pregabalin', 'lyrica', 'tramadol',
        'cyclobenzaprine', 'flexeril', 'diphenhydramine', 'benadryl',
        'cetirizine', 'zyrtec', 'loratadine', 'claritin',
    }

    # Known dangerous drug interactions (simplified)
    DANGEROUS_COMBINATIONS = [
        ({'warfarin', 'coumadin'}, {'aspirin', 'ibuprofen', 'naproxen'},
         'Blood thinner with NSAID - bleeding risk'),
        ({'metformin'}, {'contrast', 'dye'},
         'Metformin with contrast dye - kidney risk'),
        ({'ssri', 'sertraline', 'fluoxetine', 'escitalopram'}, {'tramadol'},
         'SSRI with tramadol - serotonin syndrome risk'),
        ({'alprazolam', 'lorazepam', 'clonazepam'}, {'opioid', 'hydrocodone', 'oxycodone'},
         'Benzodiazepine with opioid - respiratory depression risk'),
    ]

    # ICD-10 code pattern
    ICD10_PATTERN = r'\b([A-Z]\d{2}(?:\.\d{1,4})?)\b'

    # NPI (National Provider Identifier) pattern - 10 digits
    NPI_PATTERN = r'\b(\d{10})\b'

    def validate(self, text: str) -> MedicalValidationResult:
        """
        Validate medical entities in text.

        Args:
            text: Medical document text

        Returns:
            MedicalValidationResult with findings
        """
        result = MedicalValidationResult()
        risk_score = 0.0

        # Extract and validate drugs
        drug_findings = self._validate_drugs(text)
        result.drugs_found = drug_findings[0]
        result.invalid_drugs = drug_findings[1]
        risk_score += 0.15 * len(result.invalid_drugs)

        # Check for dangerous combinations
        result.suspicious_combinations = self._check_drug_combinations(
            result.drugs_found
        )
        risk_score += 0.2 * len(result.suspicious_combinations)

        # Extract and validate ICD codes
        icd_findings = self._validate_icd_codes(text)
        result.icd_codes_found = icd_findings[0]
        result.invalid_icd_codes = icd_findings[1]
        risk_score += 0.1 * len(result.invalid_icd_codes)

        # Extract and validate NPI numbers
        npi_findings = self._validate_npi_numbers(text)
        result.npi_numbers_found = npi_findings[0]
        result.invalid_npi_numbers = npi_findings[1]
        risk_score += 0.25 * len(result.invalid_npi_numbers)

        # Compile all issues
        result.all_issues = (
            [f"Potentially invalid drug: {d}" for d in result.invalid_drugs] +
            result.suspicious_combinations +
            [f"Invalid ICD code format: {c}" for c in result.invalid_icd_codes] +
            [f"Invalid NPI number: {n}" for n in result.invalid_npi_numbers]
        )

        result.risk_score = min(risk_score, 1.0)
        return result

    def _validate_drugs(self, text: str) -> tuple:
        """Extract and validate drug names."""
        found_drugs = []
        invalid_drugs = []

        # Simple word-based extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

        for word in words:
            if word in self.VALID_DRUGS:
                if word not in found_drugs:
                    found_drugs.append(word)
            elif self._looks_like_drug_name(word):
                # Check if it might be a drug name we don't recognize
                if word not in invalid_drugs and word not in found_drugs:
                    invalid_drugs.append(word)

        return found_drugs, invalid_drugs

    def _looks_like_drug_name(self, word: str) -> bool:
        """Heuristic to detect if a word might be a drug name."""
        # Drug names often end in these suffixes
        drug_suffixes = [
            'mab', 'nib', 'vir', 'pril', 'sartan', 'statin', 'olol',
            'azole', 'mycin', 'cillin', 'cycline', 'pam', 'lam',
            'pine', 'done', 'phen', 'zine', 'ide', 'ate'
        ]

        # Common medical but non-drug words to exclude
        excluded = {
            'patient', 'treatment', 'medicine', 'hospital', 'physician',
            'diagnosis', 'procedure', 'examination', 'prescription',
            'medication', 'symptoms', 'condition', 'disease', 'disorder',
            'history', 'allergies', 'assessment', 'evaluation', 'laboratory',
            'moderate', 'include', 'complete', 'continue', 'indicate',
            'demonstrate', 'evaluate', 'determine', 'appropriate', 'provide'
        }

        if word in excluded:
            return False

        for suffix in drug_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return True

        return False

    def _check_drug_combinations(self, drugs: List[str]) -> List[str]:
        """Check for dangerous drug combinations."""
        issues = []
        drugs_set = set(drugs)

        for group1, group2, warning in self.DANGEROUS_COMBINATIONS:
            found1 = drugs_set & group1
            found2 = drugs_set & group2
            if found1 and found2:
                issues.append(
                    f"Potentially dangerous combination: {found1} with {found2} - {warning}"
                )

        return issues

    def _validate_icd_codes(self, text: str) -> tuple:
        """Extract and validate ICD-10 codes."""
        found_codes = []
        invalid_codes = []

        matches = re.findall(self.ICD10_PATTERN, text)

        for code in matches:
            if self._is_valid_icd10_format(code):
                found_codes.append(code)
            else:
                invalid_codes.append(code)

        return found_codes, invalid_codes

    def _is_valid_icd10_format(self, code: str) -> bool:
        """Check if code follows valid ICD-10 format."""
        # ICD-10 codes: Letter + 2 digits + optional decimal + up to 4 more chars
        # Valid categories: A-Z except U (reserved)
        if not code or len(code) < 3:
            return False

        first_char = code[0].upper()
        if first_char == 'U':  # Reserved for special purposes
            return False

        # Check basic format
        if not re.match(r'^[A-TV-Z]\d{2}(\.\d{1,4})?$', code):
            return False

        return True

    def _validate_npi_numbers(self, text: str) -> tuple:
        """Extract and validate NPI numbers using Luhn algorithm."""
        found_npis = []
        invalid_npis = []

        # Look for 10-digit numbers that might be NPIs
        matches = re.findall(self.NPI_PATTERN, text)

        for npi in matches:
            # NPI numbers start with 1 or 2
            if npi[0] in ('1', '2'):
                if self._validate_npi_checksum(npi):
                    found_npis.append(npi)
                else:
                    invalid_npis.append(npi)

        return found_npis, invalid_npis

    def _validate_npi_checksum(self, npi: str) -> bool:
        """Validate NPI using Luhn algorithm with healthcare prefix."""
        if len(npi) != 10:
            return False

        # Prepend 80840 for Luhn check (standard healthcare identifier prefix)
        full_number = '80840' + npi

        # Luhn algorithm
        total = 0
        for i, digit in enumerate(reversed(full_number)):
            d = int(digit)
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            total += d

        return total % 10 == 0
