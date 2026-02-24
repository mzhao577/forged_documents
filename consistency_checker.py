"""
Consistency Checker Module
Validates internal consistency of medical documents.
"""

import re
from datetime import datetime
from typing import List, Tuple
from dataclasses import dataclass, field


@dataclass
class ConsistencyResult:
    """Results from consistency analysis."""
    dates_found: List[str] = field(default_factory=list)
    date_inconsistencies: List[str] = field(default_factory=list)
    dosage_issues: List[str] = field(default_factory=list)
    formatting_issues: List[str] = field(default_factory=list)
    terminology_issues: List[str] = field(default_factory=list)
    all_issues: List[str] = field(default_factory=list)
    risk_score: float = 0.0


class ConsistencyChecker:
    """Check medical document consistency."""

    # Common date formats in medical documents
    DATE_PATTERNS = [
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',  # MM/DD/YYYY or DD/MM/YYYY
        r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',     # YYYY-MM-DD
        r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b',  # Month DD, YYYY
        r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b',     # DD Month YYYY
    ]

    # Common dosage patterns
    DOSAGE_PATTERNS = [
        r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|mL|units?|IU)\b',
        r'\b(\d+(?:\.\d+)?)\s*(tablets?|caps?|capsules?)\b',
    ]

    # Dangerous dosage thresholds (simplified)
    DOSAGE_LIMITS = {
        'acetaminophen': {'max_single': 1000, 'unit': 'mg'},
        'ibuprofen': {'max_single': 800, 'unit': 'mg'},
        'aspirin': {'max_single': 1000, 'unit': 'mg'},
        'metformin': {'max_single': 1000, 'unit': 'mg'},
        'lisinopril': {'max_single': 80, 'unit': 'mg'},
        'atorvastatin': {'max_single': 80, 'unit': 'mg'},
        'omeprazole': {'max_single': 40, 'unit': 'mg'},
        'amoxicillin': {'max_single': 1000, 'unit': 'mg'},
    }

    # Commonly misspelled medical terms
    MISSPELLINGS = {
        'perscription': 'prescription',
        'presciption': 'prescription',
        'diagnosys': 'diagnosis',
        'symtoms': 'symptoms',
        'symptons': 'symptoms',
        'paitent': 'patient',
        'patiant': 'patient',
        'medicaiton': 'medication',
        'medecine': 'medicine',
        'hospitol': 'hospital',
        'physcian': 'physician',
        'physican': 'physician',
        'treatement': 'treatment',
        'refferal': 'referral',
        'referel': 'referral',
        'labratory': 'laboratory',
        'labrotory': 'laboratory',
        'surgury': 'surgery',
        'dosege': 'dosage',
        'alergies': 'allergies',
        'alergic': 'allergic',
    }

    def check_consistency(self, text: str) -> ConsistencyResult:
        """
        Perform comprehensive consistency checks.

        Args:
            text: Medical document text

        Returns:
            ConsistencyResult with findings
        """
        result = ConsistencyResult()
        risk_score = 0.0

        # Check dates
        date_issues = self._check_dates(text)
        result.dates_found = date_issues[0]
        result.date_inconsistencies = date_issues[1]
        risk_score += 0.15 * len(result.date_inconsistencies)

        # Check dosages
        result.dosage_issues = self._check_dosages(text)
        risk_score += 0.2 * len(result.dosage_issues)

        # Check formatting
        result.formatting_issues = self._check_formatting(text)
        risk_score += 0.1 * len(result.formatting_issues)

        # Check terminology/spelling
        result.terminology_issues = self._check_terminology(text)
        risk_score += 0.1 * len(result.terminology_issues)

        # Compile all issues
        result.all_issues = (
            result.date_inconsistencies +
            result.dosage_issues +
            result.formatting_issues +
            result.terminology_issues
        )

        result.risk_score = min(risk_score, 1.0)
        return result

    def _check_dates(self, text: str) -> Tuple[List[str], List[str]]:
        """Check for date inconsistencies."""
        dates_found = []
        issues = []

        for pattern in self.DATE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates_found.extend(matches)

        # Parse and validate dates
        parsed_dates = []
        for date_str in dates_found:
            parsed = self._parse_date(date_str)
            if parsed:
                parsed_dates.append((date_str, parsed))

                # Check for future dates
                if parsed > datetime.now():
                    issues.append(f"Future date detected: {date_str}")

                # Check for very old dates (before 1900)
                if parsed.year < 1900:
                    issues.append(f"Implausible historical date: {date_str}")

        # Check for date ordering issues (if document has start/end dates)
        if len(parsed_dates) >= 2:
            # Look for admission/discharge patterns
            text_lower = text.lower()
            if 'admission' in text_lower and 'discharge' in text_lower:
                # Simple check: discharge shouldn't be before admission
                admission_idx = text_lower.find('admission')
                discharge_idx = text_lower.find('discharge')

                if admission_idx < discharge_idx and len(parsed_dates) >= 2:
                    # First date near admission, second near discharge
                    first_date = parsed_dates[0][1]
                    for date_str, date_obj in parsed_dates[1:]:
                        if date_obj < first_date:
                            issues.append(
                                f"Possible date order issue: {date_str} appears "
                                f"after {parsed_dates[0][0]} but is earlier"
                            )

        return dates_found, issues

    def _parse_date(self, date_str: str) -> datetime:
        """Try to parse various date formats."""
        formats = [
            "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y",
            "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
            "%m/%d/%y", "%d/%m/%y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        return None

    def _check_dosages(self, text: str) -> List[str]:
        """Check for suspicious dosages."""
        issues = []
        text_lower = text.lower()

        for drug, limits in self.DOSAGE_LIMITS.items():
            if drug in text_lower:
                # Find dosages near this drug mention
                drug_pattern = rf'{drug}\s+(\d+(?:\.\d+)?)\s*{limits["unit"]}'
                matches = re.findall(drug_pattern, text_lower)

                for match in matches:
                    try:
                        dose = float(match)
                        if dose > limits['max_single']:
                            issues.append(
                                f"Potentially dangerous dosage: {drug} {dose}{limits['unit']} "
                                f"(max single dose typically {limits['max_single']}{limits['unit']})"
                            )
                    except ValueError:
                        pass

        # Check for suspicious round numbers or patterns
        dosage_matches = re.findall(r'(\d+)\s*mg', text_lower)
        dosage_values = [int(d) for d in dosage_matches if d.isdigit()]

        # Check for unusual patterns (all same dosage)
        if len(dosage_values) > 3 and len(set(dosage_values)) == 1:
            issues.append(
                f"Suspicious pattern: all dosages are {dosage_values[0]}mg"
            )

        return issues

    def _check_formatting(self, text: str) -> List[str]:
        """Check for formatting inconsistencies."""
        issues = []

        # Check for mixed case inconsistencies in headers
        lines = text.split('\n')
        header_styles = []

        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) < 50:
                if stripped.isupper():
                    header_styles.append('UPPER')
                elif stripped.istitle():
                    header_styles.append('Title')
                elif ':' in stripped and len(stripped.split(':')[0]) < 30:
                    header_styles.append('label:')

        # Too many different styles might indicate copy-paste from different sources
        if len(set(header_styles)) > 3 and len(header_styles) > 5:
            issues.append(
                "Inconsistent formatting styles detected - possible copy-paste from multiple sources"
            )

        # Check for unusual spacing patterns
        double_spaces = len(re.findall(r'  +', text))
        if double_spaces > 10:
            issues.append(
                f"Excessive irregular spacing ({double_spaces} instances of multiple spaces)"
            )

        # Check for mixed line endings
        crlf_count = text.count('\r\n')
        lf_count = text.count('\n') - crlf_count
        if crlf_count > 0 and lf_count > 0:
            issues.append("Mixed line ending styles (possible document merging)")

        return issues

    def _check_terminology(self, text: str) -> List[str]:
        """Check for misspelled medical terminology."""
        issues = []
        text_lower = text.lower()

        for misspelled, correct in self.MISSPELLINGS.items():
            if misspelled in text_lower:
                issues.append(
                    f"Misspelled medical term: '{misspelled}' should be '{correct}'"
                )

        # Check for informal language unusual in medical documents
        informal_terms = [
            ('gonna', 'going to'),
            ('wanna', 'want to'),
            ('kinda', 'kind of'),
            ('lots of', 'numerous/multiple'),
            ('a bunch of', 'multiple/several'),
        ]

        for informal, formal in informal_terms:
            if informal in text_lower:
                issues.append(
                    f"Informal language detected: '{informal}' (unusual in medical documents)"
                )

        return issues
