"""
Style Analyzer Module
Analyzes writing patterns and style consistency in medical documents.
"""

import re
import math
from typing import List, Dict
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class StyleAnalysisResult:
    """Results from style analysis."""
    avg_sentence_length: float = 0.0
    sentence_length_variance: float = 0.0
    vocabulary_richness: float = 0.0
    avg_word_length: float = 0.0
    formality_score: float = 0.0
    repetition_score: float = 0.0
    style_inconsistencies: List[str] = field(default_factory=list)
    statistical_anomalies: List[str] = field(default_factory=list)
    all_issues: List[str] = field(default_factory=list)
    risk_score: float = 0.0


class StyleAnalyzer:
    """Analyze writing style patterns."""

    # Formal medical writing indicators
    FORMAL_INDICATORS = [
        'patient presents', 'chief complaint', 'history of present illness',
        'physical examination', 'assessment', 'plan', 'differential diagnosis',
        'laboratory findings', 'radiological findings', 'impression',
        'recommendations', 'follow-up', 'prognosis', 'discussed with patient'
    ]

    # Informal writing indicators
    INFORMAL_INDICATORS = [
        'i think', "i'm", "don't", "can't", "won't", 'gonna', 'wanna',
        'kinda', 'sorta', 'yeah', 'nope', 'ok ', 'okay ',
        'pretty much', 'a lot', 'lots of', 'stuff', 'things'
    ]

    # AI-generated text patterns (common in LLM outputs)
    AI_PATTERNS = [
        r'\bdelve\b',
        r'\bfurthermore\b',
        r'\bmoreover\b',
        r'\badditionally\b',
        r'\bIt\'s important to note\b',
        r'\bIt is worth noting\b',
        r'\bIn conclusion\b',
        r'\bTo summarize\b',
        r'\bAs an AI\b',
        r'\bI don\'t have access to\b',
        r'\bcomprehensive\b.*\bapproach\b',
        r'\bholistic\b.*\bapproach\b',
        r'\bIn summary,?\s',
    ]

    def analyze(self, text: str) -> StyleAnalysisResult:
        """
        Analyze text style and patterns.

        Args:
            text: Document text to analyze

        Returns:
            StyleAnalysisResult with findings
        """
        result = StyleAnalysisResult()
        risk_score = 0.0

        if not text or len(text) < 100:
            result.all_issues.append("Text too short for style analysis")
            return result

        # Basic text statistics
        sentences = self._split_sentences(text)
        words = self._extract_words(text)

        if not sentences or not words:
            return result

        # Calculate metrics
        result.avg_sentence_length = len(words) / len(sentences)
        result.sentence_length_variance = self._calculate_variance(
            [len(self._extract_words(s)) for s in sentences]
        )
        result.vocabulary_richness = len(set(words)) / len(words)
        result.avg_word_length = sum(len(w) for w in words) / len(words)

        # Check formality
        formality = self._check_formality(text)
        result.formality_score = formality[0]
        if formality[1]:
            result.style_inconsistencies.extend(formality[1])
            risk_score += 0.1 * len(formality[1])

        # Check for repetition
        repetition = self._check_repetition(text, words)
        result.repetition_score = repetition[0]
        if repetition[1]:
            result.style_inconsistencies.extend(repetition[1])
            risk_score += 0.15 * len(repetition[1])

        # Check for AI patterns
        ai_patterns = self._check_ai_patterns(text)
        if ai_patterns:
            result.statistical_anomalies.extend(ai_patterns)
            risk_score += 0.2 * len(ai_patterns)

        # Check statistical anomalies
        stat_anomalies = self._check_statistical_anomalies(result, sentences)
        if stat_anomalies:
            result.statistical_anomalies.extend(stat_anomalies)
            risk_score += 0.15 * len(stat_anomalies)

        # Compile all issues
        result.all_issues = (
            result.style_inconsistencies +
            result.statistical_anomalies
        )

        result.risk_score = min(risk_score, 1.0)
        return result

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text."""
        return [w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', text)]

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if not values or len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def _check_formality(self, text: str) -> tuple:
        """Check formality level and consistency."""
        text_lower = text.lower()
        issues = []

        formal_count = sum(1 for ind in self.FORMAL_INDICATORS if ind in text_lower)
        informal_count = sum(1 for ind in self.INFORMAL_INDICATORS if ind in text_lower)

        # Calculate formality score (0-1, higher = more formal)
        total = formal_count + informal_count
        if total == 0:
            formality_score = 0.5
        else:
            formality_score = formal_count / total

        # Flag inconsistency: formal medical document with informal language
        if formal_count > 3 and informal_count > 2:
            issues.append(
                "Mixed formality: formal medical structure with informal language"
            )

        # Medical documents should generally be formal
        if informal_count > 5:
            issues.append(
                f"Excessive informal language ({informal_count} instances) unusual for medical document"
            )

        return formality_score, issues

    def _check_repetition(self, text: str, words: List[str]) -> tuple:
        """Check for unusual repetition patterns."""
        issues = []

        # Check phrase repetition
        # Split into 3-grams
        if len(words) >= 3:
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)

            # Find highly repeated phrases
            for trigram, count in trigram_counts.most_common(10):
                if count > 3 and count / len(trigrams) > 0.05:
                    issues.append(
                        f"Phrase repeated {count} times: '{trigram}'"
                    )

        # Check paragraph repetition (sign of copy-paste)
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        if len(paragraphs) > 1:
            para_set = set()
            for para in paragraphs:
                # Normalize for comparison
                normalized = ' '.join(para.lower().split())
                if normalized in para_set:
                    issues.append("Duplicate paragraph detected")
                    break
                para_set.add(normalized)

        # Calculate repetition score
        word_counts = Counter(words)
        if words:
            top_5_freq = sum(count for _, count in word_counts.most_common(5))
            repetition_score = top_5_freq / len(words)
        else:
            repetition_score = 0.0

        return repetition_score, issues

    def _check_ai_patterns(self, text: str) -> List[str]:
        """Check for patterns common in AI-generated text."""
        issues = []

        for pattern in self.AI_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                issues.append(
                    f"AI-associated phrase detected: '{matches[0]}'"
                )

        return issues

    def _check_statistical_anomalies(
        self,
        result: StyleAnalysisResult,
        sentences: List[str]
    ) -> List[str]:
        """Check for statistical anomalies in writing."""
        issues = []

        # Very low sentence length variance (AI tends to be uniform)
        if result.sentence_length_variance < 10 and len(sentences) > 5:
            issues.append(
                f"Unusually uniform sentence lengths (variance: {result.sentence_length_variance:.2f}) - "
                "human writing typically shows more variation"
            )

        # Very high sentence length variance (possible copy-paste from different sources)
        if result.sentence_length_variance > 500:
            issues.append(
                f"High sentence length variance ({result.sentence_length_variance:.2f}) - "
                "possible content from multiple sources"
            )

        # Unusual vocabulary richness
        if result.vocabulary_richness < 0.3 and len(sentences) > 10:
            issues.append(
                f"Low vocabulary richness ({result.vocabulary_richness:.2f}) - "
                "repetitive language pattern"
            )

        # Perfect paragraph structure (too organized)
        paragraph_lengths = [len(p.split()) for p in re.split(r'\n\n+', ' '.join(sentences)) if p.strip()]
        if len(paragraph_lengths) > 3:
            para_variance = self._calculate_variance(paragraph_lengths)
            if para_variance < 50 and all(50 < length < 150 for length in paragraph_lengths):
                issues.append(
                    "Suspiciously uniform paragraph structure"
                )

        return issues


class ComparativeStyleAnalyzer:
    """Compare document style against a reference corpus."""

    def __init__(self):
        self.analyzer = StyleAnalyzer()
        self.reference_profiles: Dict[str, StyleAnalysisResult] = {}

    def add_reference(self, name: str, text: str):
        """Add a known legitimate document as reference."""
        self.reference_profiles[name] = self.analyzer.analyze(text)

    def compare_to_references(self, text: str) -> Dict[str, float]:
        """
        Compare document to reference profiles.

        Returns dict of reference_name -> similarity_score (0-1)
        """
        if not self.reference_profiles:
            return {}

        doc_profile = self.analyzer.analyze(text)
        similarities = {}

        for name, ref_profile in self.reference_profiles.items():
            similarity = self._calculate_similarity(doc_profile, ref_profile)
            similarities[name] = similarity

        return similarities

    def _calculate_similarity(
        self,
        profile1: StyleAnalysisResult,
        profile2: StyleAnalysisResult
    ) -> float:
        """Calculate similarity between two style profiles."""
        # Simple Euclidean distance on normalized features
        features1 = [
            profile1.avg_sentence_length / 50,  # Normalize
            profile1.vocabulary_richness,
            profile1.avg_word_length / 10,
            profile1.formality_score,
        ]
        features2 = [
            profile2.avg_sentence_length / 50,
            profile2.vocabulary_richness,
            profile2.avg_word_length / 10,
            profile2.formality_score,
        ]

        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(features1, features2)))

        # Convert distance to similarity (0-1)
        similarity = 1 / (1 + distance)
        return similarity
