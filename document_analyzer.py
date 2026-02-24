"""
Medical Document Forgery Detection Pipeline
Comprehensive analysis using multiple AI detectors, metadata analysis,
consistency checking, medical entity validation, and style analysis.
"""

import os
from typing import Optional, List, Dict

from metadata_analyzer import MetadataAnalyzer
from consistency_checker import ConsistencyChecker
from medical_validator import MedicalEntityValidator
from style_analyzer import StyleAnalyzer
from ai_detectors import (
    BaseAIDetector,
    DetectorType,
    AIDetectorFactory,
    EnsembleDetector,
    GPTZeroDetector,
    HuggingFaceDetector,
    OpenAIDetector,
    OriginalityDetector,
    ZeroGPTDetector,
    CopyleaksDetector,
    BinocularsDetector,
    FastDetectGPTDetector,
    ROUGESimilarityChecker
)


class TextExtractor:
    """Extract text from various document formats."""

    @staticmethod
    def from_pdf(pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n".join(text_parts)
        except ImportError:
            raise ImportError("pdfplumber required: pip install pdfplumber")

    @staticmethod
    def from_image(image_path: str) -> str:
        """Extract text from image using OCR."""
        try:
            import pytesseract
            from PIL import Image
            image = Image.open(image_path)
            return pytesseract.image_to_string(image)
        except ImportError:
            raise ImportError("pytesseract and Pillow required: pip install pytesseract Pillow")

    @staticmethod
    def from_file(file_path: str) -> str:
        """Auto-detect file type and extract text."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return TextExtractor.from_pdf(file_path)
        elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            return TextExtractor.from_image(file_path)
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file format: {ext}")


class MedicalDocumentAnalyzer:
    """
    Complete pipeline for analyzing medical documents for forgery detection.
    Supports multiple AI detection algorithms and comprehensive document analysis.
    """

    def __init__(
        self,
        ai_detectors: Optional[List[BaseAIDetector]] = None,
        use_ensemble: bool = False,
        ensemble_weights: Optional[Dict[str, float]] = None,
        auto_detect_apis: bool = True
    ):
        """
        Initialize the analyzer with configurable AI detectors.

        Args:
            ai_detectors: List of AI detector instances to use.
                         If None and auto_detect_apis=True, will auto-configure from env vars.
            use_ensemble: If True and multiple detectors provided, use ensemble voting.
            ensemble_weights: Optional weights for ensemble voting (detector_name -> weight).
            auto_detect_apis: If True, automatically create detectors from available API keys.

        Examples:
            # Auto-detect from environment variables
            analyzer = MedicalDocumentAnalyzer()

            # Use specific detector
            from ai_detectors import HuggingFaceDetector
            analyzer = MedicalDocumentAnalyzer(
                ai_detectors=[HuggingFaceDetector()],
                auto_detect_apis=False
            )

            # Use ensemble of multiple detectors
            analyzer = MedicalDocumentAnalyzer(
                ai_detectors=[GPTZeroDetector(key), HuggingFaceDetector()],
                use_ensemble=True,
                ensemble_weights={"GPTZero": 0.6, "HuggingFace": 0.4}
            )
        """
        self.extractor = TextExtractor()
        self.metadata_analyzer = MetadataAnalyzer()
        self.consistency_checker = ConsistencyChecker()
        self.medical_validator = MedicalEntityValidator()
        self.style_analyzer = StyleAnalyzer()

        # Set up AI detectors
        if ai_detectors:
            self.ai_detectors = ai_detectors
        elif auto_detect_apis:
            self.ai_detectors = AIDetectorFactory.create_from_env()
        else:
            self.ai_detectors = []

        # Set up ensemble if requested
        self.use_ensemble = use_ensemble and len(self.ai_detectors) > 1
        self.ensemble_weights = ensemble_weights

        if self.use_ensemble:
            self.ensemble_detector = EnsembleDetector(
                self.ai_detectors,
                weights=ensemble_weights
            )

    @classmethod
    def with_gptzero(cls, api_key: str, **kwargs) -> "MedicalDocumentAnalyzer":
        """Create analyzer with GPTZero detector."""
        detector = GPTZeroDetector(api_key)
        return cls(ai_detectors=[detector], auto_detect_apis=False, **kwargs)

    @classmethod
    def with_huggingface(
        cls,
        model_name: str = "roberta-base-openai-detector",
        **kwargs
    ) -> "MedicalDocumentAnalyzer":
        """Create analyzer with local HuggingFace detector (no API key needed)."""
        detector = HuggingFaceDetector(model_name)
        return cls(ai_detectors=[detector], auto_detect_apis=False, **kwargs)

    @classmethod
    def with_originality(cls, api_key: str, **kwargs) -> "MedicalDocumentAnalyzer":
        """Create analyzer with Originality.ai detector."""
        detector = OriginalityDetector(api_key)
        return cls(ai_detectors=[detector], auto_detect_apis=False, **kwargs)

    @classmethod
    def with_zerogpt(cls, api_key: str, **kwargs) -> "MedicalDocumentAnalyzer":
        """Create analyzer with ZeroGPT detector."""
        detector = ZeroGPTDetector(api_key)
        return cls(ai_detectors=[detector], auto_detect_apis=False, **kwargs)

    @classmethod
    def with_openai_detector(cls, **kwargs) -> "MedicalDocumentAnalyzer":
        """Create analyzer with OpenAI's official RoBERTa detector (no API key needed)."""
        detector = OpenAIDetector()
        return cls(ai_detectors=[detector], auto_detect_apis=False, **kwargs)

    @classmethod
    def with_binoculars(
        cls,
        observer_model: str = "tiiuae/falcon-7b",
        performer_model: str = "tiiuae/falcon-7b-instruct",
        **kwargs
    ) -> "MedicalDocumentAnalyzer":
        """Create analyzer with Binoculars perplexity-based detector."""
        detector = BinocularsDetector(observer_model, performer_model)
        return cls(ai_detectors=[detector], auto_detect_apis=False, **kwargs)

    @classmethod
    def with_fast_detectgpt(
        cls,
        model_name: str = "gpt2-medium",
        **kwargs
    ) -> "MedicalDocumentAnalyzer":
        """Create analyzer with Fast-DetectGPT curvature-based detector."""
        detector = FastDetectGPTDetector(model_name)
        return cls(ai_detectors=[detector], auto_detect_apis=False, **kwargs)

    @classmethod
    def with_rouge_checker(
        cls,
        reference_patterns: Optional[List[str]] = None,
        threshold: float = 0.3,
        **kwargs
    ) -> "MedicalDocumentAnalyzer":
        """Create analyzer with ROUGE-based similarity checker."""
        detector = ROUGESimilarityChecker(reference_patterns, threshold)
        return cls(ai_detectors=[detector], auto_detect_apis=False, **kwargs)

    @classmethod
    def with_ensemble(
        cls,
        detectors: List[BaseAIDetector],
        weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> "MedicalDocumentAnalyzer":
        """Create analyzer with ensemble of multiple detectors."""
        return cls(
            ai_detectors=detectors,
            use_ensemble=True,
            ensemble_weights=weights,
            auto_detect_apis=False,
            **kwargs
        )

    def get_available_detectors(self) -> List[str]:
        """Return names of configured AI detectors."""
        return [d.name for d in self.ai_detectors]

    def analyze_document(
        self,
        file_path: str,
        include_ai_detection: bool = True
    ) -> dict:
        """
        Perform comprehensive analysis of a medical document.

        Args:
            file_path: Path to the document file
            include_ai_detection: Whether to include AI detection analysis

        Returns:
            Dictionary containing all analysis results
        """
        results = {
            "file": file_path,
            "text_extracted": False,
            "analyses": {},
            "all_warnings": [],
            "risk_scores": {},
            "overall_risk_level": "unknown",
            "overall_risk_score": 0.0
        }

        # Extract text
        try:
            text = self.extractor.from_file(file_path)
            results["text_extracted"] = True
            results["text_length"] = len(text)
            results["extracted_text_preview"] = text[:500] + "..." if len(text) > 500 else text
        except Exception as e:
            results["all_warnings"].append(f"Text extraction failed: {str(e)}")
            return results

        # Run all analyses
        results["analyses"]["metadata"] = self._run_metadata_analysis(file_path)
        results["analyses"]["consistency"] = self._run_consistency_analysis(text)
        results["analyses"]["medical_entities"] = self._run_medical_validation(text)
        results["analyses"]["style"] = self._run_style_analysis(text)

        if include_ai_detection and self.ai_detectors:
            results["analyses"]["ai_detection"] = self._run_ai_detection(text)

        # Aggregate results
        self._aggregate_results(results)

        return results

    def analyze_text_directly(
        self,
        text: str,
        include_ai_detection: bool = True
    ) -> dict:
        """
        Analyze text directly without file extraction.

        Args:
            text: The medical note text to analyze
            include_ai_detection: Whether to include AI detection analysis

        Returns:
            Dictionary containing all analysis results
        """
        results = {
            "text_length": len(text),
            "analyses": {},
            "all_warnings": [],
            "risk_scores": {},
            "overall_risk_level": "unknown",
            "overall_risk_score": 0.0
        }

        # Run text-based analyses (no metadata for direct text)
        results["analyses"]["consistency"] = self._run_consistency_analysis(text)
        results["analyses"]["medical_entities"] = self._run_medical_validation(text)
        results["analyses"]["style"] = self._run_style_analysis(text)

        if include_ai_detection and self.ai_detectors:
            results["analyses"]["ai_detection"] = self._run_ai_detection(text)

        # Aggregate results
        self._aggregate_results(results)

        return results

    def _run_metadata_analysis(self, file_path: str) -> dict:
        """Run metadata analysis on file."""
        result = self.metadata_analyzer.analyze_file(file_path)
        return {
            "creation_date": result.creation_date,
            "modification_date": result.modification_date,
            "author": result.author,
            "creator_tool": result.creator_tool,
            "file_size_bytes": result.file_size_bytes,
            "anomalies": result.anomalies,
            "risk_score": result.risk_score
        }

    def _run_consistency_analysis(self, text: str) -> dict:
        """Run consistency checks on text."""
        result = self.consistency_checker.check_consistency(text)
        return {
            "dates_found": result.dates_found,
            "date_inconsistencies": result.date_inconsistencies,
            "dosage_issues": result.dosage_issues,
            "formatting_issues": result.formatting_issues,
            "terminology_issues": result.terminology_issues,
            "all_issues": result.all_issues,
            "risk_score": result.risk_score
        }

    def _run_medical_validation(self, text: str) -> dict:
        """Run medical entity validation."""
        result = self.medical_validator.validate(text)
        return {
            "drugs_found": result.drugs_found,
            "invalid_drugs": result.invalid_drugs,
            "icd_codes_found": result.icd_codes_found,
            "invalid_icd_codes": result.invalid_icd_codes,
            "npi_numbers_found": result.npi_numbers_found,
            "invalid_npi_numbers": result.invalid_npi_numbers,
            "suspicious_combinations": result.suspicious_combinations,
            "all_issues": result.all_issues,
            "risk_score": result.risk_score
        }

    def _run_style_analysis(self, text: str) -> dict:
        """Run style analysis."""
        result = self.style_analyzer.analyze(text)
        return {
            "avg_sentence_length": result.avg_sentence_length,
            "sentence_length_variance": result.sentence_length_variance,
            "vocabulary_richness": result.vocabulary_richness,
            "formality_score": result.formality_score,
            "style_inconsistencies": result.style_inconsistencies,
            "statistical_anomalies": result.statistical_anomalies,
            "all_issues": result.all_issues,
            "risk_score": result.risk_score
        }

    def _run_ai_detection(self, text: str) -> dict:
        """Run AI detection using configured detectors."""
        if self.use_ensemble:
            # Use ensemble voting
            result = self.ensemble_detector.detect(text)
            return {
                "method": "ensemble",
                "detectors_used": [d.name for d in self.ai_detectors],
                "is_ai_generated": result.is_ai_generated,
                "ai_probability": result.ai_probability,
                "confidence": result.confidence,
                "individual_results": result.details.get("individual_results", []),
                "agreement_score": result.confidence,
                "error": result.error,
                "risk_score": result.ai_probability if not result.error else 0.0
            }
        elif len(self.ai_detectors) == 1:
            # Single detector
            detector = self.ai_detectors[0]
            result = detector.detect(text)
            return {
                "method": "single",
                "detector": detector.name,
                "is_ai_generated": result.is_ai_generated,
                "ai_probability": result.ai_probability,
                "confidence": result.confidence,
                "details": result.details,
                "error": result.error,
                "risk_score": result.ai_probability if not result.error else 0.0
            }
        else:
            # Multiple detectors, report all individually
            individual_results = []
            total_prob = 0.0
            successful = 0

            for detector in self.ai_detectors:
                result = detector.detect(text)
                individual_results.append({
                    "detector": detector.name,
                    "is_ai_generated": result.is_ai_generated,
                    "ai_probability": result.ai_probability,
                    "error": result.error
                })
                if not result.error:
                    total_prob += result.ai_probability
                    successful += 1

            avg_prob = total_prob / successful if successful > 0 else 0.0

            return {
                "method": "multiple",
                "detectors_used": [d.name for d in self.ai_detectors],
                "individual_results": individual_results,
                "average_probability": avg_prob,
                "is_ai_generated": avg_prob > 0.5,
                "ai_probability": avg_prob,
                "risk_score": avg_prob
            }

    def _aggregate_results(self, results: dict):
        """Aggregate all analysis results into overall risk assessment."""
        all_warnings = []
        risk_scores = {}

        for analysis_name, analysis_data in results["analyses"].items():
            if isinstance(analysis_data, dict):
                # Collect issues/anomalies
                for key in ["anomalies", "all_issues"]:
                    if key in analysis_data and analysis_data[key]:
                        all_warnings.extend(analysis_data[key])

                # Collect risk scores
                if "risk_score" in analysis_data:
                    risk_scores[analysis_name] = analysis_data["risk_score"]

        results["all_warnings"] = all_warnings
        results["risk_scores"] = risk_scores

        # Calculate weighted overall risk score
        weights = {
            "ai_detection": 0.30,
            "metadata": 0.15,
            "consistency": 0.20,
            "medical_entities": 0.20,
            "style": 0.15
        }

        total_weight = 0.0
        weighted_score = 0.0

        for analysis_name, score in risk_scores.items():
            weight = weights.get(analysis_name, 0.1)
            weighted_score += score * weight
            total_weight += weight

        if total_weight > 0:
            results["overall_risk_score"] = weighted_score / total_weight
        else:
            results["overall_risk_score"] = 0.0

        # Determine risk level
        score = results["overall_risk_score"]
        if score > 0.7:
            results["overall_risk_level"] = "high"
        elif score > 0.4:
            results["overall_risk_level"] = "medium"
        elif score > 0.0:
            results["overall_risk_level"] = "low"
        else:
            results["overall_risk_level"] = "unknown"

    def generate_report(self, results: dict) -> str:
        """
        Generate a human-readable report from analysis results.

        Args:
            results: Analysis results dictionary

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("MEDICAL DOCUMENT FORGERY DETECTION REPORT")
        lines.append("=" * 60)

        if "file" in results:
            lines.append(f"\nFile: {results['file']}")

        lines.append(f"Text Length: {results.get('text_length', 'N/A')} characters")
        lines.append(f"\nOVERALL RISK LEVEL: {results['overall_risk_level'].upper()}")
        lines.append(f"Overall Risk Score: {results['overall_risk_score']:.2f}")

        lines.append("\n" + "-" * 40)
        lines.append("INDIVIDUAL ANALYSIS SCORES")
        lines.append("-" * 40)

        for name, score in results.get("risk_scores", {}).items():
            lines.append(f"  {name}: {score:.2f}")

        if results.get("all_warnings"):
            lines.append("\n" + "-" * 40)
            lines.append("WARNINGS & ISSUES DETECTED")
            lines.append("-" * 40)
            for warning in results["all_warnings"]:
                lines.append(f"  - {warning}")

        # AI Detection section
        analyses = results.get("analyses", {})

        if "ai_detection" in analyses:
            ai = analyses["ai_detection"]
            lines.append("\n" + "-" * 40)
            lines.append("AI-GENERATED CONTENT DETECTION")
            lines.append("-" * 40)

            method = ai.get("method", "unknown")
            lines.append(f"  Method: {method}")

            if method == "ensemble":
                lines.append(f"  Detectors: {', '.join(ai.get('detectors_used', []))}")
                lines.append(f"  Agreement Score: {ai.get('agreement_score', 0):.1%}")

            if ai.get("error"):
                lines.append(f"  Error: {ai['error']}")
            else:
                lines.append(f"  AI Probability: {ai.get('ai_probability', 0):.1%}")
                lines.append(f"  Is AI Generated: {ai.get('is_ai_generated', False)}")

            # Show individual results if available
            if "individual_results" in ai and ai["individual_results"]:
                lines.append("  Individual Detector Results:")
                for ir in ai["individual_results"]:
                    detector_name = ir.get("detector", "Unknown")
                    prob = ir.get("ai_probability", 0)
                    error = ir.get("error")
                    if error:
                        lines.append(f"    - {detector_name}: Error - {error}")
                    else:
                        lines.append(f"    - {detector_name}: {prob:.1%}")

        if "medical_entities" in analyses:
            med = analyses["medical_entities"]
            lines.append("\n" + "-" * 40)
            lines.append("MEDICAL ENTITY VALIDATION")
            lines.append("-" * 40)
            lines.append(f"  Drugs Found: {', '.join(med.get('drugs_found', [])) or 'None'}")
            if med.get("suspicious_combinations"):
                lines.append("  Suspicious Combinations:")
                for combo in med["suspicious_combinations"]:
                    lines.append(f"    - {combo}")

        if "style" in analyses:
            style = analyses["style"]
            lines.append("\n" + "-" * 40)
            lines.append("WRITING STYLE ANALYSIS")
            lines.append("-" * 40)
            lines.append(f"  Avg Sentence Length: {style.get('avg_sentence_length', 0):.1f} words")
            lines.append(f"  Vocabulary Richness: {style.get('vocabulary_richness', 0):.2f}")
            lines.append(f"  Formality Score: {style.get('formality_score', 0):.2f}")

        lines.append("\n" + "=" * 60)
        lines.append("END OF REPORT")
        lines.append("=" * 60)

        return "\n".join(lines)
