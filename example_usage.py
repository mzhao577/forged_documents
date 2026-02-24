"""
Example usage of the Medical Document Forgery Detection Pipeline.

This demonstrates comprehensive document analysis including:
- Multiple AI detection algorithms (GPTZero, Originality.ai, HuggingFace, etc.)
- Ensemble voting for combined AI detection
- Metadata analysis
- Consistency checking
- Medical entity validation
- Writing style analysis

Before running:
1. Install dependencies: pip install -r requirements.txt
2. For OCR, install Tesseract: brew install tesseract (macOS)
3. For local AI detection: pip install transformers torch
4. Optionally set API keys for cloud-based detectors
"""

import os
import json
from document_analyzer import MedicalDocumentAnalyzer


def demo_with_huggingface():
    """
    Demonstrate using local HuggingFace model.
    No API key required - runs entirely locally.
    """
    print("=" * 60)
    print("DEMO: LOCAL HUGGINGFACE DETECTOR (No API Key Required)")
    print("=" * 60)

    try:
        # Create analyzer with local HuggingFace model
        analyzer = MedicalDocumentAnalyzer.with_huggingface(
            model_name="roberta-base-openai-detector"
        )

        sample_text = """
        Patient: Jane Smith
        Date: February 20, 2026

        Chief Complaint: The patient presents with persistent headache.

        History of Present Illness:
        This is a 45-year-old female experiencing headaches for 2 weeks.
        It's important to note that we should consider multiple factors.
        Furthermore, a comprehensive approach to treatment is recommended.
        Additionally, the patient reports associated symptoms of fatigue.

        Assessment: Tension headache
        Plan: Ibuprofen 400mg PRN, follow-up in 2 weeks

        Dr. John Smith, MD
        """

        print("\nAnalyzing with local HuggingFace model...")
        results = analyzer.analyze_text_directly(sample_text)
        print(analyzer.generate_report(results))

    except Exception as e:
        print(f"\nHuggingFace detector not available: {e}")
        print("Install with: pip install transformers torch")


def demo_with_gptzero():
    """Demonstrate using GPTZero API."""
    print("\n" + "=" * 60)
    print("DEMO: GPTZERO API DETECTOR")
    print("=" * 60)

    api_key = os.environ.get("GPTZERO_API_KEY")
    if not api_key:
        print("\nGPTZERO_API_KEY not set. Skipping this demo.")
        print("Get your API key from https://gptzero.me")
        return

    analyzer = MedicalDocumentAnalyzer.with_gptzero(api_key)

    sample_text = """
    Patient presents with acute bronchitis. Prescribed amoxicillin 500mg
    three times daily for 7 days. Patient advised to rest and increase
    fluid intake. Follow-up appointment scheduled for next week.
    """

    print("\nAnalyzing with GPTZero...")
    results = analyzer.analyze_text_directly(sample_text)
    print(analyzer.generate_report(results))


def demo_ensemble():
    """Demonstrate ensemble of multiple detectors."""
    print("\n" + "=" * 60)
    print("DEMO: ENSEMBLE OF MULTIPLE DETECTORS")
    print("=" * 60)

    from ai_detectors import HuggingFaceDetector, GPTZeroDetector

    detectors = []

    # Add HuggingFace (always available if transformers installed)
    try:
        detectors.append(HuggingFaceDetector())
        print("  + Added HuggingFace detector")
    except Exception:
        print("  - HuggingFace not available")

    # Add GPTZero if API key available
    gptzero_key = os.environ.get("GPTZERO_API_KEY")
    if gptzero_key:
        detectors.append(GPTZeroDetector(gptzero_key))
        print("  + Added GPTZero detector")

    if len(detectors) < 1:
        print("\nNo detectors available for ensemble demo.")
        print("Install transformers: pip install transformers torch")
        return

    # Create ensemble analyzer with custom weights
    weights = {
        "GPTZero": 0.6,
        "HuggingFace (roberta-base-openai-detector)": 0.4
    }

    analyzer = MedicalDocumentAnalyzer.with_ensemble(
        detectors=detectors,
        weights=weights
    )

    print(f"\nEnsemble configured with {len(detectors)} detector(s)")
    print(f"Available detectors: {analyzer.get_available_detectors()}")

    sample_text = """
    The patient is a 55-year-old male presenting for routine follow-up
    of hypertension. Blood pressure today is 138/85. Patient reports
    good compliance with lisinopril 10mg daily. No side effects noted.

    In conclusion, we will continue current management and recheck in
    3 months. It is worth noting that lifestyle modifications including
    diet and exercise should be encouraged. Furthermore, the patient
    should monitor blood pressure at home.
    """

    print("\nAnalyzing with ensemble voting...")
    results = analyzer.analyze_text_directly(sample_text)
    print(analyzer.generate_report(results))


def demo_all_detectors():
    """Show how to use each detector type."""
    print("\n" + "=" * 60)
    print("AVAILABLE AI DETECTOR OPTIONS")
    print("=" * 60)

    print("""
    1. HuggingFace (Local - No API Key Required)
       ----------------------------------------
       from document_analyzer import MedicalDocumentAnalyzer

       # Default model
       analyzer = MedicalDocumentAnalyzer.with_huggingface()

       # Or specify a different model
       analyzer = MedicalDocumentAnalyzer.with_huggingface(
           model_name="Hello-SimpleAI/chatgpt-detector-roberta"
       )


    2. GPTZero (API-based)
       -------------------
       export GPTZERO_API_KEY="your_key"

       analyzer = MedicalDocumentAnalyzer.with_gptzero(api_key)
       # Or auto-detect from environment:
       analyzer = MedicalDocumentAnalyzer()  # reads GPTZERO_API_KEY


    3. Originality.ai (API-based)
       --------------------------
       export ORIGINALITY_API_KEY="your_key"

       analyzer = MedicalDocumentAnalyzer.with_originality(api_key)


    4. ZeroGPT (API-based)
       -------------------
       export ZEROGPT_API_KEY="your_key"

       analyzer = MedicalDocumentAnalyzer.with_zerogpt(api_key)


    5. Copyleaks (API-based)
       ---------------------
       export COPYLEAKS_EMAIL="your_email"
       export COPYLEAKS_API_KEY="your_key"

       from ai_detectors import CopyleaksDetector
       detector = CopyleaksDetector(email, api_key)
       analyzer = MedicalDocumentAnalyzer(ai_detectors=[detector])


    6. Ensemble (Multiple Detectors Combined)
       --------------------------------------
       from ai_detectors import GPTZeroDetector, HuggingFaceDetector

       detectors = [
           GPTZeroDetector(gptzero_key),
           HuggingFaceDetector(),
       ]

       # With custom weights (higher = more influence)
       weights = {
           "GPTZero": 0.6,
           "HuggingFace (roberta-base-openai-detector)": 0.4
       }

       analyzer = MedicalDocumentAnalyzer.with_ensemble(
           detectors=detectors,
           weights=weights
       )


    7. Auto-detect from Environment
       -----------------------------
       # Automatically uses all available API keys from environment
       analyzer = MedicalDocumentAnalyzer(auto_detect_apis=True)

       # Check what was configured:
       print(analyzer.get_available_detectors())
    """)


def demo_comprehensive_analysis():
    """Demonstrate full analysis with available detectors."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE ANALYSIS DEMO")
    print("=" * 60)

    # Auto-detect available detectors from environment
    analyzer = MedicalDocumentAnalyzer(auto_detect_apis=True)

    available = analyzer.get_available_detectors()
    if available:
        print(f"\nAuto-detected detectors: {available}")
    else:
        print("\nNo AI detectors available. Running without AI detection.")
        print("Set API keys or install transformers for AI detection.")

    # Sample medical note with various issues
    sample_note = """
    Patient: Jane Smith
    Date: February 30, 2026
    DOB: 03/15/1985

    Chief Complaint: Patient presents with severe headache and fatigue.

    History of Present Illness:
    The paitent is a 40-year-old female who reports experiencing persistent
    headaches for the past 2 weeks. Associated symptons include photophobia
    and nausea.

    Current Medications:
    - Ibuprofen 2000mg PRN (patient reports taking this dose)
    - Sertraline 100mg daily
    - Tramadol 50mg PRN for pain

    Assessment:
    Migraine headache with aura. It's important to note that we should
    consider multiple factors. Furthermore, a comprehensive approach is needed.

    Plan:
    1. Continue current medications
    2. Follow-up appointment: March 35, 2026

    Dr. John Williams, MD
    NPI: 1234567890
    """

    print("\nAnalyzing sample medical note...")
    results = analyzer.analyze_text_directly(sample_note)
    print(analyzer.generate_report(results))

    # Print JSON for programmatic access
    print("\n" + "-" * 40)
    print("AI DETECTION DETAILS (JSON)")
    print("-" * 40)
    if "ai_detection" in results.get("analyses", {}):
        print(json.dumps(results["analyses"]["ai_detection"], indent=2, default=str))
    else:
        print("No AI detection results available.")


def demo_compare_detectors():
    """Compare results from different detectors on same text."""
    print("\n" + "=" * 60)
    print("COMPARING DETECTOR RESULTS")
    print("=" * 60)

    from ai_detectors import HuggingFaceDetector, GPTZeroDetector

    # AI-like text sample
    ai_text = """
    In conclusion, it is worth noting that the patient's condition has shown
    significant improvement. Furthermore, the comprehensive treatment approach
    has yielded positive results. Additionally, we recommend continuing the
    current regimen while monitoring for any adverse effects. It's important
    to note that lifestyle modifications should be encouraged alongside
    pharmaceutical interventions.
    """

    # Human-like text sample
    human_text = """
    Pt came in today, looks much better than last week. BP down to 130/80
    finally! Told him to keep taking the lisinopril and watch the salt.
    He's been walking more which helps. Will see him back in 3 months
    unless something comes up before then. - Dr. M
    """

    detectors_to_test = []

    # Try to add HuggingFace
    try:
        detectors_to_test.append(("HuggingFace", HuggingFaceDetector()))
    except Exception:
        pass

    # Try to add GPTZero
    gptzero_key = os.environ.get("GPTZERO_API_KEY")
    if gptzero_key:
        detectors_to_test.append(("GPTZero", GPTZeroDetector(gptzero_key)))

    if not detectors_to_test:
        print("\nNo detectors available for comparison.")
        print("Install transformers or set GPTZERO_API_KEY")
        return

    print(f"\nTesting {len(detectors_to_test)} detector(s)...")

    for text_name, text in [("AI-Like Text", ai_text), ("Human-Like Text", human_text)]:
        print(f"\n--- {text_name} ---")
        for detector_name, detector in detectors_to_test:
            result = detector.detect(text)
            if result.error:
                print(f"  {detector_name}: Error - {result.error}")
            else:
                status = "AI" if result.is_ai_generated else "Human"
                print(f"  {detector_name}: {result.ai_probability:.1%} ({status})")


def main():
    """Run all demonstrations."""
    # Show available options
    demo_all_detectors()

    # Run demos
    demo_with_huggingface()
    demo_with_gptzero()
    demo_ensemble()
    demo_comprehensive_analysis()
    demo_compare_detectors()

    print("\n" + "=" * 60)
    print("ENVIRONMENT VARIABLES FOR API-BASED DETECTORS")
    print("=" * 60)
    print("""
    Set these environment variables to enable API-based detectors:

    export GPTZERO_API_KEY="your_key"        # https://gptzero.me
    export ORIGINALITY_API_KEY="your_key"    # https://originality.ai
    export ZEROGPT_API_KEY="your_key"        # https://zerogpt.com
    export COPYLEAKS_EMAIL="your_email"      # https://copyleaks.com
    export COPYLEAKS_API_KEY="your_key"

    For local detection (no API key needed):
    pip install transformers torch
    """)


if __name__ == "__main__":
    main()
