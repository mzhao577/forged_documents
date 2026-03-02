"""
Run forgery detection on all test files in note_data folder.

Scans:
- note_data/*.txt (generated test data)
- note_data/cms_notes/*.txt (CMS-derived medical notes)

Outputs results to a CSV file.
"""

import os
import csv
import argparse
from pathlib import Path
from datetime import datetime
from document_analyzer import MedicalDocumentAnalyzer
from detect_ai_detectors import (
    HuggingFaceDetector, FastDetectGPTDetector, LLMDetDetector,
    BinocularsDetector, EnsembleDetector
)

BASE_DIR = Path(__file__).parent
NOTE_DATA_DIR = BASE_DIR / "note_data"
CMS_NOTES_DIR = NOTE_DATA_DIR / "cms_notes"

# Risk thresholds
HIGH_RISK_THRESHOLD = 0.6
MEDIUM_RISK_THRESHOLD = 0.3
FLAGGED_THRESHOLD = 0.2  # Show reasons for files above this threshold


def classify_risk(score: float) -> str:
    """Classify risk level based on score."""
    if score >= HIGH_RISK_THRESHOLD:
        return "HIGH"
    elif score >= MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"


def run_all_detections(folder: str = "all", limit: int = None, flawed_only: bool = False,
                       detector: str = "huggingface", output_file: str = None):
    """Run detection on all test files.

    Args:
        folder: Which folder to scan - "generated", "cms", or "all"
        limit: Maximum number of files to process (None = all)
        flawed_only: Only process files with 'flawed' in the name
        detector: Which AI detector to use - "huggingface", "llmdet", "fast-detectgpt", or "ensemble"
        output_file: Path to output CSV file (default: detection_results_YYYYMMDD_HHMMSS.csv)
    """
    # Set default output filename with timestamp
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"detection_results_{timestamp}.csv"

    # Create the requested AI detector
    print("Initializing Medical Document Analyzer...")
    print("=" * 70)

    detectors = []

    if detector == "llmdet":
        try:
            llmdet_detector = LLMDetDetector()
            detectors.append(llmdet_detector)
            print("Using LLMDet (proxy perplexity with LLM source identification)")
        except Exception as e:
            print(f"LLMDet not available ({e})")

    elif detector == "fast-detectgpt":
        try:
            fastgpt_detector = FastDetectGPTDetector(model_name='gpt2', threshold=15.0)
            detectors.append(fastgpt_detector)
            print("Using Fast-DetectGPT (curvature-based detection)")
        except Exception as e:
            print(f"Fast-DetectGPT not available ({e})")

    elif detector == "binoculars":
        try:
            binoculars_detector = BinocularsDetector(
                observer_model="tiiuae/falcon-7b",
                performer_model="tiiuae/falcon-7b-instruct",
                threshold=0.9
            )
            detectors.append(binoculars_detector)
            print("Using Binoculars (perplexity comparison - requires ~28GB RAM)")
            print("  Observer model: tiiuae/falcon-7b")
            print("  Performer model: tiiuae/falcon-7b-instruct")
        except Exception as e:
            print(f"Binoculars not available ({e})")

    elif detector == "ensemble":
        # Create ensemble with multiple detectors
        ensemble_detectors = []
        try:
            ensemble_detectors.append(HuggingFaceDetector())
            print("  + HuggingFace RoBERTa")
        except Exception as e:
            print(f"  - HuggingFace failed: {e}")

        try:
            ensemble_detectors.append(LLMDetDetector())
            print("  + LLMDet")
        except Exception as e:
            print(f"  - LLMDet failed: {e}")

        try:
            ensemble_detectors.append(FastDetectGPTDetector(model_name='gpt2', threshold=15.0))
            print("  + Fast-DetectGPT")
        except Exception as e:
            print(f"  - Fast-DetectGPT failed: {e}")

        if ensemble_detectors:
            detectors.append(EnsembleDetector(ensemble_detectors))
            print(f"Using Ensemble of {len(ensemble_detectors)} detectors")

    else:  # default: huggingface
        try:
            hf_detector = HuggingFaceDetector()
            detectors.append(hf_detector)
            print("Using HuggingFace local AI detector")
        except Exception as e:
            print(f"HuggingFace not available ({e})")

    if detectors:
        analyzer = MedicalDocumentAnalyzer(ai_detectors=detectors)
    else:
        print("No AI detectors available - will still check consistency, medical validation, etc.")
        analyzer = MedicalDocumentAnalyzer(ai_detectors=[])

    print("=" * 70)
    print()

    # Get test files based on folder selection
    test_files = []
    if folder in ["generated", "all"]:
        test_files.extend(sorted(NOTE_DATA_DIR.glob("*.txt")))
    if folder in ["cms", "all"]:
        if CMS_NOTES_DIR.exists():
            test_files.extend(sorted(CMS_NOTES_DIR.glob("*.txt")))

    # Filter for flawed files only if requested
    if flawed_only:
        test_files = [f for f in test_files if "flawed" in f.name.lower()]

    if not test_files:
        print("No test files found")
        return

    # Apply limit if specified
    if limit:
        test_files = test_files[:limit]

    print(f"Found {len(test_files)} test files to process\n")
    print(f"Results will be saved to: {output_file}\n")

    # Process each file
    results_summary = []
    csv_rows = []

    for filepath in test_files:
        print("\n" + "=" * 70)
        print(f"FILE: {filepath.name}")
        print("=" * 70)

        try:
            # Analyze the document
            results = analyzer.analyze_document(str(filepath))

            # Generate and print report
            report = analyzer.generate_report(results)
            print(report)

            # Get all analyses
            analyses = results.get("analyses", {})

            # Get AI detection info
            ai_info = analyses.get("ai_detection", {})
            ai_probability = ai_info.get("ai_probability", 0) if isinstance(ai_info, dict) else 0

            # Get individual component scores and determine which contributed
            contributing_components = []

            # Check AI Detection
            if ai_probability > 0.2:
                contributing_components.append("AI_Detection")

            # Check Consistency
            consistency_info = analyses.get("consistency", {})
            consistency_score = consistency_info.get("risk_score", 0) if isinstance(consistency_info, dict) else 0
            if consistency_score > 0:
                contributing_components.append("Consistency")

            # Check Style
            style_info = analyses.get("style", {})
            style_score = style_info.get("risk_score", 0) if isinstance(style_info, dict) else 0
            if style_score > 0:
                contributing_components.append("Style")

            # Check Medical Entities
            medical_info = analyses.get("medical_entities", {})
            medical_score = medical_info.get("risk_score", 0) if isinstance(medical_info, dict) else 0
            if medical_score > 0:
                contributing_components.append("Medical_Entities")

            # Check Metadata
            metadata_info = analyses.get("metadata", {})
            metadata_score = metadata_info.get("risk_score", 0) if isinstance(metadata_info, dict) else 0
            if metadata_score > 0:
                contributing_components.append("Metadata")

            # Get warnings (filter out metadata warning for .txt files)
            # Warnings are stored under "all_warnings" in the results
            warnings = results.get("all_warnings", [])
            significant_warnings = [
                w for w in warnings
                if "Metadata analysis not supported" not in w
                and "Excessive irregular spacing" not in w  # Common in generated notes
            ]

            # Calculate risk score and level
            risk_score = results.get("overall_risk_score", 0)
            risk_level = classify_risk(risk_score)

            # Classify as AIText or human_created based on probability
            classification = "AIText" if ai_probability > 0.2 else "human_created"

            # Build explanation
            explanation_parts = []
            if ai_probability > 0.2:
                explanation_parts.append(f"High AI probability ({ai_probability:.1%})")
            else:
                explanation_parts.append(f"Low AI probability ({ai_probability:.1%})")
            if significant_warnings:
                explanation_parts.extend(significant_warnings[:3])  # Include top 3 warnings
            explanation = "; ".join(explanation_parts) if explanation_parts else "No issues detected"

            # Format contributing components
            components_str = ", ".join(contributing_components) if contributing_components else "None"

            # Determine if document failed any validation check (non-AI checks)
            failed_validation = "Yes" if (consistency_score > 0 or style_score > 0 or
                                          medical_score > 0 or metadata_score > 0) else "No"

            # Determine if any abnormality was detected (AI-Classification is AIText OR Failed_Validation is Yes)
            abnormal_detected = "Yes" if (classification == "AIText" or failed_validation == "Yes") else "No"

            # Add row for CSV
            csv_rows.append({
                "file_name": filepath.name,
                "Abnormal_detected": abnormal_detected,
                "AI-Classification": classification,
                "ai_probability": round(ai_probability, 4),
                "Failed_Validation": failed_validation,
                "explanation": explanation,
                "contributing_components": components_str
            })

            # Store summary with detailed info
            results_summary.append({
                "file": filepath.name,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "ai_probability": ai_probability,
                "warnings": significant_warnings,
                "all_warnings": warnings,
                "issues_found": len(significant_warnings)
            })

        except Exception as e:
            print(f"Error analyzing {filepath.name}: {e}")
            csv_rows.append({
                "file_name": filepath.name,
                "Abnormal_detected": "Yes",
                "AI-Classification": "ERROR",
                "ai_probability": 0,
                "Failed_Validation": "ERROR",
                "explanation": f"Error during analysis: {str(e)}",
                "contributing_components": "ERROR"
            })
            results_summary.append({
                "file": filepath.name,
                "risk_score": None,
                "risk_level": "ERROR",
                "ai_probability": 0,
                "warnings": [str(e)],
                "all_warnings": [str(e)],
                "issues_found": 1
            })

    # Write results to CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_name', 'Abnormal_detected', 'AI-Classification', 'ai_probability', 'Failed_Validation', 'explanation', 'contributing_components']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n*** Results saved to: {output_file} ***")

    # Print final summary
    print("\n" + "=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)
    print(f"\n{'File':<40} {'Risk':<8} {'AI %':<8} {'Issues'}")
    print("-" * 70)

    for r in results_summary:
        score = f"{r['risk_score']:.2f}" if r['risk_score'] is not None else "N/A"
        ai_pct = f"{r['ai_probability']:.0%}" if r['ai_probability'] else "0%"
        level_indicator = ""
        if r['risk_level'] == "HIGH":
            level_indicator = " [!!!]"
        elif r['risk_level'] == "MEDIUM":
            level_indicator = " [!]"

        print(f"{r['file']:<40} {score:<8} {ai_pct:<8} {r['issues_found']}{level_indicator}")

    # Print detailed flagged files section
    print("\n" + "=" * 70)
    print("FLAGGED FILES - DETAILED REASONS")
    print("=" * 70)

    flagged_files = [r for r in results_summary
                     if r['risk_score'] is not None and
                     (r['risk_score'] >= FLAGGED_THRESHOLD or r['ai_probability'] >= 0.5)]

    if not flagged_files:
        print("\nNo files flagged for review.")
    else:
        print(f"\n{len(flagged_files)} file(s) flagged for review:\n")

        for r in sorted(flagged_files, key=lambda x: x['risk_score'] or 0, reverse=True):
            print(f"  {r['file']}")
            print(f"  Risk Score: {r['risk_score']:.2f} ({r['risk_level']})")

            if r['ai_probability'] >= 0.5:
                print(f"  AI Detection: {r['ai_probability']:.1%} probability of AI-generated content")

            if r['warnings']:
                print("  Reasons:")
                for warning in r['warnings']:
                    print(f"    - {warning}")
            elif r['ai_probability'] >= 0.5:
                print("  Reasons:")
                print(f"    - High AI-generated content probability ({r['ai_probability']:.1%})")

            print()

    # Group by risk level
    print("-" * 70)
    high_risk = [r for r in results_summary if r['risk_level'] == 'HIGH']
    medium_risk = [r for r in results_summary if r['risk_level'] == 'MEDIUM']
    low_risk = [r for r in results_summary if r['risk_level'] == 'LOW']

    print(f"\nRISK SUMMARY:")
    print(f"  HIGH RISK:   {len(high_risk)} files (score >= {HIGH_RISK_THRESHOLD})")
    print(f"  MEDIUM RISK: {len(medium_risk)} files (score >= {MEDIUM_RISK_THRESHOLD})")
    print(f"  LOW RISK:    {len(low_risk)} files (score < {MEDIUM_RISK_THRESHOLD})")

    if high_risk:
        print(f"\nHIGH RISK FILES:")
        for r in high_risk:
            print(f"  - {r['file']} (score: {r['risk_score']:.2f})")
            for w in r['warnings'][:3]:  # Show top 3 warnings
                print(f"      * {w}")

    if medium_risk:
        print(f"\nMEDIUM RISK FILES:")
        for r in medium_risk:
            print(f"  - {r['file']} (score: {r['risk_score']:.2f})")
            for w in r['warnings'][:2]:  # Show top 2 warnings
                print(f"      * {w}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run forgery detection on test files")
    parser.add_argument("--folder", choices=["generated", "cms", "all"], default="all",
                        help="Which folder to scan (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of files to process")
    parser.add_argument("--flawed-only", action="store_true",
                        help="Only process files with 'flawed' in the name")
    parser.add_argument("--detector", choices=["huggingface", "llmdet", "fast-detectgpt", "binoculars", "ensemble"],
                        default="huggingface",
                        help="AI detector to use (default: huggingface). Note: binoculars requires ~28GB RAM")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file path (default: detection_results_YYYYMMDD_HHMMSS.csv)")

    args = parser.parse_args()

    run_all_detections(folder=args.folder, limit=args.limit, flawed_only=args.flawed_only,
                       detector=args.detector, output_file=args.output)
