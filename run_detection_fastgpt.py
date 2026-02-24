"""
Run forgery detection using Fast-DetectGPT algorithm.

Fast-DetectGPT uses curvature-based detection without sampling.
No API key required - runs locally with GPT-2 model.

Requirements:
- transformers
- torch
"""

import argparse
from pathlib import Path
from document_analyzer import MedicalDocumentAnalyzer
from ai_detectors import FastDetectGPTDetector, HuggingFaceDetector, EnsembleDetector

BASE_DIR = Path(__file__).parent
NOTE_DATA_DIR = BASE_DIR / "note_data"
CMS_NOTES_DIR = NOTE_DATA_DIR / "cms_notes"


def run_with_fastdetectgpt(folder: str = "all", limit: int = None, use_ensemble: bool = False):
    """Run detection using Fast-DetectGPT."""

    print("=" * 70)
    print("FAST-DETECTGPT MEDICAL DOCUMENT ANALYSIS")
    print("=" * 70)

    # Create detectors
    print("\nLoading AI detectors...")

    detectors = []

    # Add Fast-DetectGPT
    try:
        fast_detector = FastDetectGPTDetector(model_name='gpt2', threshold=15.0)
        detectors.append(fast_detector)
        print("  + Fast-DetectGPT (GPT-2 curvature-based)")
    except Exception as e:
        print(f"  - Fast-DetectGPT failed: {e}")

    # Optionally add HuggingFace for ensemble
    if use_ensemble:
        try:
            hf_detector = HuggingFaceDetector()
            detectors.append(hf_detector)
            print("  + HuggingFace RoBERTa detector")
        except Exception as e:
            print(f"  - HuggingFace failed: {e}")

    if not detectors:
        print("No detectors available!")
        return

    # Create analyzer
    if len(detectors) > 1:
        ensemble = EnsembleDetector(detectors)
        analyzer = MedicalDocumentAnalyzer(ai_detectors=[ensemble])
        print(f"\nUsing ensemble of {len(detectors)} detectors")
    else:
        analyzer = MedicalDocumentAnalyzer(ai_detectors=detectors)
        print(f"\nUsing {detectors[0].name}")

    print("=" * 70)

    # Get test files
    test_files = []
    if folder in ["generated", "all"]:
        test_files.extend(sorted(NOTE_DATA_DIR.glob("*.txt")))
    if folder in ["cms", "all"]:
        if CMS_NOTES_DIR.exists():
            test_files.extend(sorted(CMS_NOTES_DIR.glob("*.txt")))

    if limit:
        test_files = test_files[:limit]

    print(f"\nProcessing {len(test_files)} files...\n")

    results = []

    for filepath in test_files:
        try:
            result = analyzer.analyze_document(str(filepath))

            ai_info = result.get("analyses", {}).get("ai_detection", {})
            ai_prob = ai_info.get("ai_probability", 0) if isinstance(ai_info, dict) else 0

            warnings = result.get("all_warnings", [])
            significant = [w for w in warnings if "Metadata" not in w]

            risk_score = result.get("overall_risk_score", 0)

            results.append({
                "file": filepath.name,
                "risk_score": risk_score,
                "ai_probability": ai_prob,
                "warnings": significant,
                "details": ai_info.get("details", {}) if isinstance(ai_info, dict) else {}
            })

            # Print progress
            flag = ""
            if risk_score >= 0.3:
                flag = " [!]"
            elif ai_prob >= 0.5:
                flag = " [AI]"
            print(f"  {filepath.name}: risk={risk_score:.2f}, AI={ai_prob:.0%}{flag}")

        except Exception as e:
            print(f"  {filepath.name}: ERROR - {e}")
            results.append({
                "file": filepath.name,
                "risk_score": None,
                "ai_probability": 0,
                "warnings": [str(e)],
                "details": {}
            })

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    flagged = [r for r in results if r["risk_score"] and
               (r["risk_score"] >= 0.2 or r["ai_probability"] >= 0.5)]

    if flagged:
        print(f"\n{len(flagged)} file(s) flagged:\n")
        for r in sorted(flagged, key=lambda x: x["risk_score"] or 0, reverse=True):
            print(f"  {r['file']}")
            print(f"    Risk: {r['risk_score']:.2f}, AI: {r['ai_probability']:.0%}")
            if r["warnings"]:
                for w in r["warnings"][:3]:
                    print(f"    - {w}")
            if r["details"]:
                curv = r["details"].get("curvature")
                if curv:
                    print(f"    - Curvature: {curv:.2f}")
            print()
    else:
        print("\nNo files flagged.")

    print(f"Total files: {len(results)}")
    print(f"Flagged: {len(flagged)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fast-DetectGPT detection")
    parser.add_argument("--folder", choices=["generated", "cms", "all"], default="generated",
                        help="Which folder to scan")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max files to process")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use ensemble with HuggingFace detector")

    args = parser.parse_args()
    run_with_fastdetectgpt(folder=args.folder, limit=args.limit, use_ensemble=args.ensemble)
