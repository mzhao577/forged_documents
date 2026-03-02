"""
AI Text Detection using HuggingFace RoBERTa Models

This script detects AI-generated text in medical notes using multiple
RoBERTa-based models from HuggingFace.

Supported models:
- openai-community/roberta-base-openai-detector (OpenAI's original detector)
- Hello-SimpleAI/chatgpt-detector-roberta (ChatGPT-specific detector)
- roberta-base-openai-detector (alternative name for OpenAI detector)

IMPORTANT: Models run locally only. No data is sent to remote servers.
- Models must be downloaded once before first use
- Run: python roberta_ai_detector.py --download-models
- Models are cached at: ~/.cache/huggingface/hub/
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime

import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class DetectionResult:
    """Result from AI detection analysis."""
    filename: str
    model_name: str
    is_ai_generated: bool
    ai_probability: float
    human_probability: float
    confidence: float
    raw_label: str
    raw_score: float
    text_length: int
    error: Optional[str] = None


class RoBERTaAIDetector:
    """
    AI text detector using HuggingFace RoBERTa models.
    """

    # Available models for AI detection
    AVAILABLE_MODELS = {
        "openai": "openai-community/roberta-base-openai-detector",
        "chatgpt": "Hello-SimpleAI/chatgpt-detector-roberta",
        "roberta-base": "roberta-base-openai-detector",
    }

    def __init__(self, model_name: str = "openai", device: Optional[str] = None):
        """
        Initialize the detector with a specific model.

        Args:
            model_name: Model key from AVAILABLE_MODELS or full HuggingFace model path
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
        """
        # Resolve model name
        if model_name in self.AVAILABLE_MODELS:
            self.model_path = self.AVAILABLE_MODELS[model_name]
        else:
            self.model_path = model_name

        self.model_name = model_name

        # Auto-detect device
        # Note: MPS (Apple Silicon) has compatibility issues with some models,
        # so we default to CPU on macOS unless CUDA is available
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                # MPS often has issues with transformers pipelines, use CPU
                self.device = "cpu"
        else:
            self.device = device

        self._pipeline = None
        self._load_error = None

    def _load_model(self):
        """Load the model pipeline."""
        if self._pipeline is not None or self._load_error is not None:
            return

        try:
            print(f"Loading model: {self.model_path}")
            print(f"Using device: {self.device}")

            self._pipeline = pipeline(
                "text-classification",
                model=self.model_path,
                tokenizer=self.model_path,
                device=self.device if self.device != "mps" else -1,  # MPS needs special handling
                truncation=True,
                max_length=512,
                local_files_only=True  # Always use locally cached models, never download
            )

            # For MPS, we need to move the model manually
            if self.device == "mps":
                self._pipeline.model = self._pipeline.model.to("mps")

            print(f"Model loaded successfully!")

        except Exception as e:
            error_msg = str(e)
            if "local_files_only" in error_msg or "not found" in error_msg.lower():
                self._load_error = (
                    f"Model '{self.model_path}' not found in local cache. "
                    f"Run 'python roberta_ai_detector.py --download-models' to download models first."
                )
            else:
                self._load_error = f"Failed to load model: {error_msg}"
            print(f"Error: {self._load_error}")

    def detect(self, text: str, filename: str = "unknown") -> DetectionResult:
        """
        Analyze text for AI-generated content.

        Args:
            text: The text to analyze
            filename: Name of the source file (for reporting)

        Returns:
            DetectionResult with analysis details
        """
        # Validate input
        if not text or len(text.strip()) < 50:
            return DetectionResult(
                filename=filename,
                model_name=self.model_name,
                is_ai_generated=False,
                ai_probability=0.0,
                human_probability=1.0,
                confidence=0.0,
                raw_label="",
                raw_score=0.0,
                text_length=len(text) if text else 0,
                error="Text too short (minimum 50 characters required)"
            )

        # Load model if needed
        self._load_model()

        if self._load_error:
            return DetectionResult(
                filename=filename,
                model_name=self.model_name,
                is_ai_generated=False,
                ai_probability=0.0,
                human_probability=1.0,
                confidence=0.0,
                raw_label="",
                raw_score=0.0,
                text_length=len(text),
                error=self._load_error
            )

        try:
            # Truncate very long texts (models typically handle 512 tokens)
            truncated_text = text[:4000] if len(text) > 4000 else text

            # Run detection
            result = self._pipeline(truncated_text)[0]
            label = result["label"].lower()
            score = result["score"]

            # Interpret the label
            # Different models use different labels:
            # - OpenAI detector: "LABEL_0" (human) / "LABEL_1" (AI) or "Real" / "Fake"
            # - ChatGPT detector: "Human" / "ChatGPT"
            if "fake" in label or "ai" in label or "chatgpt" in label or "1" in label or "generated" in label:
                ai_prob = score
                human_prob = 1 - score
            elif "real" in label or "human" in label or "0" in label:
                ai_prob = 1 - score
                human_prob = score
            else:
                # Unknown label format - assume score represents AI probability
                ai_prob = score
                human_prob = 1 - score

            # Calculate confidence (how far from 0.5)
            confidence = abs(ai_prob - 0.5) * 2

            return DetectionResult(
                filename=filename,
                model_name=self.model_name,
                is_ai_generated=ai_prob > 0.5,
                ai_probability=round(ai_prob, 4),
                human_probability=round(human_prob, 4),
                confidence=round(confidence, 4),
                raw_label=result["label"],
                raw_score=round(result["score"], 4),
                text_length=len(text)
            )

        except Exception as e:
            return DetectionResult(
                filename=filename,
                model_name=self.model_name,
                is_ai_generated=False,
                ai_probability=0.0,
                human_probability=1.0,
                confidence=0.0,
                raw_label="",
                raw_score=0.0,
                text_length=len(text),
                error=f"Detection failed: {str(e)}"
            )


class MultiModelDetector:
    """
    Runs detection using multiple RoBERTa models and aggregates results.
    """

    def __init__(self, models: List[str] = None, device: Optional[str] = None):
        """
        Initialize with multiple models.

        Args:
            models: List of model names to use (defaults to all available)
            device: Device to use for all models
        """
        if models is None:
            models = list(RoBERTaAIDetector.AVAILABLE_MODELS.keys())

        self.detectors = []
        for model_name in models:
            self.detectors.append(RoBERTaAIDetector(model_name, device))

    def detect(self, text: str, filename: str = "unknown") -> Dict:
        """
        Run detection with all models.

        Returns:
            Dictionary with individual results and aggregate scores
        """
        results = []
        ai_probs = []

        for detector in self.detectors:
            result = detector.detect(text, filename)
            results.append(result)
            if result.error is None:
                ai_probs.append(result.ai_probability)

        # Calculate aggregate scores
        if ai_probs:
            avg_ai_prob = sum(ai_probs) / len(ai_probs)
            consensus = sum(1 for r in results if r.is_ai_generated and r.error is None)
            total_valid = sum(1 for r in results if r.error is None)
        else:
            avg_ai_prob = 0.0
            consensus = 0
            total_valid = 0

        return {
            "filename": filename,
            "individual_results": [asdict(r) for r in results],
            "aggregate": {
                "average_ai_probability": round(avg_ai_prob, 4),
                "consensus_ai_count": consensus,
                "total_models": total_valid,
                "majority_ai": consensus > total_valid / 2 if total_valid > 0 else False
            }
        }


def download_models(models: List[str] = None):
    """
    Download models from HuggingFace Hub to local cache.

    Args:
        models: List of model keys to download (defaults to all)
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if models is None:
        models = list(RoBERTaAIDetector.AVAILABLE_MODELS.keys())

    print("Downloading models to local cache...")
    print("=" * 60)

    for model_key in models:
        if model_key in RoBERTaAIDetector.AVAILABLE_MODELS:
            model_path = RoBERTaAIDetector.AVAILABLE_MODELS[model_key]
        else:
            model_path = model_key

        print(f"\nDownloading: {model_path}")
        try:
            # Download tokenizer and model
            AutoTokenizer.from_pretrained(model_path)
            AutoModelForSequenceClassification.from_pretrained(model_path)
            print(f"  ✓ Successfully downloaded and cached")
        except Exception as e:
            print(f"  ✗ Failed to download: {e}")

    print("\n" + "=" * 60)
    print("Download complete. Models are now cached locally.")
    print(f"Cache location: ~/.cache/huggingface/hub/")


def load_notes_from_folder(folder_path: str) -> List[Dict]:
    """
    Load all text files from a folder.

    Args:
        folder_path: Path to the folder containing note files

    Returns:
        List of dictionaries with 'filename' and 'text' keys
    """
    notes = []
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return notes

    # Get all text files
    txt_files = sorted(folder.glob("*.txt"))

    if not txt_files:
        print(f"Warning: No .txt files found in '{folder_path}'")
        return notes

    print(f"Found {len(txt_files)} text files in '{folder_path}'")

    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
            notes.append({
                "filename": txt_file.name,
                "text": text
            })
        except Exception as e:
            print(f"Error reading {txt_file.name}: {e}")

    return notes


def generate_explanation(result: Dict, use_multi: bool) -> str:
    """
    Generate a human-readable explanation for why text was classified as AI or human.

    Args:
        result: Detection result dictionary
        use_multi: Whether multi-model detection was used

    Returns:
        Explanation string
    """
    explanations = []

    if use_multi:
        agg = result["aggregate"]
        individual = result["individual_results"]

        # Model consensus explanation
        total = agg["total_models"]
        ai_votes = agg["consensus_ai_count"]
        human_votes = total - ai_votes

        if agg["majority_ai"]:
            explanations.append(f"Majority vote: {ai_votes}/{total} models classified as AI-generated")
        else:
            explanations.append(f"Majority vote: {human_votes}/{total} models classified as human-written")

        # Individual model details
        model_details = []
        for ind in individual:
            if ind.get("error"):
                continue
            model_name = ind["model_name"]
            ai_prob = ind["ai_probability"]
            verdict = "AI" if ind["is_ai_generated"] else "Human"
            model_details.append(f"{model_name}: {ai_prob:.1%} ({verdict})")

        if model_details:
            explanations.append("Model scores: " + "; ".join(model_details))

        # Confidence assessment
        avg_prob = agg["average_ai_probability"]
        if avg_prob > 0.8:
            explanations.append("High confidence AI detection (>80% avg probability)")
        elif avg_prob > 0.6:
            explanations.append("Moderate confidence AI detection (60-80% avg probability)")
        elif avg_prob > 0.4:
            explanations.append("Borderline case - models show mixed signals (40-60% avg probability)")
        elif avg_prob > 0.2:
            explanations.append("Likely human-written with some AI-like patterns (20-40% avg probability)")
        else:
            explanations.append("High confidence human-written (<20% avg probability)")
    else:
        # Single model result
        if result.get("error"):
            return f"Error: {result['error']}"

        ai_prob = result["ai_probability"]
        raw_label = result["raw_label"]
        confidence = result["confidence"]

        explanations.append(f"Model raw output: {raw_label} (score: {result['raw_score']:.4f})")
        explanations.append(f"Confidence level: {confidence:.1%}")

        if ai_prob > 0.8:
            explanations.append("Strong AI-generated text patterns detected")
        elif ai_prob > 0.6:
            explanations.append("Moderate AI-generated text indicators")
        elif ai_prob > 0.4:
            explanations.append("Borderline - text shows mixed human/AI characteristics")
        elif ai_prob > 0.2:
            explanations.append("Mostly human-written with minor AI-like patterns")
        else:
            explanations.append("Strong human-written text characteristics")

    return " | ".join(explanations)


def save_results_csv(
    results: List[Dict],
    output_path: str,
    use_multi: bool = True
):
    """
    Save detection results to CSV file.

    Args:
        results: List of detection results
        output_path: Path to save CSV file
        use_multi: Whether multi-model detection was used

    Output columns:
        1. file_name: Name of the document
        2. classification: "AI_text" or "human_created"
        3. ai_probability: Probability score (0.0 to 1.0)
        4. supporting_details: Explanation and model details
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            "file_name",
            "classification",
            "ai_probability",
            "supporting_details"
        ])

        # Write data rows
        for result in results:
            if use_multi:
                filename = result["filename"]
                classification = "AI_text" if result["aggregate"]["majority_ai"] else "human_created"
                probability = result["aggregate"]["average_ai_probability"]
            else:
                filename = result["filename"]
                classification = "AI_text" if result.get("is_ai_generated", False) else "human_created"
                probability = result.get("ai_probability", 0.0)

            supporting_details = generate_explanation(result, use_multi)

            writer.writerow([
                filename,
                classification,
                f"{probability:.4f}",
                supporting_details
            ])

    print(f"CSV results saved to: {output_path}")


def run_detection(
    data_folder: str = "note_data/cms_notes",
    models: List[str] = None,
    output_file: str = None,
    csv_file: str = None,
    device: str = None,
    single_model: str = None
):
    """
    Run AI detection on all notes in the specified folder.

    Args:
        data_folder: Path to folder containing note files
        models: List of models to use (for multi-model detection)
        output_file: Path to save JSON results (optional)
        csv_file: Path to save CSV results (optional)
        device: Device to use ('cuda', 'mps', 'cpu')
        single_model: Use only this single model instead of multi-model
    """
    # Load notes
    notes = load_notes_from_folder(data_folder)

    if not notes:
        print("No notes to process. Exiting.")
        return

    print(f"\nProcessing {len(notes)} notes...")
    print("=" * 60)

    # Initialize detector(s)
    if single_model:
        detector = RoBERTaAIDetector(single_model, device)
        use_multi = False
    else:
        detector = MultiModelDetector(models, device)
        use_multi = True

    all_results = []

    for i, note in enumerate(notes, 1):
        filename = note["filename"]
        text = note["text"]

        print(f"\n[{i}/{len(notes)}] Processing: {filename}")

        if use_multi:
            result = detector.detect(text, filename)
            all_results.append(result)

            # Print summary
            agg = result["aggregate"]
            print(f"  Average AI Probability: {agg['average_ai_probability']:.2%}")
            print(f"  Consensus: {agg['consensus_ai_count']}/{agg['total_models']} models say AI")
            print(f"  Verdict: {'AI-GENERATED' if agg['majority_ai'] else 'HUMAN-WRITTEN'}")
        else:
            result = detector.detect(text, filename)
            all_results.append(asdict(result))

            # Print summary
            if result.error:
                print(f"  Error: {result.error}")
            else:
                print(f"  AI Probability: {result.ai_probability:.2%}")
                print(f"  Confidence: {result.confidence:.2%}")
                print(f"  Verdict: {'AI-GENERATED' if result.is_ai_generated else 'HUMAN-WRITTEN'}")

    # Print overall summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if use_multi:
        ai_count = sum(1 for r in all_results if r["aggregate"]["majority_ai"])
        avg_prob = sum(r["aggregate"]["average_ai_probability"] for r in all_results) / len(all_results)
    else:
        ai_count = sum(1 for r in all_results if r.get("is_ai_generated", False))
        valid_results = [r for r in all_results if not r.get("error")]
        avg_prob = sum(r["ai_probability"] for r in valid_results) / len(valid_results) if valid_results else 0

    print(f"Total notes processed: {len(notes)}")
    print(f"Detected as AI-generated: {ai_count} ({ai_count/len(notes)*100:.1f}%)")
    print(f"Detected as Human-written: {len(notes) - ai_count} ({(len(notes)-ai_count)/len(notes)*100:.1f}%)")
    print(f"Average AI probability: {avg_prob:.2%}")

    # Save results if output file specified
    if output_file:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "data_folder": data_folder,
            "total_notes": len(notes),
            "summary": {
                "ai_generated_count": ai_count,
                "human_written_count": len(notes) - ai_count,
                "average_ai_probability": round(avg_prob, 4)
            },
            "results": all_results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nJSON results saved to: {output_file}")

    # Save CSV results (always save, use default filename if not specified)
    if csv_file is None:
        csv_file = "ai_detection_results.csv"
    save_results_csv(all_results, csv_file, use_multi)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Detect AI-generated text in medical notes using HuggingFace RoBERTa models"
    )

    parser.add_argument(
        "--data-folder",
        default="note_data/cms_notes",
        help="Path to folder containing note text files (default: note_data/cms_notes)"
    )

    parser.add_argument(
        "--model",
        choices=list(RoBERTaAIDetector.AVAILABLE_MODELS.keys()),
        help="Use a single model instead of multi-model detection"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(RoBERTaAIDetector.AVAILABLE_MODELS.keys()),
        help="Specify which models to use for multi-model detection"
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        help="Device to use (default: auto-detect)"
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Path to save JSON results"
    )

    parser.add_argument(
        "--csv",
        default="ai_detection_results.csv",
        help="Path to save CSV results (default: ai_detection_results.csv)"
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )

    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download models from HuggingFace Hub to local cache and exit"
    )

    args = parser.parse_args()

    # List models if requested
    if args.list_models:
        print("Available RoBERTa models for AI detection:")
        print("-" * 50)
        for key, path in RoBERTaAIDetector.AVAILABLE_MODELS.items():
            print(f"  {key:12} -> {path}")
        return

    # Download models if requested
    if args.download_models:
        download_models(args.models)
        return

    # Run detection
    run_detection(
        data_folder=args.data_folder,
        models=args.models,
        output_file=args.output,
        csv_file=args.csv,
        device=args.device,
        single_model=args.model
    )


if __name__ == "__main__":
    main()
