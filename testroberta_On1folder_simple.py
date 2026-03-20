"""
Simple RoBERTa AI Text Detector
Usage: python testroberta_On1folder_simple.py <input_file_or_folder> [output_csv] [--model openai|fakespot|chatgpt|roberta]

Uses detect_ai_detectors.py for detection with full analysis features including:
- Duplicate content detection
- Segment-level AI analysis
- Detailed classification reasons
"""

import sys
import os
import csv
import glob
import argparse

# Import from detect_ai_detectors module
from detect_ai_detectors import (
    HuggingFaceDetector,
    perform_detailed_analysis,
    generate_explanation,
    generate_analysis_reason
)


def analyze_file(file_path, detector):
    """Analyze a single file and return results dictionary."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Run detection
    result = detector.detect(text)

    # Perform detailed analysis for AI-classified texts
    if result.is_ai_generated and not result.error:
        result.analysis = perform_detailed_analysis(text, result, detector)

    # Extract analysis fields
    has_duplicates = ""
    duplicate_ratio = ""
    high_ai_segments = ""
    analysis_reason = generate_analysis_reason(result)

    if result.analysis:
        has_duplicates = "Yes" if result.analysis.has_duplicates else "No"
        duplicate_ratio = f"{result.analysis.duplicate_ratio:.1%}"
        high_ai_segments = str(len(result.analysis.high_ai_segments))

    return {
        'filename': os.path.basename(file_path),
        'char_count': len(text),
        'word_count': len(text.split()),
        'classification': 'AI_text' if result.is_ai_generated else 'human_created',
        'ai_probability': f"{result.ai_probability:.4f}",
        'confidence': f"{result.confidence:.4f}",
        'analysis_reason': analysis_reason,
        'has_duplicates': has_duplicates,
        'duplicate_ratio': duplicate_ratio,
        'high_ai_segments': high_ai_segments,
        'raw_label': result.details.get('raw_label', ''),
        'explanation': generate_explanation(result)
    }


def main():
    parser = argparse.ArgumentParser(description='RoBERTa AI Text Detector')
    parser.add_argument('input_path', help='Input file or folder to analyze')
    parser.add_argument('output_csv', nargs='?', default=None, help='Output CSV file (optional)')
    parser.add_argument('--model', '-m',
                        choices=list(HuggingFaceDetector.MODEL_ALIASES.keys()),
                        default='openai',
                        help=f'Model to use: {", ".join(HuggingFaceDetector.MODEL_ALIASES.keys())} (default: openai)')

    args = parser.parse_args()
    input_path = args.input_path
    model_choice = args.model

    # Determine if input is file or folder
    if os.path.isdir(input_path):
        # Get all text files in the folder
        file_list = glob.glob(os.path.join(input_path, "*.txt"))
        file_list.sort()
        if not file_list:
            print(f"Error: No .txt files found in '{input_path}'")
            sys.exit(1)
        is_folder = True
    elif os.path.isfile(input_path):
        file_list = [input_path]
        is_folder = False
    else:
        print(f"Error: '{input_path}' is not a valid file or folder.")
        sys.exit(1)

    # Output CSV file
    if args.output_csv:
        output_csv = args.output_csv
    else:
        if is_folder:
            folder_name = os.path.basename(os.path.normpath(input_path))
            output_csv = f"{folder_name}_results.csv"
        else:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_csv = f"{base_name}_results.csv"

    # Create detector using the model alias
    detector = HuggingFaceDetector(model_name=model_choice)

    print(f"{'='*60}")
    print(f"RoBERTa AI Text Detection (with Analysis)")
    print(f"{'='*60}")
    print(f"Model: {detector.name}")
    print(f"Model alias: {model_choice} -> {detector.model_name}")
    print(f"Input: {input_path}")
    print(f"Files to process: {len(file_list)}")
    print(f"Output CSV: {output_csv}")
    print(f"{'='*60}\n")

    print(f"Loading model...")

    # CSV fieldnames
    fieldnames = [
        'filename',
        'char_count',
        'word_count',
        'classification',
        'ai_probability',
        'confidence',
        'analysis_reason',
        'has_duplicates',
        'duplicate_ratio',
        'high_ai_segments',
        'raw_label',
        'explanation'
    ]

    # Process files and write to CSV
    results_summary = {'ai': 0, 'human': 0, 'errors': 0}

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, file_path in enumerate(file_list, 1):
            filename = os.path.basename(file_path)
            print(f"[{i}/{len(file_list)}] Processing: {filename}...", end=" ")

            try:
                result = analyze_file(file_path, detector)
                writer.writerow(result)

                if result['classification'] == 'AI_text':
                    results_summary['ai'] += 1
                    print(f"AI: {result['ai_probability']} - {result['classification']}")
                    if result['analysis_reason']:
                        print(f"           Reason: {result['analysis_reason']}")
                else:
                    results_summary['human'] += 1
                    print(f"AI: {result['ai_probability']} - {result['classification']}")

            except Exception as e:
                results_summary['errors'] += 1
                print(f"Error: {e}")

    total = results_summary['ai'] + results_summary['human']
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(file_list)}")
    print(f"Detected as AI-generated: {results_summary['ai']} ({results_summary['ai']/total*100:.1f}%)" if total > 0 else "")
    print(f"Detected as Human-written: {results_summary['human']} ({results_summary['human']/total*100:.1f}%)" if total > 0 else "")
    if results_summary['errors'] > 0:
        print(f"Errors: {results_summary['errors']}")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
