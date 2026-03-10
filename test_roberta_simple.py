"""
Simple RoBERTa AI Text Detector
Usage: python test_roberta_simple.py <input_file_or_folder> [output_csv]
"""

import sys
import os
import csv
import glob
from transformers import pipeline


def analyze_file(file_path, classifier):
    """Analyze a single file and return results dictionary."""
    with open(file_path, "r") as f:
        text = f.read()

    # RoBERTa Classification
    result = classifier(text, truncation=True, max_length=512)
    label = result[0]['label']
    score = result[0]['score']

    # Calculate AI probability
    if label in ['LABEL_1', 'Fake']:
        ai_prob = score
        human_prob = 1 - score
    else:
        ai_prob = 1 - score
        human_prob = score

    return {
        'filename': os.path.basename(file_path),
        'char_count': len(text),
        'word_count': len(text.split()),
        'roberta_label': label,
        'roberta_confidence': f"{score:.4f}",
        'ai_probability': f"{ai_prob:.4f}",
        'human_probability': f"{human_prob:.4f}",
        'classification': 'AI-GENERATED' if ai_prob > 0.5 else 'HUMAN-WRITTEN'
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_roberta_simple.py <input_file_or_folder> [output_csv]")
        print("Examples:")
        print("  python test_roberta_simple.py ./note_data/cms_notes/discharge_summary_018.txt results.csv")
        print("  python test_roberta_simple.py ./note_data/cms_notes/ results.csv")
        sys.exit(1)

    input_path = sys.argv[1]

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
    if len(sys.argv) >= 3:
        output_csv = sys.argv[2]
    else:
        if is_folder:
            folder_name = os.path.basename(os.path.normpath(input_path))
            output_csv = f"{folder_name}_results.csv"
        else:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_csv = f"{base_name}_results.csv"

    print(f"{'='*60}")
    print(f"RoBERTa AI Text Detection")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Files to process: {len(file_list)}")
    print(f"Output CSV: {output_csv}")
    print(f"{'='*60}\n")

    # Load RoBERTa model from local path
    print("Loading RoBERTa AI detector...")
    model_path = "~/.cache/huggingface/hub/models--roberta-base-openai-detector/snapshots/6cba99c003b711c7fe94f8a3aa2be35a792cb6fa/"
    model_path = model_path.replace("~", "/Users/max-imac")
    classifier = pipeline("text-classification", model=model_path, local_files_only=True)

    print(f"\nProcessing {len(file_list)} file(s)...\n")

    # CSV fieldnames
    fieldnames = [
        'filename',
        'char_count',
        'word_count',
        'roberta_label',
        'roberta_confidence',
        'ai_probability',
        'human_probability',
        'classification'
    ]

    # Process files and write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, file_path in enumerate(file_list, 1):
            filename = os.path.basename(file_path)
            print(f"[{i}/{len(file_list)}] Processing: {filename}...", end=" ")

            try:
                result = analyze_file(file_path, classifier)
                writer.writerow(result)
                print(f"AI: {result['ai_probability']} - {result['classification']}")
            except Exception as e:
                print(f"Error: {e}")

    print(f"\n{'='*60}")
    print(f"Completed processing {len(file_list)} file(s)")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
