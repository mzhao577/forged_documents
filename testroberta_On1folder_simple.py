"""
Simple RoBERTa AI Text Detector
Usage: python test_roberta_simple.py <input_file_or_folder> [output_csv] [--model openai|fakespot]
"""

import sys
import os
import csv
import glob
import argparse
from transformers import pipeline

# Model configurations
MODELS = {
    'openai': {
        'name': 'RoBERTa OpenAI Detector',
        'path': '~/.cache/huggingface/hub/models--roberta-base-openai-detector/snapshots/6cba99c003b711c7fe94f8a3aa2be35a792cb6fa/'
    },
    'fakespot': {
        'name': 'Fakespot AI Detector',
        'path': '~/.cache/huggingface/hub/models--fakespot-ai--roberta-base-ai-text-detection-v1/snapshots/f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db/'
    }
}


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
    parser = argparse.ArgumentParser(description='RoBERTa AI Text Detector')
    parser.add_argument('input_path', help='Input file or folder to analyze')
    parser.add_argument('output_csv', nargs='?', default=None, help='Output CSV file (optional)')
    parser.add_argument('--model', '-m', choices=['openai', 'fakespot'], default='openai',
                        help='Model to use: openai (default) or fakespot')

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

    # Get model configuration
    model_config = MODELS[model_choice]

    print(f"{'='*60}")
    print(f"RoBERTa AI Text Detection")
    print(f"{'='*60}")
    print(f"Model: {model_config['name']}")
    print(f"Input: {input_path}")
    print(f"Files to process: {len(file_list)}")
    print(f"Output CSV: {output_csv}")
    print(f"{'='*60}\n")

    # Load RoBERTa model from local path
    print(f"Loading {model_config['name']}...")
    model_path = model_config['path'].replace("~", os.path.expanduser("~"))
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
