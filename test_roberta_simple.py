"""
Simple RoBERTa AI Text Detector Test with Perplexity and Burstiness
Usage: python test_roberta_simple.py <input_file_or_folder> [output_csv]
"""

import sys
import os
import csv
import re
import glob
import numpy as np
from transformers import pipeline, GPT2LMHeadModel, GPT2TokenizerFast
import torch


def calculate_perplexity(text, model, tokenizer, device="cpu"):
    """Calculate perplexity using GPT-2 language model."""
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    perplexity = torch.exp(loss).item()
    return perplexity


def calculate_burstiness(text):
    """
    Calculate burstiness based on sentence length variation.
    Burstiness measures how "bursty" or varied the sentence structure is.
    Human text tends to have higher burstiness (more variation).
    AI text tends to be more uniform (lower burstiness).
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < 2:
        return 0.0, 0.0, 0.0, 0

    # Calculate word counts per sentence
    word_counts = [len(s.split()) for s in sentences]

    # Burstiness = (std - mean) / (std + mean)
    # Range: -1 to 1, where higher values indicate more variation
    mean_wc = np.mean(word_counts)
    std_wc = np.std(word_counts)

    if (std_wc + mean_wc) == 0:
        return 0.0, mean_wc, std_wc, len(sentences)

    burstiness = (std_wc - mean_wc) / (std_wc + mean_wc)

    return burstiness, mean_wc, std_wc, len(sentences)


def analyze_file(file_path, classifier, gpt2_model, gpt2_tokenizer, device):
    """Analyze a single file and return results dictionary."""
    with open(file_path, "r") as f:
        text = f.read()

    # RoBERTa Classification
    result = classifier(text, truncation=True, max_length=512)
    label = result[0]['label']
    score = result[0]['score']

    if label in ['LABEL_1', 'Fake']:
        ai_prob = score
    else:
        ai_prob = 1 - score

    # Perplexity
    perplexity = calculate_perplexity(text, gpt2_model, gpt2_tokenizer, device)

    # Burstiness
    burstiness, mean_words, std_words, num_sentences = calculate_burstiness(text)

    # Overall assessment
    ai_indicators = 0
    if ai_prob > 0.5:
        ai_indicators += 1
    if perplexity < 40:
        ai_indicators += 1
    if burstiness < -0.2:
        ai_indicators += 1

    if ai_indicators >= 2:
        overall_assessment = "LIKELY AI-GENERATED"
    elif ai_indicators == 1:
        overall_assessment = "UNCERTAIN"
    else:
        overall_assessment = "LIKELY HUMAN-WRITTEN"

    return {
        'filename': os.path.basename(file_path),
        'char_count': len(text),
        'word_count': len(text.split()),
        'roberta_label': label,
        'roberta_confidence': f"{score:.4f}",
        'ai_probability': f"{ai_prob:.4f}",
        'perplexity': f"{perplexity:.2f}",
        'burstiness': f"{burstiness:.4f}",
        'num_sentences': num_sentences,
        'mean_words_per_sentence': f"{mean_words:.1f}",
        'std_words_per_sentence': f"{std_words:.1f}",
        'overall_assessment': overall_assessment
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
    print(f"AI Text Detection Analysis")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Files to process: {len(file_list)}")
    print(f"Output CSV: {output_csv}")
    print(f"{'='*60}\n")

    # Load models once
    print("Loading RoBERTa AI detector...")
    model_path = "~/.cache/huggingface/hub/models--roberta-base-openai-detector/snapshots/6cba99c003b711c7fe94f8a3aa2be35a792cb6fa/"
    model_path = model_path.replace("~", "/Users/max-imac")
    classifier = pipeline("text-classification", model=model_path, local_files_only=True)

    print("Loading GPT-2 for perplexity calculation...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt2_model.eval()

    print(f"\nProcessing {len(file_list)} file(s)...\n")

    # CSV fieldnames
    fieldnames = [
        'filename',
        'char_count',
        'word_count',
        'roberta_label',
        'roberta_confidence',
        'ai_probability',
        'perplexity',
        'burstiness',
        'num_sentences',
        'mean_words_per_sentence',
        'std_words_per_sentence',
        'overall_assessment'
    ]

    # Process files and write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, file_path in enumerate(file_list, 1):
            filename = os.path.basename(file_path)
            print(f"[{i}/{len(file_list)}] Processing: {filename}...", end=" ")

            try:
                result = analyze_file(file_path, classifier, gpt2_model, gpt2_tokenizer, device)
                writer.writerow(result)
                print(f"Done - {result['overall_assessment']}")
            except Exception as e:
                print(f"Error: {e}")

    print(f"\n{'='*60}")
    print(f"Completed processing {len(file_list)} file(s)")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
