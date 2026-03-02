"""
Simple RoBERTa AI Text Detector Test
"""

from transformers import pipeline

# Load model from local path
model_path = "~/.cache/huggingface/hub/models--roberta-base-openai-detector/snapshots/6cba99c003b711c7fe94f8a3aa2be35a792cb6fa/"
model_path = model_path.replace("~", "/Users/max-imac")

# Load the pipeline
classifier = pipeline("text-classification", model=model_path, local_files_only=True)

# Read the text file
with open("./note_data/cms_notes/discharge_summary_018.txt", "r") as f:
    text = f.read()

# Classify
result = classifier(text, truncation=True, max_length=512)

# Print results
print(f"Label: {result[0]['label']}")
print(f"Score: {result[0]['score']:.4f}")

# Interpret
if result[0]['label'] in ['LABEL_1', 'Fake']:
    ai_prob = result[0]['score']
else:
    ai_prob = 1 - result[0]['score']

print(f"AI Probability: {ai_prob:.2%}")
