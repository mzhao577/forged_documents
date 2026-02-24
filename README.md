# Medical Document Forgery Detection Pipeline

A comprehensive Python toolkit for detecting potentially forged or AI-generated medical documents. This pipeline combines multiple detection methods including AI text detection, medical consistency checking, and entity validation.

## Features

- **AI-Generated Text Detection** - Multiple algorithms to detect machine-generated content
- **Medical Consistency Checking** - Validates dates, dosages, and drug interactions
- **Medical Entity Validation** - Verifies NPI numbers, ICD codes, and drug names
- **Writing Style Analysis** - Analyzes sentence structure and vocabulary patterns
- **Ensemble Detection** - Combines multiple detectors for improved accuracy

## Installation

```bash
# Clone the repository
git clone https://github.com/mzhao577/forged_documents.git
cd forged_documents

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run detection with default HuggingFace detector
python run_detection.py --limit 5

# Run with Fast-DetectGPT (recommended for accuracy)
python run_detection.py --detector fast-detectgpt --limit 5

# Run on specific folder
python run_detection.py --detector fast-detectgpt --folder cms

# Run with ensemble (combines multiple detectors)
python run_detection.py --detector ensemble --limit 10
```

## Available Detectors

| Detector | Command | Memory | Description |
|----------|---------|--------|-------------|
| HuggingFace RoBERTa | `--detector huggingface` | ~500MB | OpenAI's RoBERTa-based detector (default) |
| Fast-DetectGPT | `--detector fast-detectgpt` | ~500MB | Curvature-based detection using GPT-2 |
| Binoculars | `--detector binoculars` | ~28GB | Perplexity comparison with Falcon-7B models |
| Ensemble | `--detector ensemble` | ~1GB+ | Combines multiple detectors with voting |

## Command Line Options

```bash
python run_detection.py [OPTIONS]

Options:
  --folder {generated,cms,all}  Which folder to scan (default: all)
  --limit N                     Max number of files to process
  --flawed-only                 Only process files with 'flawed' in name
  --detector {huggingface,fast-detectgpt,binoculars,ensemble}
                                AI detector to use (default: huggingface)
```

## Project Structure

```
forged_documents/
├── run_detection.py          # Main detection runner
├── document_analyzer.py      # Core analyzer orchestrating all checks
├── ai_detectors.py           # AI detection algorithms
├── consistency_checker.py    # Date/dosage/format validation
├── medical_validator.py      # Drug/ICD/NPI validation
├── style_analyzer.py         # Writing pattern analysis
├── metadata_analyzer.py      # Document metadata analysis
├── requirements.txt          # Python dependencies
│
├── note_data/                # Test data
│   ├── *.txt                 # Generated test files
│   └── cms_notes/            # CMS-derived medical notes
│
└── Utility Scripts
    ├── generate_test_data.py    # Generate synthetic test data
    ├── download_cms_data.py     # Download CMS sample data
    └── convert_cms_to_notes.py  # Convert CMS data to notes
```

## Detection Capabilities

### AI-Generated Content Detection
- **Fast-DetectGPT**: Uses conditional probability curvature to detect AI patterns
- **HuggingFace RoBERTa**: Pre-trained classifier for AI text detection
- **Binoculars**: Compares perplexity between observer and performer models
- **ROUGE Similarity**: Pattern matching against known AI phrases

### Medical Consistency Checks
- Future date detection
- Date sequence validation
- Dangerous dosage detection (e.g., Metformin > 1000mg)
- Drug interaction warnings (e.g., Warfarin + NSAIDs)
- Format consistency analysis

### Medical Entity Validation
- NPI number validation (Luhn algorithm)
- ICD-9/ICD-10 code format verification
- Drug name validation against known databases
- Provider credential verification

### Writing Style Analysis
- Sentence length variance (AI text often uniform)
- Vocabulary richness measurement
- AI-associated phrase detection
- Formality score calculation

## Example Output

```
======================================================================
FILE: suspicious_note.txt
======================================================================
OVERALL RISK LEVEL: MEDIUM
Overall Risk Score: 0.47

WARNINGS & ISSUES DETECTED:
  - AI Probability: 97.5%
  - Potentially dangerous dosage: metformin 5000.0mg (max: 1000mg)
  - Invalid NPI number: 1357924680
  - AI-associated phrase detected: 'Furthermore'

AI-GENERATED CONTENT DETECTION:
  Method: Fast-DetectGPT
  AI Probability: 97.5%
  Is AI Generated: True
```

## Running on High-Memory Machines

For the Binoculars detector (requires ~28GB RAM):

```bash
# On a machine with 32GB+ RAM or GPU
python run_detection.py --detector binoculars --folder all

# With GPU acceleration (install CUDA PyTorch first)
pip install torch --index-url https://download.pytorch.org/whl/cu118
python run_detection.py --detector binoculars
```

## API-Based Detectors (Optional)

The pipeline also supports API-based detectors. Set environment variables:

```bash
export GPTZERO_API_KEY="your-key"        # https://gptzero.me
export ORIGINALITY_API_KEY="your-key"    # https://originality.ai
export ZEROGPT_API_KEY="your-key"        # https://zerogpt.com
export COPYLEAKS_EMAIL="your-email"      # https://copyleaks.com
export COPYLEAKS_API_KEY="your-key"
```

## Generating Test Data

```bash
# Generate synthetic test files with various flaws
python generate_test_data.py

# Download CMS sample data
python download_cms_data.py

# Convert CMS claims to medical notes
python convert_cms_to_notes.py
```

## Risk Scoring

| Risk Level | Score Range | Description |
|------------|-------------|-------------|
| HIGH | >= 0.6 | Multiple serious issues detected |
| MEDIUM | >= 0.3 | Some concerns requiring review |
| LOW | < 0.3 | Minimal issues found |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is for educational and research purposes.

## Acknowledgments

- Fast-DetectGPT paper: "Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text"
- Binoculars paper: "Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text"
- OpenAI RoBERTa detector model
- CMS DE-SynPUF synthetic patient data
