"""
Download Sample Medical Data from CMS (Centers for Medicare & Medicaid Services)

This script downloads:
1. CMS Forms (CMS-1500 claim form, Certificate of Medical Necessity, etc.)
2. DE-SynPUF (Data Entrepreneurs' Synthetic Public Use File) - synthetic patient data

Sources:
- CMS Forms: https://www.cms.gov/medicare/forms-notices/cms-forms-list
- DE-SynPUF: https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files
"""

import os
import zipfile
import requests
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent / "note_data" / "cms_samples"

# CMS Form URLs (PDFs)
CMS_FORMS = {
    "CMS-1500_claim_form.pdf": "https://www.cms.gov/Medicare/CMS-Forms/CMS-Forms/downloads/CMS1500.pdf",
    "CMS-849_certificate_medical_necessity.pdf": "https://www.cms.gov/medicare/cms-forms/cms-forms/downloads/cms849.pdf",
    "CMS-1490S_patient_request_payment.pdf": "https://www.cms.gov/sites/default/files/repo-new/42/1490S_DME_Claim_Form.pdf",
    "Medicare_billing_guide.pdf": "https://www.cms.gov/files/document/mln006976-medicare-billing-cms-1500-837p.pdf",
}

# DE-SynPUF URLs (Sample 1 only - smaller download)
# Full dataset has 20 samples, each with multiple files
# See: https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf/de10-sample-1
CMS_BASE = "https://www.cms.gov"
DOWNLOADS_BASE = "http://downloads.cms.gov/files"

DE_SYNPUF_FILES = {
    # Sample 1 files - Beneficiary Summary (demographics, chronic conditions)
    "DE1_0_2008_Beneficiary_Summary_File_Sample_1.zip": f"{CMS_BASE}/research-statistics-data-and-systems/downloadable-public-use-files/synpufs/downloads/de1_0_2008_beneficiary_summary_file_sample_1.zip",
    "DE1_0_2009_Beneficiary_Summary_File_Sample_1.zip": f"{CMS_BASE}/research-statistics-data-and-systems/downloadable-public-use-files/synpufs/downloads/de1_0_2009_beneficiary_summary_file_sample_1.zip",
    "DE1_0_2010_Beneficiary_Summary_File_Sample_1.zip": f"{CMS_BASE}/sites/default/files/2020-09/DE1_0_2010_Beneficiary_Summary_File_Sample_1.zip",

    # Inpatient Claims (hospital stays, diagnoses, procedures)
    "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.zip": f"{CMS_BASE}/research-statistics-data-and-systems/downloadable-public-use-files/synpufs/downloads/de1_0_2008_to_2010_inpatient_claims_sample_1.zip",

    # Outpatient Claims
    "DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.zip": f"{CMS_BASE}/research-statistics-data-and-systems/downloadable-public-use-files/synpufs/downloads/de1_0_2008_to_2010_outpatient_claims_sample_1.zip",

    # Carrier Claims (physician/supplier claims)
    "DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.zip": f"{DOWNLOADS_BASE}/DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.zip",

    # Prescription Drug Events
    "DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_1.zip": f"{DOWNLOADS_BASE}/DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_1.zip",
}


def ensure_output_dir():
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "forms").mkdir(exist_ok=True)
    (OUTPUT_DIR / "synpuf").mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


def download_file(url: str, filepath: Path, description: str = "") -> bool:
    """Download a file from URL with progress indication."""
    try:
        print(f"  Downloading: {description or filepath.name}...")

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r  Progress: {percent:.1f}% ({downloaded:,} / {total_size:,} bytes)", end="")

        print(f"\r  Downloaded: {filepath.name} ({downloaded:,} bytes)          ")
        return True

    except requests.exceptions.RequestException as e:
        print(f"\r  Failed to download {filepath.name}: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract a zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"  Extracted: {zip_path.name}")
        return True
    except zipfile.BadZipFile as e:
        print(f"  Failed to extract {zip_path.name}: {e}")
        return False


def download_cms_forms():
    """Download CMS form PDFs."""
    print("\n" + "=" * 60)
    print("DOWNLOADING CMS FORMS")
    print("=" * 60)

    forms_dir = OUTPUT_DIR / "forms"
    downloaded = 0

    for filename, url in CMS_FORMS.items():
        filepath = forms_dir / filename
        if filepath.exists():
            print(f"  Already exists: {filename}")
            downloaded += 1
        elif download_file(url, filepath, filename):
            downloaded += 1

    print(f"\nDownloaded {downloaded}/{len(CMS_FORMS)} CMS forms")
    return downloaded


def download_synpuf(extract: bool = True, max_files: int = None):
    """
    Download DE-SynPUF synthetic patient data.

    Args:
        extract: Whether to extract zip files after download
        max_files: Maximum number of files to download (None = all)
    """
    print("\n" + "=" * 60)
    print("DOWNLOADING DE-SynPUF SYNTHETIC PATIENT DATA")
    print("=" * 60)
    print("Note: This is synthetic data based on real Medicare claims")
    print("Source: CMS 2008-2010 Data Entrepreneurs' Synthetic PUF")
    print()

    synpuf_dir = OUTPUT_DIR / "synpuf"
    downloaded = 0
    files_to_download = list(DE_SYNPUF_FILES.items())

    if max_files:
        files_to_download = files_to_download[:max_files]

    for filename, url in files_to_download:
        filepath = synpuf_dir / filename
        csv_name = filename.replace('.zip', '.csv')
        csv_path = synpuf_dir / csv_name

        # Check if already extracted
        if csv_path.exists():
            print(f"  Already exists: {csv_name}")
            downloaded += 1
            continue

        # Check if zip already downloaded
        if filepath.exists():
            print(f"  Already downloaded: {filename}")
        else:
            if not download_file(url, filepath, filename):
                continue

        # Extract if requested
        if extract and filepath.exists():
            extract_zip(filepath, synpuf_dir)
            # Optionally remove zip to save space
            # filepath.unlink()

        downloaded += 1

    print(f"\nDownloaded {downloaded}/{len(files_to_download)} SynPUF files")
    return downloaded


def show_data_structure():
    """Display information about the downloaded data structure."""
    print("\n" + "=" * 60)
    print("DE-SynPUF DATA STRUCTURE")
    print("=" * 60)
    print("""
The DE-SynPUF contains synthetic Medicare claims data with:

1. BENEFICIARY SUMMARY FILES (one per year: 2008, 2009, 2010)
   - Demographics: DOB, sex, race, state, county
   - Chronic conditions: Alzheimer's, CHF, diabetes, etc.
   - Medicare enrollment and coverage info

2. INPATIENT CLAIMS (hospital stays)
   - Admission/discharge dates
   - Diagnosis codes (ICD-9)
   - Procedure codes
   - DRG codes
   - Payment amounts

3. OUTPATIENT CLAIMS
   - Service dates
   - Diagnosis codes
   - HCPCS codes
   - Revenue center codes

4. CARRIER CLAIMS (physician/supplier)
   - Service dates
   - Diagnosis codes
   - HCPCS/CPT codes
   - Provider NPIs
   - Payment amounts

IMPORTANT: This is SYNTHETIC data - not real patient records.
The data mimics the structure of real Medicare claims for testing.
""")


def list_downloaded_files():
    """List all downloaded files."""
    print("\n" + "=" * 60)
    print("DOWNLOADED FILES")
    print("=" * 60)

    if not OUTPUT_DIR.exists():
        print("No files downloaded yet.")
        return

    for subdir in ["forms", "synpuf"]:
        dir_path = OUTPUT_DIR / subdir
        if dir_path.exists():
            files = list(dir_path.glob("*"))
            if files:
                print(f"\n{subdir.upper()}/")
                for f in sorted(files):
                    size = f.stat().st_size
                    if size > 1024 * 1024:
                        size_str = f"{size / (1024*1024):.1f} MB"
                    elif size > 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size} bytes"
                    print(f"  {f.name}: {size_str}")


def main():
    """Main function to download CMS sample data."""
    import argparse

    parser = argparse.ArgumentParser(description="Download CMS sample medical data")
    parser.add_argument("--forms-only", action="store_true",
                        help="Only download CMS forms (no SynPUF data)")
    parser.add_argument("--synpuf-only", action="store_true",
                        help="Only download SynPUF data (no forms)")
    parser.add_argument("--max-synpuf", type=int, default=3,
                        help="Max number of SynPUF files to download (default: 3)")
    parser.add_argument("--no-extract", action="store_true",
                        help="Don't extract zip files")
    parser.add_argument("--list", action="store_true",
                        help="List downloaded files and exit")
    parser.add_argument("--info", action="store_true",
                        help="Show data structure info and exit")

    args = parser.parse_args()

    if args.info:
        show_data_structure()
        return

    if args.list:
        list_downloaded_files()
        return

    ensure_output_dir()

    if not args.synpuf_only:
        download_cms_forms()

    if not args.forms_only:
        download_synpuf(
            extract=not args.no_extract,
            max_files=args.max_synpuf
        )

    list_downloaded_files()
    show_data_structure()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Files saved to: {OUTPUT_DIR}")
    print("\nTo use this data for fraud detection testing:")
    print("  1. CSV files contain structured claims data")
    print("  2. PDFs contain form templates")
    print("  3. Use the claims data to generate realistic test documents")


if __name__ == "__main__":
    main()
