"""
Collapse nested folder structure into a single folder.
Usage: python collapse_folders.py <source_folder> [output_folder]

Structure expected:
  source_folder/
    level1_subfolder/
      level2_subfolder/
        full_text_processed.txt

Output:
  output_folder/
    level1_subfolder_level2_subfolder_full_text_processed.txt
"""

import os
import sys
import shutil
import glob

def collapse_folders(source_folder, output_folder="Collapsed"):
    """Collapse nested folders into a single folder with renamed files."""

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Find all full_text_processed.txt files
    pattern = os.path.join(source_folder, "*", "*", "full_text_processed.txt")
    files = glob.glob(pattern)

    if not files:
        print(f"No files found matching pattern: {pattern}")
        return

    print(f"Found {len(files)} files to process\n")

    copied = 0
    for file_path in sorted(files):
        # Extract level1 and level2 folder names
        parts = file_path.split(os.sep)
        # parts[-1] = full_text_processed.txt
        # parts[-2] = level2 subfolder
        # parts[-3] = level1 subfolder
        level2 = parts[-2]
        level1 = parts[-3]

        # Create new filename
        new_filename = f"{level1}_{level2}_full_text_processed.txt"
        dest_path = os.path.join(output_folder, new_filename)

        # Copy file
        shutil.copy2(file_path, dest_path)
        print(f"Copied: {level1}/{level2}/full_text_processed.txt -> {new_filename}")
        copied += 1

    print(f"\nDone! Copied {copied} files to '{output_folder}'")


def main():
    if len(sys.argv) < 2:
        print("Usage: python collapse_folders.py <source_folder> [output_folder]")
        print("Example: python collapse_folders.py ./testfolder ./Collapsed")
        sys.exit(1)

    source_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "Collapsed"

    if not os.path.isdir(source_folder):
        print(f"Error: '{source_folder}' is not a valid directory")
        sys.exit(1)

    collapse_folders(source_folder, output_folder)


if __name__ == "__main__":
    main()
