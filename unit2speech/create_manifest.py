import os
import json
from pathlib import Path
import argparse

def create_manifest(wav_dir, units_dir, manifest_path, code_type=None):
    """
    Creates a manifest file for the CodeDataset.

    Args:
        wav_dir (str): Path to the directory containing wav files.
        units_dir (str): Path to the directory containing units files.
        manifest_path (str): Path where the manifest file will be saved.
        code_type (str, optional): Type of code data ('cpc_km100', 'vqvae256', 'hubert', etc.).
                                   If None, only wav paths are included.
    """
    wav_dir = Path(wav_dir)
    units_dir = Path(units_dir)
    manifest_path = Path(manifest_path)

    if not wav_dir.exists():
        raise FileNotFoundError(f"Wav directory not found: {wav_dir}")
    if code_type and not units_dir.exists():
        raise FileNotFoundError(f"Units directory not found: {units_dir}")

    wav_files = sorted(wav_dir.rglob("*.wav"))
    manifest_entries = []

    for wav_file in wav_files:
        # Assume corresponding units file has the same base name with .units extension
        units_file = units_dir / wav_file.with_suffix('.unit').name

        if units_file.exists() and code_type:
            # Read the units file and convert to space-separated string of integers
            with open(units_file, 'r') as f:
                units_content = f.read().strip()
                # Assuming units are space-separated integers; modify as needed
                # Convert to a string of integers separated by space
                # If units are not integers, adjust the parsing accordingly
                units_str = ' '.join(map(str, map(int, units_content.split())))
            
            entry = {
                "audio": str(wav_file),
                code_type: units_str
            }
            manifest_entries.append(entry)
        else:
            # Only include wav path
            pass

    # Write to manifest file
    with open(manifest_path, 'w') as mf:
        for entry in manifest_entries:
            if isinstance(entry, dict):
                mf.write(json.dumps(entry) + "\n")
            else:
                mf.write(entry + "\n")
    
    print(f"Manifest file created at: {manifest_path}")
    print(f"Total entries: {len(manifest_entries)}")

def main():
    parser = argparse.ArgumentParser(description="Create manifest file for CodeDataset.")
    parser.add_argument('--wav_dir', type=str, required=True, help="Path to wav files directory.")
    parser.add_argument('--units_dir', type=str, default=None, help="Path to units files directory.")
    parser.add_argument('--manifest_path', type=str, required=True, help="Output path for the manifest file.")
    parser.add_argument('--code_type', type=str, default=None, help="Type of code data (e.g., 'cpc_km100', 'vqvae256', 'hubert').")
    
    args = parser.parse_args()

    create_manifest(args.wav_dir, args.units_dir, args.manifest_path, args.code_type)

if __name__ == "__main__":
    main()
