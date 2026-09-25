"""
Extract the intensity (y) values from Rigaku .ras XRD files.

For every .ras file in the "xrd_files" folder, this script writes a .txt file
with the same name into the "_xrd_results_clean" folder. Each .txt file holds
only the intensity values, one per line, with no header. The values are copied
exactly as they appear in the .ras file.

Both folders are located next to this script, so it can be run from anywhere:
    python ras_to_txt.py
"""

from pathlib import Path

# Build the input and output folder paths from this script's own location,
# so the script finds xrd_files no matter which folder the terminal is in.
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "xrd_files"
OUTPUT_DIR = SCRIPT_DIR / "_xrd_results_clean"


def read_intensities(ras_path):
    """Return the intensity values from one .ras file as a list of strings.

    A .ras file has a long header of instrument settings, followed by the
    measurements between the lines *RAS_INT_START and *RAS_INT_END.
    Each measurement line has three numbers separated by spaces:
        2-theta (deg)    intensity (counts)    correction factor
    for example: 5.0200 949.0000 1.0000
    """
    # latin-1 is used because .ras headers can contain special characters
    # that the default text encoding cannot read.
    lines = ras_path.read_text(encoding="latin-1").splitlines()

    intensities = []
    in_data = False
    for line in lines:
        line = line.strip()
        if line.startswith("*RAS_INT_START"):
            in_data = True  # the measurements start after this line
        elif line.startswith("*RAS_INT_END"):
            break  # the measurements are over; stop reading
        elif in_data and line:
            columns = line.split()
            intensities.append(columns[1])  # keep only the 2nd column (intensity)
    return intensities


# Create the output folder if it does not exist yet.
OUTPUT_DIR.mkdir(exist_ok=True)

# Find every .ras file in the input folder (in alphabetical order).
# Other file types, such as .raw or .asc, are ignored.
ras_files = sorted(f for f in INPUT_DIR.iterdir() if f.suffix.lower() == ".ras")

for ras_file in ras_files:
    intensities = read_intensities(ras_file)

    # A file with no measurements is reported and skipped.
    if not intensities:
        print(f"SKIPPED  {ras_file.name}: no data found")
        continue

    # Keep the original file name, swapping .ras for .txt (HD-10.ras -> HD-10.txt).
    txt_file = OUTPUT_DIR / (ras_file.stem + ".txt")

    # Write one intensity value per line.
    txt_file.write_text("\n".join(intensities) + "\n")
    print(f"OK       {ras_file.name} -> {txt_file.name} ({len(intensities)} values)")

print(f"\nDone. {len(ras_files)} .ras file(s) processed. Results are in: {OUTPUT_DIR}")
