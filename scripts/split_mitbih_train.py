#!/usr/bin/env python3
"""Split mitbih_train.csv into 4 approximately equal parts.

Usage:
  python scripts/split_mitbih_train.py --input data/ecg_mitbih/mitbih_train.csv --parts 4

Creates files alongside the input named <basename>1.csv .. <basename><parts>.csv
"""
import argparse
import os


def split_file(input_path, parts=4, out_dir=None, prefix=None, has_header=False):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    if out_dir is None:
        out_dir = os.path.dirname(input_path)
    os.makedirs(out_dir, exist_ok=True)

    if prefix is None:
        prefix = os.path.splitext(os.path.basename(input_path))[0]

    # Count total lines (optionally skipping header)
    with open(input_path, "r", newline="") as f:
        if has_header:
            header = f.readline()
        else:
            header = None
        total = sum(1 for _ in f)

    if total == 0:
        raise ValueError("Input CSV has no data rows to split.")

    base = total // parts
    rem = total % parts

    # Open output files and write headers if present
    out_files = []
    for i in range(parts):
        partnum = i + 1
        out_name = os.path.join(out_dir, f"{prefix}{partnum}.csv")
        out_f = open(out_name, "w", newline="")
        if header is not None:
            out_f.write(header)
        out_files.append(out_f)

    # Distribute rows into parts, first `rem` parts get one extra row
    with open(input_path, "r", newline="") as f:
        if has_header:
            _ = f.readline()  # skip header
        for i in range(parts):
            count = base + (1 if i < rem else 0)
            out_f = out_files[i]
            for _ in range(count):
                line = f.readline()
                if not line:
                    break
                out_f.write(line)

    for of in out_files:
        of.close()

    return [os.path.join(out_dir, f"{prefix}{i+1}.csv") for i in range(parts)]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=False,
                   default=os.path.join("data", "ecg_mitbih", "mitbih_train.csv"),
                   help="Path to mitbih_train.csv")
    p.add_argument("--parts", "-p", type=int, default=4, help="Number of parts to split into")
    p.add_argument("--has-header", dest="has_header", action="store_true", help="Treat the first line as a header and preserve it in each part")
    args = p.parse_args()

    try:
        out_paths = split_file(args.input, parts=args.parts, prefix=None, out_dir=None, has_header=args.has_header)
        print("Wrote:")
        for pth in out_paths:
            print(" -", pth)
    except Exception as e:
        print("ERROR:", e)
