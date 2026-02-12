#!/usr/bin/env python
"""
Parse ECLS-K:2011 ASCII Fixed-Width Data File
==============================================

Reads the .dct dictionary file and extracts specified columns
from the ASCII data file.

Note: The ECLS data uses a multi-line format where each record
spans 27 physical lines that must be concatenated.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Number of physical lines per logical record
LINES_PER_RECORD = 27


def parse_dct_file(dct_path: str) -> Tuple[Dict[str, Tuple[int, int, str, int]], Dict[int, int]]:
    """
    Parse Stata dictionary file to extract column specifications.

    Returns:
        Tuple of (specs dict, line_length)
        specs: Dict mapping variable name to (start_col, width, dtype, line_num)
    """
    specs = {}
    current_line = 1
    line_lengths = {}

    # Pattern: _column(start) type varname format "label"
    col_pattern = r'_column\((\d+)\)\s+(str\d+|long|int|byte|double|float)\s+(\w+)\s+%(\d+)'
    line_pattern = r'_line\((\d+)\)'

    with open(dct_path, 'r') as f:
        for text_line in f:
            # Check for line directive
            line_match = re.search(line_pattern, text_line)
            if line_match:
                current_line = int(line_match.group(1))
                continue

            # Check for column definition
            match = re.search(col_pattern, text_line)
            if match:
                start = int(match.group(1)) - 1  # Convert to 0-indexed
                dtype = match.group(2)
                varname = match.group(3)
                width = int(match.group(4).split('.')[0])

                # Map Stata types to Python
                if dtype.startswith('str'):
                    width = int(dtype[3:])
                    pytype = 'str'
                elif dtype in ['double', 'float']:
                    pytype = 'float'
                else:
                    pytype = 'int'

                specs[varname] = (start, width, pytype, current_line)

                # Track max position per line to compute line length
                end_pos = start + width
                if current_line not in line_lengths or end_pos > line_lengths[current_line]:
                    line_lengths[current_line] = end_pos

    # Compute cumulative offsets for each line
    line_offsets = {1: 0}
    for i in range(2, LINES_PER_RECORD + 1):
        line_offsets[i] = line_offsets[i-1] + line_lengths.get(i-1, 0)

    # Adjust start positions to be relative to concatenated record
    adjusted_specs = {}
    for varname, (start, width, pytype, line_num) in specs.items():
        adjusted_start = line_offsets[line_num] + start
        adjusted_specs[varname] = (adjusted_start, width, pytype, line_num)

    return adjusted_specs, line_lengths


def read_multiline_fixed_width(
    dat_path: str,
    dct_specs: Dict[str, Tuple[int, int, str, int]],
    columns: List[str],
    batch_size: int = 1000
) -> pd.DataFrame:
    """
    Read specific columns from multi-line fixed-width ASCII file.

    Args:
        dat_path: Path to .dat file
        dct_specs: Column specifications from dictionary
        columns: List of column names to extract
        batch_size: Number of records to process at once
    """
    # Filter to requested columns that exist
    available = [c for c in columns if c in dct_specs]
    missing = set(columns) - set(available)
    if missing:
        logger.warning(f"Columns not found in dictionary: {missing}")

    logger.info(f"Reading {len(available)} columns from {dat_path}")

    # Read and concatenate multi-line records
    records = []
    with open(dat_path, 'r', encoding='latin-1') as f:
        line_buffer = []
        record_count = 0

        for line in f:
            line_buffer.append(line.rstrip('\n\r'))

            if len(line_buffer) == LINES_PER_RECORD:
                # Concatenate lines into single record
                record = ''.join(line_buffer)

                # Extract values for requested columns
                row = {}
                for col in available:
                    start, width, pytype, _ = dct_specs[col]
                    value = record[start:start + width].strip()

                    if value in ['', '.'] or (value.startswith('-1') and len(value) < 3):
                        row[col] = np.nan
                    elif pytype == 'float':
                        try:
                            row[col] = float(value) if value else np.nan
                        except ValueError:
                            row[col] = np.nan
                    elif pytype == 'int':
                        try:
                            row[col] = int(float(value)) if value else np.nan
                        except ValueError:
                            row[col] = np.nan
                    else:
                        row[col] = value if value else None

                records.append(row)
                line_buffer = []
                record_count += 1

                if record_count % 5000 == 0:
                    logger.info(f"Processed {record_count:,} records...")

    logger.info(f"Total records: {len(records):,}")

    df = pd.DataFrame(records)
    return df


def main():
    """Main entry point."""
    # Paths
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    dct_path = raw_dir / "ECLSK2011_K5PUF.dct"
    dat_path = raw_dir / "childK5p.dat"

    # Parse dictionary
    logger.info("Parsing dictionary file...")
    specs, line_lengths = parse_dct_file(str(dct_path))
    logger.info(f"Found {len(specs)} variables in dictionary")
    logger.info(f"Line lengths: {dict(list(line_lengths.items())[:5])}...")

    # Variables to extract (using actual names from dictionary)
    columns = [
        # Identifiers
        "CHILDID",

        # Outcomes (5th grade)
        "X9RTHETK5",  # Reading theta
        "X9MTHETK5",  # Math theta

        # Demographics
        "X_RACETH_R",  # Race/ethnicity
        "X_CHSEX_R",   # Sex
        "X12SESL",     # SES continuous
        "X12LANGST",   # Home language

        # Baseline cognitive (K-2nd grade)
        "X1RTHETK5",  # K fall reading
        "X2RTHETK5",  # K spring reading
        "X1MTHETK5",  # K fall math
        "X2MTHETK5",  # K spring math

        # Executive function
        "X6DCCSSCR",  # DCCS score (spring 2013)

        # Approaches to learning
        "X1TCHAPP",   # K fall
        "X2TCHAPP",   # K spring
        "X4TCHAPP",   # 1st grade spring

        # Weights
        "W9C29P_9A0",  # Longitudinal weight
    ]

    # Read data
    df = read_multiline_fixed_width(str(dat_path), specs, columns)

    # Rename to match config expectations
    rename_map = {
        "X9RTHETK5": "X9RTHETA",
        "X9MTHETK5": "X9MTHETA",
        "X1RTHETK5": "X1RTHETK",
        "X2RTHETK5": "X2RTHETK",
        "X1MTHETK5": "X1MTHETK",
        "X2MTHETK5": "X2MTHETK",
        "X12SESL": "X12SESL",
        "W9C29P_9A0": "W9C29P_20",
    }
    df = df.rename(columns=rename_map)

    # Create SES quintiles from continuous SES
    if "X12SESL" in df.columns:
        df["X1SESQ5"] = pd.qcut(
            df["X12SESL"].dropna(),
            q=5,
            labels=[1, 2, 3, 4, 5]
        ).astype('Int64')
        # Fill back NaN positions
        df.loc[df["X12SESL"].isna(), "X1SESQ5"] = pd.NA

    # Save to parquet
    output_path = processed_dir / "ecls_extracted.parquet"
    df.to_parquet(output_path)
    logger.info(f"Saved to {output_path}")

    # Print summary
    logger.info("\nData Summary:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"\nMissing values:\n{df.isnull().sum()}")


if __name__ == "__main__":
    main()
