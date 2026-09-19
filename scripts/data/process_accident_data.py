# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openpyxl",
#     "pandas",
# ]
# ///
# ruff: noqa: E402
"""
Process accident dispatch data from Excel files into a single CSV.

Reads all `派遣回報紀錄YYYYMM.xlsx` files from the input directory,
keeps only non-cancelled accident rows
(`任務狀態 != 任務取消` and `性質 in {事故, 交通事故}`),
standardizes freeway names (`國1 -> N1`), and outputs a consolidated
`accident_data.csv`.

Usage:
    uv run scripts/data/process_accident_data.py
    uv run scripts/data/process_accident_data.py --input-dir 派遣系統資料 --output data/accident_data.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "派遣系統資料"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "accident_data.csv"


COLUMNS_TO_KEEP = [
    "任務編號",
    "國道",
    "方向",
    "里程數",
    "路段",
    "位置",
    "主線車道數",
    "輔助車道數",
    "剩餘車道數",
    "性質",
    "事故類別",
    "車種",
    "發送",        # got report time
    "到達",        # response team arrived
    "處理完成",    # cleared / completed
    "開始清理",
    "受傷人數",
    "死亡人數",
    "任務狀態",
]
CANCELLED_STATUS = "任務取消"
ACCIDENT_NATURES = {"事故", "交通事故"}


def standardize_freeway(name: str | None) -> str | None:
    if not name:
        return name
    m = re.match(r"國(\d+)", str(name))
    if m:
        return f"N{m.group(1)}"
    return str(name)


def process_single_file(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    available = [c for c in COLUMNS_TO_KEEP if c in df.columns]
    df = df[available].copy()

    if "任務狀態" in df.columns:
        df = df[df["任務狀態"] != CANCELLED_STATUS].copy()
    if "性質" in df.columns:
        df = df[df["性質"].isin(ACCIDENT_NATURES)].copy()
    if "任務狀態" in df.columns:
        df = df.drop(columns=["任務狀態"])
    if "國道" in df.columns:
        df["國道"] = df["國道"].apply(standardize_freeway)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Process accident dispatch Excel to CSV")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help=f"Default: {DEFAULT_INPUT_DIR}")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help=f"Default: {DEFAULT_OUTPUT_PATH}")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    files = sorted(input_dir.glob("派遣回報紀錄*.xlsx"))
    if not files:
        print(f"[ERROR] No dispatch Excel files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} dispatch file(s):")
    for f in files:
        print(f"  {f.name}")

    dfs: list[pd.DataFrame] = []
    for f in files:
        df = process_single_file(f)
        print(f"  {f.name}: {len(df)} rows (after filtering)")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(combined)} rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
