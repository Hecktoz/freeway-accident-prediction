#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "openpyxl"]
# ///
"""
Parse LINE group chat export into structured accident report tables.

Outputs three CSVs in data/:
  line_reports_initial.csv      — 初報 + 無標記 (opening reports)
  line_reports_final.csv        — 結報 (closing/resolution reports)
  line_reports_intermediate.csv — N報 (mid-incident updates)

All three share `incident_key` for cross-table joining.
"""

import io
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parents[2]
FILE = ROOT / "data/[LINE] 中分局重大及特殊災害群組的聊天11002_11406.txt"
OUT_DIR = ROOT / "data"

# ── regexes ─────────────────────────────────────────────────────────────────────

DATE_HDR  = re.compile(r"^(\d{4}/\d{2}/\d{2})[（(]")
MSG_HDR   = re.compile(r"^(\d{2}:\d{2})\t([^\t]+)\t(.*)")
FIELD_HDR = re.compile(r"^\s*(\d)[.．]\s*[^：:\n]+[：:]\s*", re.MULTILINE)
LABEL_RE  = re.compile(r"\s*[（(](初報|結報|\d+報)[）)]\s*")
LOC_RE    = re.compile(r"國([1-9])([北南東西])向\s*([\d.]+)\s*[Kk]", re.IGNORECASE)
TIME_RE   = re.compile(r"(\d{1,2})/(\d{1,2})[,，]\s*(\d{2}):(\d{2})")
RESOLV_RE = re.compile(r"(\d{2}:\d{2})\s*(?:排除|解除|全線通車)")

DIR_MAP = {"北": "N", "南": "S", "東": "E", "西": "W"}


# ── file reader ──────────────────────────────────────────────────────────────

def read_messages(path: Path):
    """Yield (line_date, send_time, sender, body) for each message."""
    lines = path.read_text(encoding="utf-8").splitlines()
    current_date = None
    i = 0
    while i < len(lines):
        m = DATE_HDR.match(lines[i])
        if m:
            current_date = date.fromisoformat(m.group(1).replace("/", "-"))
            i += 1
            continue

        m = MSG_HDR.match(lines[i])
        if m and current_date:
            send_time = m.group(1)
            sender    = m.group(2).strip()
            body_first = m.group(3)

            if body_first.startswith('"') and not (
                len(body_first) > 1 and body_first.endswith('"')
            ):
                # multi-line quoted body: accumulate until closing "
                parts = [body_first[1:]]
                i += 1
                while i < len(lines):
                    ln = lines[i]
                    i += 1
                    if ln == '"' or ln.endswith('"'):
                        if ln != '"':
                            parts.append(ln.rstrip('"'))
                        break
                    parts.append(ln)
                body = "\n".join(parts).strip()
            else:
                body = body_first.strip('"').strip()
                i += 1

            yield current_date, send_time, sender, body
        else:
            i += 1


# ── field extractor ───────────────────────────────────────────────────────

def extract_fields(body: str) -> dict[int, str] | None:
    """Extract numbered fields (1–9) from a 9-field structured report."""
    matches = list(FIELD_HDR.finditer(body))
    if len(matches) < 6:
        return None
    fields: dict[int, str] = {}
    for idx, m in enumerate(matches):
        num   = int(m.group(1))
        start = m.end()
        end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        fields[num] = (
            body[start:end].strip().replace("\t", " ").replace("\n", " / ")
        )
    return fields if 2 in fields and 3 in fields else None


# ── helpers ──────────────────────────────────────────────────────────────

def infer_event_date(time_str: str, line_date: date) -> date | None:
    m = TIME_RE.match(time_str.strip())
    if not m:
        return None
    mo, dd = int(m.group(1)), int(m.group(2))
    for yr in (line_date.year, line_date.year - 1):
        try:
            d = date(yr, mo, dd)
            if abs((line_date - d).days) <= 60:
                return d
        except ValueError:
            continue
    return None


# ── main parser ───────────────────────────────────────────────────────────

def parse_reports() -> pd.DataFrame:
    rows = []
    for line_date, send_time, sender, body in read_messages(FILE):
        if sender != "高公局中區交控":
            continue
        fields = extract_fields(body)
        if not fields:
            continue

        # --- field 3: location + report label ---
        location_full  = fields.get(3, "")
        label_m        = LABEL_RE.search(location_full)
        report_label   = label_m.group(1) if label_m else "無標記"
        location_clean = LABEL_RE.sub("", location_full).strip()

        # --- field 2: event time ---
        time_str  = fields.get(2, "")
        evt_date  = infer_event_date(time_str, line_date)
        time_m    = TIME_RE.match(time_str.strip())
        event_time = f"{time_m.group(3)}:{time_m.group(4)}" if time_m else ""

        # --- parse freeway / direction / mileage from location ---
        loc_m = LOC_RE.search(location_clean)
        if loc_m:
            freeway   = f"N{loc_m.group(1)}"
            direction = DIR_MAP[loc_m.group(2)]
            mileage_m = round(float(loc_m.group(3).rstrip(".")) * 1000)
        else:
            freeway = direction = ""
            mileage_m = None

        # --- incident_key: stable across 初報/N報/結報 ---
        if evt_date and event_time and freeway:
            incident_key = (
                f"{evt_date.strftime('%Y%m%d')}"
                f"_{event_time.replace(':', '')}"
                f"_{freeway}{direction}{mileage_m}"
            )
        else:
            incident_key = (
                f"{line_date.strftime('%Y%m%d')}"
                f"_{send_time.replace(':', '')}"
                f"_{location_clean[:30]}"
            )

        # --- resolution time from progress field ---
        prog     = fields.get(7, "")
        resolv_m = RESOLV_RE.search(prog)
        resolution_time = resolv_m.group(1) if resolv_m else ""

        rows.append({
            "incident_key":     incident_key,
            "report_label":     report_label,
            "line_date":        str(line_date),
            "send_time":        send_time,
            "unit":             fields.get(1, ""),
            "event_date":       str(evt_date) if evt_date else "",
            "event_time":       event_time,
            "location":         location_clean,
            "freeway":          freeway,
            "direction":        direction,
            "mileage_m":        mileage_m,
            "description":      fields.get(4, ""),
            "casualties":       fields.get(5, ""),
            "traffic_impact":   fields.get(6, ""),
            "progress":         prog,
            "resolution_time":  resolution_time,
            "source":           fields.get(8, ""),
        })

    return pd.DataFrame(rows)


# ── entry point ───────────────────────────────────────────────────────────

def main() -> None:
    df = parse_reports()
    if df.empty:
        print("ERROR: no records parsed — check FILE path or message format", file=sys.stderr)
        sys.exit(1)

    print(f"Total structured reports parsed: {len(df)}")
    print("\n報類分布:")
    print(df["report_label"].value_counts().to_string())

    init_df  = df[df["report_label"].isin(["初報", "無標記"])].copy()
    final_df = df[df["report_label"] == "結報"].copy()
    inter_df = df[df["report_label"].str.fullmatch(r"\d+報")].copy()

    for tbl in (init_df, final_df, inter_df):
        tbl.insert(0, "report_id", range(1, len(tbl) + 1))

    out_init  = OUT_DIR / "line_reports_initial.csv"
    out_final = OUT_DIR / "line_reports_final.csv"
    out_inter = OUT_DIR / "line_reports_intermediate.csv"

    for path, tbl, label in [
        (out_init,  init_df,  "初報/無標記"),
        (out_final, final_df, "結報"),
        (out_inter, inter_df, "中間報"),
    ]:
        try:
            tbl.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"{label} → {path.name}  ({len(tbl):,} rows)")
        except PermissionError:
            print(f"[skip] {path.name} is open in another program — CSV not updated")

    init_keys  = set(init_df["incident_key"])
    final_keys = set(final_df["incident_key"])
    matched    = init_keys & final_keys
    print(
        f"\nincident_key 對照:"
        f"  {len(matched):,} 組有完整初報+結報"
        f"  / {len(init_keys):,} 初報"
        f"  / {len(final_keys):,} 結報"
    )

    # filtered sheet: confirmed 初報 with valid mileage
    confirmed_df = init_df[
        (init_df["report_label"] == "初報") &
        init_df["mileage_m"].notna()
    ].copy()

    out_excel = OUT_DIR / "line_reports.xlsx"
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        init_df.to_excel(writer, sheet_name="初報_全部", index=False)
        confirmed_df.to_excel(writer, sheet_name="初報_確認有里程", index=False)

    print(f"\nExcel → {out_excel.name}")
    print(f"  工作表 [初報_全部]     {len(init_df):,} 筆")
    print(f"  工作表 [初報_確認有里程] {len(confirmed_df):,} 筆"
          f"  (排除 {len(init_df) - len(confirmed_df):,} 筆無里程或無標記)")

    # preview
    print("\n--- 初報_確認有里程 前5筆 ---")
    preview_cols = ["report_id", "incident_key", "event_date", "event_time",
                    "freeway", "direction", "mileage_m", "description", "casualties"]
    print(confirmed_df[preview_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()
