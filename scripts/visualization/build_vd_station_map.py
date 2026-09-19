# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lxml",
#     "pyarrow",
#     "pandas",
# ]
# ///
"""
Build a single-page HTML viewer for VD station inventory.

Sources:
  data/metadata/vd_static.xml   -- from tisvcloud.freeway.gov.tw/history/motc20/VD.xml
  data/metadata/vd_dim.parquet  -- built by build_station_vd_drive.py

Output:
  output/vd_station_map.html

Example:
  uv run scripts/visualization/build_vd_station_map.py
  uv run scripts/visualization/build_vd_station_map.py --fetch   # re-download VD.xml first
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

VDXML_URL = "https://tisvcloud.freeway.gov.tw/history/motc20/VD.xml"
VDXML_PATH = PROJECT_ROOT / "data/metadata/vd_static.xml"
VD_DIM_PATH = PROJECT_ROOT / "data/metadata/vd_dim.parquet"
OUTPUT_PATH = PROJECT_ROOT / "output/vd_station_map.html"

VD_TYPE_LABEL = {"1": "感應線圈", "2": "微波", "3": "影像辨識"}
LOCATION_TYPE_LABEL = {"1": "匝道", "5": "主線", "6": "其他"}
DETECTION_TYPE_LABEL = {"1": "單向", "2": "雙向", "3": "多向"}

FREEWAY_RE = re.compile(r"^VD-(N\d+)-([NSEW])-([\d.]+)")

FREEWAY_COLORS = {"N1": "#e74c3c", "N3": "#3498db", "N4": "#27ae60", "N6": "#f39c12"}
DIRECTION_LABEL = {"N": "北向", "S": "南向", "E": "東向", "W": "西向"}


# ---------------------------------------------------------------------------
# Parse VD.xml
# ---------------------------------------------------------------------------

def fetch_vd_xml() -> None:
    import urllib.request
    print(f"Fetching {VDXML_URL} ...", flush=True)
    urllib.request.urlretrieve(VDXML_URL, VDXML_PATH)
    print(f"Saved to {VDXML_PATH}", flush=True)


def parse_vd_xml(xml_path: Path) -> list[dict]:
    import lxml.etree as ET

    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    ns = {"m": "http://traffic.transportdata.tw/standard/traffic/schema/"}

    records = []
    for vd in root.findall(".//m:VD", ns):
        vdid = vd.findtext("m:VDID", namespaces=ns) or ""
        m = FREEWAY_RE.match(vdid)
        if not m:
            continue

        # Only keep N1 / N3 / N4 / N6
        freeway = m.group(1)
        if freeway not in ("N1", "N3", "N4", "N6"):
            continue

        links = vd.findall(".//m:DetectionLink", ns)
        lane_num = int(links[0].findtext("m:LaneNum", namespaces=ns) or 0) if links else 0
        actual_lane_num = int(links[0].findtext("m:ActualLaneNum", namespaces=ns) or 0) if links else 0
        link_ids = [lk.findtext("m:LinkID", namespaces=ns) or "" for lk in links]

        sec_start = vd.findtext("m:RoadSection/m:Start", namespaces=ns) or ""
        sec_end = vd.findtext("m:RoadSection/m:End", namespaces=ns) or ""

        records.append({
            "vdid": vdid,
            "freeway": freeway,
            "direction": m.group(2),
            "mileage_km": float(m.group(3)),
            "lat": float(vd.findtext("m:PositionLat", namespaces=ns) or 0),
            "lon": float(vd.findtext("m:PositionLon", namespaces=ns) or 0),
            "vd_type": vd.findtext("m:VDType", namespaces=ns) or "",
            "location_type": vd.findtext("m:LocationType", namespaces=ns) or "",
            "detection_type": vd.findtext("m:DetectionType", namespaces=ns) or "",
            "lane_num": lane_num,
            "actual_lane_num": actual_lane_num,
            "road_name": vd.findtext("m:RoadName", namespaces=ns) or "",
            "road_section": f"{sec_start}～{sec_end}" if sec_start or sec_end else "",
            "location_mile": vd.findtext("m:LocationMile", namespaces=ns) or "",
            "bi_directional": int(vd.findtext("m:BiDirectional", namespaces=ns) or 0),
            "link_ids": ",".join(link_ids),
        })

    print(f"Parsed {len(records)} VDs from VD.xml", flush=True)
    return records


# ---------------------------------------------------------------------------
# Merge with vd_dim
# ---------------------------------------------------------------------------

def load_vd_dim(parquet_path: Path) -> dict[str, dict]:
    import pandas as pd
    df = pd.read_parquet(str(parquet_path))
    total_days = df["n_source_days"].max()
    # Use 426 as denominator if that matches the known range, else use max
    # Actually use n_source_days vs a reference of 426 days (2024-04-01 to 2025-05-31)
    REFERENCE_DAYS = 426
    result = {}
    for _, row in df.iterrows():
        result[row["vdid"]] = {
            "first_date": str(row["first_source_date"]),
            "last_date": str(row["last_source_date"]),
            "n_days": int(row["n_source_days"]),
            "coverage": round(row["n_source_days"] / REFERENCE_DAYS, 3),
            "n_rows": int(row["n_rows_station_1m"]),
        }
    print(f"Loaded {len(result)} VDs from vd_dim.parquet", flush=True)
    return result


def merge(static_records: list[dict], dim_map: dict[str, dict]) -> list[dict]:
    merged = []
    in_dim = 0
    for rec in static_records:
        dim = dim_map.get(rec["vdid"])
        if dim:
            rec.update(dim)
            rec["in_dataset"] = True
            in_dim += 1
        else:
            rec.update({
                "first_date": None, "last_date": None,
                "n_days": 0, "coverage": 0.0, "n_rows": 0,
                "in_dataset": False,
            })
        merged.append(rec)
    print(f"Matched {in_dim}/{len(static_records)} static VDs to vd_dim", flush=True)
    return merged


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def build_html(records: list[dict]) -> str:
    # Summary stats
    n_total = len(records)
    n_in_dataset = sum(1 for r in records if r["in_dataset"])
    by_freeway = {}
    for r in records:
        fw = r["freeway"]
        if fw not in by_freeway:
            by_freeway[fw] = {"total": 0, "in_dataset": 0}
        by_freeway[fw]["total"] += 1
        if r["in_dataset"]:
            by_freeway[fw]["in_dataset"] += 1

    # JSON for map
    features = []
    for r in records:
        if r["lat"] == 0 and r["lon"] == 0:
            continue
        coverage_pct = int(r["coverage"] * 100)
        coverage_class = (
            "cov-high" if r["coverage"] >= 0.9
            else "cov-mid" if r["coverage"] >= 0.5
            else "cov-low" if r["coverage"] > 0
            else "cov-none"
        )
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {
                "vdid": r["vdid"],
                "freeway": r["freeway"],
                "direction": r["direction"],
                "direction_label": DIRECTION_LABEL.get(r["direction"], r["direction"]),
                "mileage_km": r["mileage_km"],
                "lat": r["lat"],
                "lon": r["lon"],
                "vd_type": VD_TYPE_LABEL.get(r["vd_type"], r["vd_type"]),
                "location_type": LOCATION_TYPE_LABEL.get(r["location_type"], r["location_type"]),
                "detection_type": DETECTION_TYPE_LABEL.get(r["detection_type"], r["detection_type"]),
                "lane_num": r["lane_num"],
                "actual_lane_num": r["actual_lane_num"],
                "road_section": r["road_section"],
                "location_mile": r["location_mile"],
                "bi_directional": r["bi_directional"],
                "link_ids": r["link_ids"],
                "in_dataset": r["in_dataset"],
                "first_date": r["first_date"] or "—",
                "last_date": r["last_date"] or "—",
                "n_days": r["n_days"],
                "coverage_pct": coverage_pct,
                "coverage_class": coverage_class,
                "n_rows": r["n_rows"],
                "color": FREEWAY_COLORS.get(r["freeway"], "#888"),
            },
        })

    geojson = json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False)

    # Table rows (all records, sorted by freeway / direction / mileage)
    sorted_records = sorted(records, key=lambda r: (r["freeway"], r["direction"], r["mileage_km"], r["vdid"]))

    table_rows = []
    for r in sorted_records:
        cov = int(r["coverage"] * 100)
        cov_bar_color = "#27ae60" if cov >= 90 else "#f39c12" if cov >= 50 else "#e74c3c" if cov > 0 else "#ccc"
        in_ds = "●" if r["in_dataset"] else "○"
        fw_color = FREEWAY_COLORS.get(r["freeway"], "#888")
        table_rows.append(
            f'<tr data-freeway="{r["freeway"]}" data-direction="{r["direction"]}" '
            f'data-inds="{1 if r["in_dataset"] else 0}" data-vdid="{r["vdid"]}" '
            f'data-vdtype="{r["vd_type"]}" data-loctype="{r["location_type"]}">'
            f'<td><span class="fw-badge" style="background:{fw_color}">{r["freeway"]}</span></td>'
            f'<td>{DIRECTION_LABEL.get(r["direction"], r["direction"])}</td>'
            f'<td class="mono">{r["mileage_km"]:.3f} K</td>'
            f'<td class="mono vdid-cell" title="{r["vdid"]}">{r["vdid"]}</td>'
            f'<td>{VD_TYPE_LABEL.get(r["vd_type"], r["vd_type"])}</td>'
            f'<td>{LOCATION_TYPE_LABEL.get(r["location_type"], r["location_type"])}</td>'
            f'<td>{r["lane_num"]}</td>'
            f'<td>{r["road_section"]}</td>'
            f'<td style="color:{"#27ae60" if r["in_dataset"] else "#aaa"}">{in_ds}</td>'
            f'<td>'
            f'<div class="cov-bar-wrap" title="{cov}%">'
            f'<div class="cov-bar" style="width:{cov}%;background:{cov_bar_color}"></div>'
            f'<span class="cov-label">{cov}%</span>'
            f'</div>'
            f'</td>'
            f'<td class="mono">{r["first_date"] or "—"}</td>'
            f'<td class="mono">{r["last_date"] or "—"}</td>'
            f'<td class="num">{r["n_rows"]:,}</td>'
            f'</tr>'
        )

    summary_freeway_html = ""
    for fw in sorted(by_freeway.keys()):
        d = by_freeway[fw]
        col = FREEWAY_COLORS.get(fw, "#888")
        summary_freeway_html += (
            f'<div class="fw-stat">'
            f'<span class="fw-badge" style="background:{col}">{fw}</span>'
            f'<span>{d["in_dataset"]}/{d["total"]} in dataset</span>'
            f'</div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<title>VD 站點總覽</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, "Segoe UI", sans-serif; font-size: 13px; background: #f5f5f5; color: #222; }}
#app {{ display: flex; flex-direction: column; height: 100vh; }}
header {{ background: #1a2332; color: #fff; padding: 10px 16px; display: flex; align-items: center; gap: 16px; flex-shrink: 0; }}
header h1 {{ font-size: 16px; font-weight: 600; }}
.header-meta {{ font-size: 12px; color: #99aabb; }}
.summary-bar {{ background: #fff; border-bottom: 1px solid #ddd; padding: 8px 16px; display: flex; align-items: center; gap: 20px; flex-shrink: 0; flex-wrap: wrap; }}
.fw-stat {{ display: flex; align-items: center; gap: 6px; }}
.fw-badge {{ display: inline-block; padding: 1px 7px; border-radius: 10px; color: #fff; font-size: 11px; font-weight: 600; }}
.total-stat {{ font-size: 13px; color: #555; }}
.main {{ display: flex; flex: 1; overflow: hidden; }}
#map {{ flex: 1; }}
.sidebar {{ width: 680px; display: flex; flex-direction: column; background: #fff; border-left: 1px solid #ddd; overflow: hidden; flex-shrink: 0; }}
.filter-bar {{ padding: 8px 10px; border-bottom: 1px solid #eee; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; background: #fafafa; }}
.filter-bar label {{ font-size: 11px; color: #666; }}
.filter-bar select, .filter-bar input {{ font-size: 12px; padding: 2px 6px; border: 1px solid #ccc; border-radius: 4px; background: #fff; height: 26px; }}
.filter-bar input[type=text] {{ width: 160px; }}
#row-count {{ font-size: 11px; color: #888; margin-left: auto; }}
.table-wrap {{ overflow: auto; flex: 1; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
thead th {{ position: sticky; top: 0; background: #2c3e50; color: #fff; padding: 6px 8px; text-align: left; white-space: nowrap; cursor: pointer; user-select: none; z-index: 2; }}
thead th:hover {{ background: #3d5166; }}
thead th.sort-asc::after {{ content: ' ▲'; }}
thead th.sort-desc::after {{ content: ' ▼'; }}
tbody tr {{ border-bottom: 1px solid #f0f0f0; cursor: pointer; }}
tbody tr:hover {{ background: #e8f4ff; }}
tbody tr.highlighted {{ background: #fffde7; }}
td {{ padding: 4px 8px; vertical-align: middle; white-space: nowrap; }}
.mono {{ font-family: monospace; font-size: 11px; }}
.vdid-cell {{ max-width: 220px; overflow: hidden; text-overflow: ellipsis; }}
.num {{ text-align: right; }}
.cov-bar-wrap {{ display: flex; align-items: center; gap: 4px; min-width: 80px; }}
.cov-bar {{ height: 10px; border-radius: 3px; min-width: 1px; }}
.cov-label {{ font-size: 11px; color: #555; width: 30px; }}
.popup-content {{ font-size: 13px; line-height: 1.7; min-width: 240px; }}
.popup-content h3 {{ font-size: 13px; font-weight: 700; margin-bottom: 6px; word-break: break-all; }}
.popup-content .row {{ display: flex; gap: 8px; }}
.popup-content .key {{ color: #666; width: 80px; flex-shrink: 0; }}
.popup-content .val {{ font-weight: 500; }}
.popup-content .tag {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 11px; font-weight: 600; color: #fff; margin-right: 3px; }}
</style>
</head>
<body>
<div id="app">
  <header>
    <h1>VD 站點總覽</h1>
    <span class="header-meta">來源：tisvcloud.freeway.gov.tw VD.xml ＋ vd_dim.parquet（2024-04-01 ~ 2025-05-31）</span>
  </header>
  <div class="summary-bar">
    <span class="total-stat">靜態 VD 總數（N1/N3/N4/N6）：<b>{n_total}</b>　在資料集中：<b>{n_in_dataset}</b></span>
    {summary_freeway_html}
    <span class="total-stat" style="margin-left:auto;color:#888;font-size:11px">
      覆蓋率 = n_source_days / 426 天（2024-04-01~2025-05-31）
    </span>
  </div>
  <div class="main">
    <div id="map"></div>
    <div class="sidebar">
      <div class="filter-bar">
        <label>路線
          <select id="f-freeway">
            <option value="">全部</option>
            <option>N1</option><option>N3</option><option>N4</option><option>N6</option>
          </select>
        </label>
        <label>方向
          <select id="f-direction">
            <option value="">全部</option>
            <option value="N">北向</option><option value="S">南向</option>
            <option value="E">東向</option><option value="W">西向</option>
          </select>
        </label>
        <label>偵測器類型
          <select id="f-vdtype">
            <option value="">全部</option>
            <option value="1">感應線圈</option>
            <option value="2">微波</option>
            <option value="3">影像辨識</option>
          </select>
        </label>
        <label>位置類型
          <select id="f-loctype">
            <option value="">全部</option>
            <option value="5">主線</option>
            <option value="1">匝道</option>
          </select>
        </label>
        <label>在資料集
          <select id="f-inds">
            <option value="">全部</option>
            <option value="1">是</option>
            <option value="0">否</option>
          </select>
        </label>
        <label><input type="text" id="f-search" placeholder="搜尋 VDID…"></label>
        <span id="row-count"></span>
      </div>
      <div class="table-wrap">
        <table id="vd-table">
          <thead>
            <tr>
              <th data-col="freeway">路線</th>
              <th data-col="direction">方向</th>
              <th data-col="mileage">里程</th>
              <th data-col="vdid">VDID</th>
              <th data-col="vdtype">偵測器</th>
              <th data-col="loctype">位置</th>
              <th data-col="lanes">車道</th>
              <th data-col="section">路段</th>
              <th data-col="inds">in DS</th>
              <th data-col="coverage">覆蓋率</th>
              <th data-col="first">起始日</th>
              <th data-col="last">末日</th>
              <th data-col="rows">列數</th>
            </tr>
          </thead>
          <tbody id="table-body">
            {''.join(table_rows)}
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
const GEOJSON = {geojson};
const FREEWAY_COLORS = {json.dumps(FREEWAY_COLORS)};

// ---- Map ----
const map = L.map('map', {{center: [23.8, 121.0], zoom: 8}});
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>',
  subdomains: 'abcd', maxZoom: 20
}}).addTo(map);

function markerColor(props) {{
  const base = FREEWAY_COLORS[props.freeway] || '#888';
  return props.in_dataset ? base : '#aaa';
}}

function makeCircle(props) {{
  const color = markerColor(props);
  return L.circleMarker([props.lat, props.lon], {{
    radius: props.in_dataset ? 6 : 4,
    fillColor: color,
    color: props.in_dataset ? '#fff' : '#ccc',
    weight: 1,
    fillOpacity: props.in_dataset ? 0.9 : 0.5,
  }});
}}

function popupHtml(p) {{
  const inDs = p.in_dataset
    ? `<span class="tag" style="background:#27ae60">在資料集</span>`
    : `<span class="tag" style="background:#aaa">靜態僅存</span>`;
  const covBar = p.in_dataset
    ? `<div style="background:#eee;border-radius:4px;height:10px;margin-top:4px"><div style="width:${{p.coverage_pct}}%;background:#27ae60;height:10px;border-radius:4px"></div></div><small>${{p.coverage_pct}}% 覆蓋率（${{p.n_days}} 天）</small>`
    : '';
  return `<div class="popup-content">
    <h3>${{p.vdid}}</h3>
    ${{inDs}}
    <div class="row"><span class="key">路線</span><span class="val">${{p.freeway}} ${{p.direction_label}}</span></div>
    <div class="row"><span class="key">里程</span><span class="val">${{p.mileage_km.toFixed(3)}} K（${{p.location_mile}}姉</span></div>
    <div class="row"><span class="key">路段</span><span class="val">${{p.road_section}}</span></div>
    <div class="row"><span class="key">偵測器</span><span class="val">${{p.vd_type}}</span></div>
    <div class="row"><span class="key">位置</span><span class="val">${{p.location_type}}</span></div>
    <div class="row"><span class="key">車道數</span><span class="val">${{p.lane_num}} 條（實際 ${{p.actual_lane_num}}）</span></div>
    <div class="row"><span class="key">偵測方向</span><span class="val">${{p.detection_type}}</span></div>
    <div class="row"><span class="key">LinkID</span><span class="val" style="font-family:monospace;font-size:11px">${{p.link_ids}}</span></div>
    ${{p.in_dataset ? `<div class="row"><span class="key">資料期間</span><span class="val">${{p.first_date}} ~ ${{p.last_date}}</span></div>` : ''}}
    ${{covBar}}
    ${{p.in_dataset ? `<div class="row"><span class="key">總列數</span><span class="val">${{p.n_rows.toLocaleString()}}</span></div>` : ''}}
  </div>`;
}}

const allMarkers = {{}};
const layerGroups = {{}};
for (const fw of ['N1','N3','N4','N6']) layerGroups[fw] = L.layerGroup().addTo(map);

for (const f of GEOJSON.features) {{
  const p = f.properties;
  const m = makeCircle(p);
  m.bindPopup(popupHtml(p), {{maxWidth: 300}});
  m.on('click', () => highlightRow(p.vdid));
  allMarkers[p.vdid] = m;
  layerGroups[p.freeway]?.addLayer(m);
}}

// ---- Table filtering ----
const tbody = document.getElementById('table-body');
const rows = Array.from(tbody.querySelectorAll('tr'));
const rowCount = document.getElementById('row-count');

let activeFilters = {{}};
let searchText = '';
let sortCol = null, sortDir = 1;

function applyFilters() {{
  const fw = document.getElementById('f-freeway').value;
  const dir = document.getElementById('f-direction').value;
  const vdt = document.getElementById('f-vdtype').value;
  const lct = document.getElementById('f-loctype').value;
  const inds = document.getElementById('f-inds').value;
  searchText = document.getElementById('f-search').value.toLowerCase();

  let visible = 0;
  const visibleVdids = new Set();
  rows.forEach(row => {{
    const fwOk = !fw || row.dataset.freeway === fw;
    const dirOk = !dir || row.dataset.direction === dir;
    const vdtOk = !vdt || row.dataset.vdtype === vdt;
    const lctOk = !lct || row.dataset.loctype === lct;
    const indsOk = !inds || row.dataset.inds === inds;
    const searchOk = !searchText || row.dataset.vdid.toLowerCase().includes(searchText);
    const show = fwOk && dirOk && vdtOk && lctOk && indsOk && searchOk;
    row.style.display = show ? '' : 'none';
    if (show) {{ visible++; visibleVdids.add(row.dataset.vdid); }}
  }});
  rowCount.textContent = `顯示 ${{visible}} / ${{rows.length}} 筆`;

  // Update map marker visibility
  for (const [vdid, marker] of Object.entries(allMarkers)) {{
    const show = visibleVdids.has(vdid);
    const fw = marker.options._fw || vdid.split('-')[1];
    const grp = layerGroups[fw];
    if (!grp) continue;
    if (show && !grp.hasLayer(marker)) grp.addLayer(marker);
    else if (!show && grp.hasLayer(marker)) grp.removeLayer(marker);
  }}
}}

// Store freeway on marker for layer management
for (const f of GEOJSON.features) {{
  const m = allMarkers[f.properties.vdid];
  if (m) m.options._fw = f.properties.freeway;
}}

document.querySelectorAll('#filter-bar select, #filter-bar input').forEach(el =>
  el.addEventListener('change', applyFilters));
document.getElementById('f-search').addEventListener('input', applyFilters);
document.querySelectorAll('#f-freeway,#f-direction,#f-vdtype,#f-loctype,#f-inds').forEach(el =>
  el.addEventListener('change', applyFilters));

// ---- Sort ----
document.querySelectorAll('thead th').forEach(th => {{
  th.addEventListener('click', () => {{
    const col = th.dataset.col;
    if (sortCol === col) sortDir *= -1;
    else {{ sortCol = col; sortDir = 1; }}
    document.querySelectorAll('thead th').forEach(t => t.classList.remove('sort-asc','sort-desc'));
    th.classList.add(sortDir === 1 ? 'sort-asc' : 'sort-desc');
    sortTable(col, sortDir);
  }});
}});

const colIndex = {{'freeway':0,'direction':1,'mileage':2,'vdid':3,'vdtype':4,'loctype':5,'lanes':6,'section':7,'inds':8,'coverage':9,'first':10,'last':11,'rows':12}};
function sortTable(col, dir) {{
  const idx = colIndex[col] ?? 0;
  const sorted = [...rows].sort((a, b) => {{
    const av = a.cells[idx]?.textContent.trim() || '';
    const bv = b.cells[idx]?.textContent.trim() || '';
    const an = parseFloat(av.replace(/[^0-9.-]/g,''));
    const bn = parseFloat(bv.replace(/[^0-9.-]/g,''));
    if (!isNaN(an) && !isNaN(bn)) return (an - bn) * dir;
    return av.localeCompare(bv, 'zh-Hant') * dir;
  }});
  tbody.append(...sorted);
}}

// ---- Highlight row from map click ----
function highlightRow(vdid) {{
  rows.forEach(r => r.classList.remove('highlighted'));
  const row = rows.find(r => r.dataset.vdid === vdid);
  if (row) {{
    row.classList.add('highlighted');
    row.scrollIntoView({{behavior:'smooth', block:'center'}});
  }}
}}

// ---- Row click → fly to marker ----
rows.forEach(row => {{
  row.addEventListener('click', () => {{
    const vdid = row.dataset.vdid;
    const marker = allMarkers[vdid];
    if (marker) {{
      const ll = marker.getLatLng();
      map.flyTo(ll, Math.max(map.getZoom(), 14), {{duration: 0.8}});
      marker.openPopup();
    }}
    highlightRow(vdid);
  }});
}});

// Initial count
applyFilters();
</script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build VD station HTML viewer")
    parser.add_argument("--fetch", action="store_true", help="Re-download VD.xml from tisvcloud")
    parser.add_argument(
        "--vd-xml", type=Path, default=VDXML_PATH,
        help=f"Path to VD static XML (default: {VDXML_PATH})"
    )
    parser.add_argument(
        "--vd-dim", type=Path, default=VD_DIM_PATH,
        help=f"Path to vd_dim.parquet (default: {VD_DIM_PATH})"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help=f"Output HTML path (default: {OUTPUT_PATH})"
    )
    args = parser.parse_args()

    if args.fetch or not args.vd_xml.exists():
        args.vd_xml.parent.mkdir(parents=True, exist_ok=True)
        fetch_vd_xml()
        args.vd_xml = VDXML_PATH

    print(f"Parsing {args.vd_xml} ...", flush=True)
    static_records = parse_vd_xml(args.vd_xml)

    print(f"Loading {args.vd_dim} ...", flush=True)
    dim_map = load_vd_dim(args.vd_dim)

    merged = merge(static_records, dim_map)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(merged)
    args.output.write_text(html, encoding="utf-8")
    print(f"Written: {args.output}  ({len(html):,} chars)", flush=True)


if __name__ == "__main__":
    main()
