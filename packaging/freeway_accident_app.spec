# PyInstaller spec for the freeway accident impact GUI.
# Build command (from repo root, with venv active):
#   pyinstaller packaging/freeway_accident_app.spec
#
# Output: dist/freeway_accident_app/  (one-folder bundle, ~300 MB)
# Delivery: copy the entire dist/freeway_accident_app/ folder to the workstation,
#           then double-click freeway_accident_app.exe.
#
# Bundle includes:
#   - All Python code and dependencies
#   - Champion checkpoint (output/models/.../best_model.pth)
#   - Historical speed profile (data/historical_speed_profile_5m/)
#   - VD dimension table (data/metadata/vd_dim.parquet)
#   - Champion spec (reports/v1_champion.json)
#   - Time-progressive specs (reports/feature_time_grouping.md)
#
# NOT bundled (stays on Drive, too large):
#   - vd_raw_lane_1m_full/ (~100 GB)
#   - vd_station_1m/ (~9 GB)
#   - vd_station_5m/ (~2.7 GB)
#   - accident_samples_* training artifacts

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent

a = Analysis(
    [str(ROOT / "scripts" / "gui" / "app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        # Champion and time-progressive checkpoint specs
        (str(ROOT / "reports" / "v1_champion.json"), "reports"),
        # Historical speed profile (partition layout preserved)
        (str(ROOT / "data" / "historical_speed_profile_5m"), "data/historical_speed_profile_5m"),
        # VD dimension metadata
        (str(ROOT / "data" / "metadata"), "data/metadata"),
        # Champion t15 checkpoint
        (str(ROOT / "output" / "models" / "unet_feature_fusion" / "20260521T110235Z"),
         "output/models/unet_feature_fusion/20260521T110235Z"),
        # t5 checkpoint
        (str(ROOT / "output" / "models" / "unet_feature_fusion" / "20260521T150048Z"),
         "output/models/unet_feature_fusion/20260521T150048Z"),
        # t0 checkpoint
        (str(ROOT / "output" / "models" / "unet_feature_fusion" / "20260521T150042Z"),
         "output/models/unet_feature_fusion/20260521T150042Z"),
        # Gradio static assets
        *collect_data_files("gradio"),
    ],
    hiddenimports=[
        "scripts.lib.inference",
        "scripts.lib.windows",
        "scripts.lib.model_architectures",
        "scripts.lib.data_loaders",
        "scripts.lib.live_vd",
        "scripts.lib.calendar",
        "scripts.lib.visualization",
        *collect_submodules("torch"),
        *collect_submodules("pyarrow"),
        *collect_submodules("gradio"),
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch.cuda",          # CPU-only workstation
        "torch.distributed",
        "torch.backends.cuda",
        "torch._dynamo",
        "matplotlib.tests",
        "numpy.testing",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="freeway_accident_app",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,   # no console window; errors go to log file
    icon=None,       # add a .ico here if one becomes available
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="freeway_accident_app",
)