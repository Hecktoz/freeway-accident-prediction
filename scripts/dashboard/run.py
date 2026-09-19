# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "gradio>=4.36",
#     "folium>=0.16",
#     "branca>=0.7",
#     "pandas",
#     "pyarrow",
#     "duckdb",
#     "numpy",
#     "plotly>=5.20",
#     "torch",
#     "lxml",
#     "matplotlib",
#     "scikit-learn",
#     "tqdm",
# ]
# ///
"""高速公路事故速度預測分析儀表板 — uv-run launcher (Gradio).

Usage:
    uv run scripts/dashboard/run.py
    uv run scripts/dashboard/run.py -- --server-port 7861
    uv run scripts/dashboard/run.py -- --share          # public Gradio link

In Colab:
    !pip install uv
    !uv run scripts/dashboard/run.py -- --share
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch the Gradio dashboard",
        add_help=True,
    )
    parser.add_argument("--server-name", default="127.0.0.1",
                        help="Bind address (use 0.0.0.0 for network access)")
    parser.add_argument("--server-port", type=int, default=7860,
                        help="Port to listen on")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override data root directory")
    parser.add_argument("--models-root", type=str, default=None,
                        help="Override models root directory")
    args, _ = parser.parse_known_args()

    from scripts.dashboard.loaders import load_all
    from scripts.dashboard.app import build_app

    data_root = Path(args.data_root) if args.data_root else None
    models_root = Path(args.models_root) if args.models_root else None

    app_state = load_all(data_root=data_root, models_root=models_root)
    demo = build_app(app_state)

    css = (
        ".gradio-container { max-width: 1600px; margin: 0 auto; }"
        # Ensure Plotly plot wrappers stretch to full column width inside nested tabs.
        " .gradio-container .plot-container { width: 100% !important; }"
    )
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=True,
        css=css,
    )


if __name__ == "__main__":
    main()
