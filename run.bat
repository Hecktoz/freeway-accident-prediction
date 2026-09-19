@echo off
REM 高速公路事故影響預測 — 一鍵啟動
REM 雙擊此檔案即可開啟 GUI（需要已安裝 uv）

cd /d "%~dp0"
uv run scripts\gui\app.py --data-root data
