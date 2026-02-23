#!/bin/bash
# WaveCast Production Dashboard
# Launch: bash scripts/run_dashboard.sh
cd /home/musicofhel/wavecast
source .venv/bin/activate
streamlit run dashboard/app.py --server.port 8501 --server.address localhost --server.headless true
