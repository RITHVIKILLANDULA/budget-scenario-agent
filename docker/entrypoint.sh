#!/bin/sh
set -e
python scripts/seed_db.py --wait 60
exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501 \
     --server.headless=true --browser.gatherUsageStats=false
