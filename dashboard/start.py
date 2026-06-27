#!/usr/bin/env python3
"""Simple daemon to start Streamlit in the background."""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('/tmp/streamlit_d.log', 'w') as log:
    proc = subprocess.Popen(
        [sys.executable, '-m', 'streamlit', 'run', 'Dashboard_Home.py',
         '--server.port', '8501', '--server.headless', 'true'],
        stdout=log, stderr=log,
        start_new_session=True
    )
    print(f"Streamlit PID: {proc.pid}")
