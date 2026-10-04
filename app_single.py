"""Single-file entry point — runs the full dashboard implemented in app.py.

Some runners expect a single self-contained file; this alias executes the
modular app unchanged so behaviour is guaranteed identical:

    streamlit run app_single.py          # == streamlit run app.py
"""
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

with open(os.path.join(_DIR, "app.py"), "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), "app.py", "exec"))