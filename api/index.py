import os
import sys

# Ensure root directory is on the Python path so app can resolve models, templates, and modules
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from app import app

# Vercel entrypoint
app = app
