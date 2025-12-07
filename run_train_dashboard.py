# ─────────────────────────────────────────────────────────────
# File: run_train_dashboard.py
# Launch the Training Dashboard
#
# Usage:
#   python run_train_dashboard.py
#
# Then open http://localhost:8765 in your browser
# ─────────────────────────────────────────────────────────────

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from traindashboard.server import main

if __name__ == "__main__":
    main()
