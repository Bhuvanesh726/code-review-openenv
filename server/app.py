# This file is required by openenv validate for multi-mode deployment.
# The actual server logic is in server.py at the project root.
# This re-exports the FastAPI app instance.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app

__all__ = ["app"]
