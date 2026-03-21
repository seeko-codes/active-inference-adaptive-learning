"""Vercel serverless function wrapper for the problem browser."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from browse_server import app
