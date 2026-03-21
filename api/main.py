"""Vercel serverless function wrapper for the main adaptive learning API."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import app
