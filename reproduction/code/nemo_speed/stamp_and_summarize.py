"""Compatibility entrypoint; the single implementation lives in inference_speed."""
from pathlib import Path
import runpy
import sys
root = Path(__file__).resolve().parent.parent / 'inference_speed'
sys.path.insert(0, str(root))
runpy.run_path(str(root / 'stamp_and_summarize.py'), run_name='__main__')
