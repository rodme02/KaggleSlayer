#!/usr/bin/env python3
"""
Convenience entry point for the new KaggleSlayer pipeline.
This is a wrapper that calls the refactored pipeline.
"""

import sys
from pathlib import Path

# Import and run the new pipeline script
from scripts.run_pipeline import main

if __name__ == "__main__":
    main()