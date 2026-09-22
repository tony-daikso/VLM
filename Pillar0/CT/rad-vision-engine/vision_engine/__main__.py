"""
Entry point for vision-engine package.
Allows running with: python -m vision_engine
"""

from .cli.main import main

if __name__ == "__main__":
    exit(main())
