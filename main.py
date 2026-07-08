import sys
from pathlib import Path

# Ensure the root of the students_code workspace is in the import path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from src.main import main

if __name__ == "__main__":
    main()
