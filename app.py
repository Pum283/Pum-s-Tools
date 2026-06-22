import sys
import os

# Add root folder to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.TextToSpeech.app import main

if __name__ == "__main__":
    main()
