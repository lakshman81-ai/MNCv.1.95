
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from backend.pipeline.models import TranscriptionResult, AnalysisData
    print("Import successful")
    try:
        t = TranscriptionResult(musicxml="xml", analysis_data=AnalysisData(), midi=b"midi")
        print("Instantiation with midi= passed")
    except TypeError as e:
        print(f"Instantiation with midi= failed: {e}")

    try:
        t = TranscriptionResult(musicxml="xml", analysis_data=AnalysisData(), midi_bytes=b"midi")
        print("Instantiation with midi_bytes= passed")
    except TypeError as e:
        print(f"Instantiation with midi_bytes= failed: {e}")

except ImportError as e:
    print(f"Import failed: {e}")
