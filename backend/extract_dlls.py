import pefile
import numpy as np
import os

# -----------------------------
# LOAD DLL FEATURE LIST
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DLL_LIST_PATH = os.path.join(BASE_DIR, "..", "models", "dll_features.txt")

with open(DLL_LIST_PATH, "r", errors="ignore") as f:
    DLL_LIST = [line.strip().lower() for line in f if line.strip()]

# -----------------------------
# DLL FEATURE EXTRACTION
# -----------------------------
def extract_dll_features(file_path):
    pe = pefile.PE(file_path, fast_load=True)
    pe.parse_data_directories()
    imported_dlls = set()

    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            imported_dlls.add(entry.dll.decode(errors="ignore").lower())

    # Create feature vector (length = 838)
    features = np.zeros(len(DLL_LIST), dtype=np.float32)

    for idx, dll in enumerate(DLL_LIST):
        if dll in imported_dlls:
            features[idx] = 1.0
    pe.close()
    return features
