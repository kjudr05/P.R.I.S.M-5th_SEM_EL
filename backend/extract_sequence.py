import numpy as np

MAX_LEN = 2000  # ⚠️ adjust if your model expects different

def extract_sequence_features(file_path):
    with open(file_path, "rb") as f:
        data = f.read()

    seq = list(data[:MAX_LEN])

    if len(seq) < MAX_LEN:
        seq += [0] * (MAX_LEN - len(seq))

    return np.array(seq, dtype=np.int32).reshape(1, -1)
