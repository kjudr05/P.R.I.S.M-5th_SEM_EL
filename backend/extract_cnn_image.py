import numpy as np

IMG_SIZE = 64

def extract_cnn_features(file_path):
    with open(file_path, "rb") as f:
        byte_data = f.read()

    byte_arr = np.frombuffer(byte_data, dtype=np.uint8)

    needed = IMG_SIZE * IMG_SIZE
    if len(byte_arr) < needed:
        byte_arr = np.pad(byte_arr, (0, needed - len(byte_arr)), mode="constant")
    else:
        byte_arr = byte_arr[:needed]

    # (64, 64)
    image = byte_arr.reshape((IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0

    # 🔥 ADD CHANNEL
    image = image.reshape((1, IMG_SIZE, IMG_SIZE, 1))

    # 🔥 ADD TIME DIMENSION (THIS IS THE KEY)
    image = np.expand_dims(image, axis=1)

    # FINAL SHAPE: (1, 1, 64, 64, 1)
    return image
