"""
anxiety_transform.py

Modul ini menangani transformasi fitur untuk preprocessing data
menggunakan TensorFlow Transform (TFT).
"""

import tensorflow as tf
import tensorflow_transform as tft

# Daftar numerical fitur pada dataset
NUMERICAL_FEATURES = [
    "Age",
    "Sleep Hours",
    "Physical Activity (hrs/week)",
    "Caffeine Intake (mg/day)",
    "Alcohol Consumption (drinks/week)",
    "Stress Level (1-10)",
    "Heart Rate (bpm during attack)",
    "Breathing Rate (breaths/min)",
    "Sweating Level (1-5)",
    "Therapy Sessions (per month)",
    "Diet Quality (1-10)",
]

# Daftar categorical fitur pada dataset
CATEGORICAL_FEATURES = [
    "Gender",
    "Occupation",
    "Smoking",
    "Family History of Anxiety",
    "Dizziness",
    "Medication",
    "Recent Major Life Event",
]

# Label key
LABEL_KEY = "Anxiety Category Encoded"

def transformed_name(key):
    """
    Menambahkan suffix '_xf' untuk fitur yang telah ditransformasikan.

    Args:
        key (str): Nama fitur sebelum transformasi.

    Returns:
        str: Nama fitur setelah transformasi.
    """
    return f"{key}_xf"

def preprocessing_fn(inputs):
    """
    Melakukan preprocessing pada fitur input.

    Args:
        inputs (dict): Dictionary dari feature keys ke raw features.

    Returns:
        dict: Dictionary dari feature keys ke transformed features.
    """
    outputs = {}

    # 1️⃣ Encoding fitur kategorikal menjadi integer (menggunakan vocabulary encoding)
    encoded_categorical_features = {
        feature: tft.compute_and_apply_vocabulary(
            tf.strings.strip(tf.strings.lower(inputs[feature]))
        )
        for feature in CATEGORICAL_FEATURES
        if feature in inputs
    }

    # 2️⃣ Gabungkan semua fitur numerik dan fitur kategorikal yang telah dienkode
    all_numeric_features = {**encoded_categorical_features}
    for feature in NUMERICAL_FEATURES:
        if feature in inputs:
            all_numeric_features[feature] = tf.cast(inputs[feature], tf.float32)

    # 3️⃣ Normalisasi semua fitur numerik agar berada dalam rentang [0,1]
    for feature, tensor in all_numeric_features.items():
        outputs[transformed_name(feature)] = tft.scale_to_0_1(tensor)

    # 4️⃣ Transformasi label target menjadi integer
    if LABEL_KEY in inputs:
        outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
