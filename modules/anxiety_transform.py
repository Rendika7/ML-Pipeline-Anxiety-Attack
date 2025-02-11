"""
anxiety_transform.py

Modul ini menangani transformasi fitur untuk preprocessing data
menggunakan TensorFlow Transform (TFT).
"""

import tensorflow as tf
import tensorflow_transform as tft

# Daftar fitur pada dataset yang perlu di-encode
FEATURES = ['Age', 'Gender', 'Occupation', 'Sleep Hours',
            'Physical Activity (hrs/week)', 'Caffeine Intake (mg/day)',
            'Alcohol Consumption (drinks/week)', 'Smoking',
            'Family History of Anxiety', 'Stress Level (1-10)',
            'Heart Rate (bpm during attack)', 'Breathing Rate (breaths/min)',
            'Sweating Level (1-5)', 'Dizziness', 'Medication',
            'Therapy Sessions (per month)', 'Recent Major Life Event',
            'Diet Quality (1-10)']

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

    # 1️⃣ Encoding semua fitur menjadi kategorikal (menggunakan vocabulary encoding)
    for feature in FEATURES:
        if feature in inputs:
            # Cek apakah fitur berupa string dan lakukan encoding
            if isinstance(inputs[feature], tf.Tensor) and inputs[feature].dtype == tf.string:
                encoded_feature = tft.compute_and_apply_vocabulary(
                    tf.strings.strip(tf.strings.lower(inputs[feature]))
                )
                outputs[transformed_name(feature)] = encoded_feature
            else:
                # Jika bukan string, biarkan saja tanpa transformasi
                outputs[transformed_name(feature)] = inputs[feature]

    # 2️⃣ Transformasi label target menjadi integer (label tetap diproses seperti semula)
    if LABEL_KEY in inputs:
        transformed_label = tf.cast(inputs[LABEL_KEY], tf.int64)
        outputs[transformed_name(LABEL_KEY)] = transformed_label

    return outputs
