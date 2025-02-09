import tensorflow as tf
import tensorflow_transform as tft

# Daftar numerical fitur pada dataset
NUMERICAL_FEATURES = [
    'Age',
    'Sleep Hours',
    'Physical Activity (hrs/week)',
    'Caffeine Intake (mg/day)',
    'Alcohol Consumption (drinks/week)',
    'Stress Level (1-10)',
    'Heart Rate (bpm during attack)',
    'Breathing Rate (breaths/min)',
    'Sweating Level (1-5)',
    'Therapy Sessions (per month)',
    'Diet Quality (1-10)'
]

CATEGORICAL_FEATURES = [
    'Gender',
    'Occupation',
    'Smoking',
    'Family History of Anxiety',
    'Dizziness',
    'Medication',
    'Recent Major Life Event'
]

# Label key
LABEL_KEY = "Anxiety Category Encoded"

# Fungsi untuk mengubah nama fitur yang sudah ditransformasi
def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"



# Fungsi untuk melakukan preprocessing
def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features.
    
    Args:
        inputs: Dictionary dari feature keys ke raw features.
    
    Returns:
        outputs: Dictionary dari feature keys ke transformed features.
    """
    outputs = {}

    # 1️⃣ Encoding fitur kategorikal terlebih dahulu (mengubah teks jadi angka)
    encoded_categorical_features = {}
    for feature in CATEGORICAL_FEATURES:
        if feature in inputs:
            encoded_categorical_features[feature] = tft.compute_and_apply_vocabulary(
                tf.strings.strip(tf.strings.lower(inputs[feature]))  # Normalisasi teks
            )
    
    # 2️⃣ Gabungkan semua fitur numerik + hasil encoding ke dalam satu dictionary numerik
    all_numeric_features = {**encoded_categorical_features}  # Mulai dengan fitur kategorikal yang sudah diencode
    for feature in NUMERICAL_FEATURES:
        if feature in inputs:
            all_numeric_features[feature] = tf.cast(inputs[feature], tf.float32)  # Pastikan tipe float32
    
    # 3️⃣ Lakukan normalisasi ke semua fitur yang sekarang sudah numerik
    for feature, tensor in all_numeric_features.items():
        outputs[transformed_name(feature)] = tft.scale_to_0_1(tensor)  # Normalisasi ke [0,1]
    
    # 4️⃣ Transformasi label target (pastikan label dalam bentuk integer)
    if LABEL_KEY in inputs:
        outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
