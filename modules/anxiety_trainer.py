# Import library
import os
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import tensorflow_transform as tft
from tfx.v1.components import TunerFnResult
from tfx.components.trainer.fn_args_utils import FnArgs
from anxiety_transform import (
    LABEL_KEY,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    transformed_name,
)

# Fungsi untuk membuat model dengan hyperparameter terbaik dari tuner
def get_model(hyperparameters, show_summary=True):
    """
    Membuat model dengan hyperparameter terbaik dari tuner.
    """

    input_features = []

    for feature in NUMERICAL_FEATURES:
        input_features.append(tf.keras.Input(shape=(1,), name=transformed_name(feature)))
        
    for feature in CATEGORICAL_FEATURES:
        input_features.append(tf.keras.Input(shape=(1,), name=transformed_name(feature)))

    concatenate = tf.keras.layers.concatenate(input_features)

    # Ambil hyperparameter terbaik dari tuner
    unit_1 = hyperparameters.get('unit_1', 128)
    dropout_1 = hyperparameters.get('dropout_1', 0.2)
    unit_2 = hyperparameters.get('unit_2', 64)
    dropout_2 = hyperparameters.get('dropout_2', 0.2)
    unit_3 = hyperparameters.get('unit_3', 32)
    dropout_3 = hyperparameters.get('dropout_3', 0.2)
    learning_rate = hyperparameters.get('learning_rate', 0.001)

    # Lapisan Dense berdasarkan hyperparameter yang dituning
    deep = tf.keras.layers.Dense(unit_1, activation="relu")(concatenate)
    deep = tf.keras.layers.Dropout(dropout_1)(deep)

    deep = tf.keras.layers.Dense(unit_2, activation="relu")(deep)
    deep = tf.keras.layers.Dropout(dropout_2)(deep)

    deep = tf.keras.layers.Dense(unit_3, activation="relu")(deep)
    deep = tf.keras.layers.Dropout(dropout_3)(deep)

    # Output layer untuk klasifikasi multi-kelas
    outputs = tf.keras.layers.Dense(4, activation="softmax")(deep)

    # Buat model
    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)

    # Kompilasi model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    if show_summary:
        model.summary()

    return model

# Fungsi untuk membaca data yang sudah di kompres
def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

# Fungsi untuk mendapatkan fitur yang sudah di transform
def get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )

        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn

# Fungsi untuk membuat dataset
def input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Generates features and labels for tuning/training."""
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset

def run_fn(fn_args):
    """
    Fungsi utama untuk melatih model berdasarkan hasil tuning dari tuner.
    """

    # Load hasil transformasi fitur
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Ambil hyperparameters terbaik dari tuner
    hyperparameters = fn_args.hyperparameters

    # Ambil dataset pelatihan dan evaluasi
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=64)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=64)

    # Buat model dengan hyperparameter terbaik
    model = get_model(hyperparameters)

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq="batch")
    
    # Tambahkan callback untuk optimalisasi training
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=8, restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001
    )
    
    # Latih model
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback, early_stopping, reduce_lr],
        epochs=10
    )

    # Simpan model untuk serving
    signatures = {
        "serving_default": get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }

    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
