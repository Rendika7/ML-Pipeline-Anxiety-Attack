# Import library
import tensorflow as tf
import keras_tuner as kt
import tensorflow_transform as tft
import keras_tuner as kt
from tfx.v1.components import TunerFnResult
from tfx.components.trainer.fn_args_utils import FnArgs
from anxiety_trainer import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, transformed_name, input_fn


# Model builder for hyperparameter tuning
def model_builder(hyperparameters):
    """
    This function defines a Keras model and returns the model as a
    Keras object.
    """

    input_features = []

    for key in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(key))
        )
        
    for key in CATEGORICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(key))
        )


    concatenate = tf.keras.layers.concatenate(input_features)

    # Hyperparameter yang lebih luas untuk optimasi lebih baik
    unit_1 = hyperparameters.Int('unit_1', min_value=128, max_value=512, step=64)
    dropout_1 = hyperparameters.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
    
    unit_2 = hyperparameters.Int('unit_2', min_value=64, max_value=256, step=32)
    dropout_2 = hyperparameters.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)

    unit_3 = hyperparameters.Int('unit_3', min_value=32, max_value=128, step=32)
    dropout_3 = hyperparameters.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.1)

    learning_rate = hyperparameters.Choice('learning_rate', [0.0001, 0.0005, 0.001, 0.005])

    # Membangun arsitektur model
    deep = tf.keras.layers.Dense(unit_1, activation="relu")(concatenate)
    deep = tf.keras.layers.Dropout(dropout_1)(deep)

    deep = tf.keras.layers.Dense(unit_2, activation="relu")(deep)
    deep = tf.keras.layers.Dropout(dropout_2)(deep)

    deep = tf.keras.layers.Dense(unit_3, activation="relu")(deep)
    deep = tf.keras.layers.Dropout(dropout_3)(deep)

    outputs = tf.keras.layers.Dense(4, activation="softmax")(deep)  # 4 kelas untuk klasifikasi

    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)

    # Kompilasi model dengan optimizer yang dituning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    return model
    
# Fungsi tuner untuk tuning hyperparameter
def tuner_fn(fn_args: FnArgs):
    """
    Melakukan tuning hyperparameter menggunakan Keras Tuner.
    """

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Ambil dataset pelatihan dan evaluasi
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=32)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=32)

    # Inisialisasi RandomSearch tuner
    tuner = kt.RandomSearch(
        model_builder,
        objective='val_sparse_categorical_accuracy',  # Optimasi berdasarkan akurasi validasi
        max_trials=10,
        executions_per_trial=2,
        directory=fn_args.working_dir,
        project_name='anxiety_severity_tuner'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": 8
        }
    )
