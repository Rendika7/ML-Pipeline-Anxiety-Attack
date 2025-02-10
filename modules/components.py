"""
components.py

Modul ini berisi fungsi untuk inisialisasi komponen TFX dalam pipeline ML.
"""

import os
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator,
    Transform, Trainer, Tuner, Evaluator, Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy
)

def init_components(config):
    """
    Inisialisasi dan mengembalikan komponen TFX untuk pipeline.

    Args:
        config (dict): Konfigurasi pipeline yang mencakup path modul, jumlah langkah pelatihan,
                       path data, dan direktori model serving.

    Returns:
        tuple: Komponen-komponen TFX yang siap digunakan dalam pipeline.
    """

    # 1. Konfigurasi split dataset: 90% training, 10% evaluasi
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=9),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=1)
        ])
    )

    # 2. Komponen ExampleGen
    example_gen = CsvExampleGen(input_base=config["DATA_ROOT"], output_config=output)

    # 3. Komponen StatisticsGen
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    # 4. Komponen SchemaGen
    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])

    # 5. Komponen ExampleValidator
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"]
    )

    # 6. Komponen Transform
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.abspath(config["transform_module"])
    )

    # Validasi path module
    assert os.path.exists(config["transform_module"]), "Transform module file not found!"

    # 7. Komponen Tuner
    tuner = Tuner(
        module_file=os.path.abspath(config["tuner_module"]),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=config["training_steps"]),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=config["eval_steps"])
    )

    assert os.path.exists(config["tuner_module"]), "Tuner module file not found!"

    # 8. Komponen Trainer
    trainer = Trainer(
        module_file=os.path.abspath(config["training_module"]),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        hyperparameters=tuner.outputs["best_hyperparameters"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=config["training_steps"]),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=config["eval_steps"])
    )

    assert os.path.exists(config["training_module"]), "Training module file not found!"

    # 9. Komponen Model Resolver
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id("Latest_blessed_model_resolver")

    # 10. Konfigurasi Evaluator
    metrics_specs = [
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name="AUC"),
            tfma.MetricConfig(class_name="Precision"),
            tfma.MetricConfig(class_name="Recall"),
            tfma.MetricConfig(class_name="ExampleCount"),
            tfma.MetricConfig(
                class_name="SparseCategoricalAccuracy",
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(lower_bound={"value": 0.2}),
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={"value": 0.0001}
                    )
                )
            )
        ])
    ]

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="Anxiety Category Encoded")],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=metrics_specs
    )

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config
    )

    # 11. Komponen Pusher
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=config["serving_model_dir"]
            )
        )
    )

    # Mengembalikan tuple komponen untuk pipeline
    return (
        example_gen, statistics_gen, schema_gen, example_validator,
        transform, tuner, trainer, model_resolver, evaluator, pusher
    )
