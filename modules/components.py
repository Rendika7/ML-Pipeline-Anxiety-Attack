# Import library
import os
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen, 
    StatisticsGen, 
    SchemaGen, 
    ExampleValidator, 
    Transform, 
    Trainer,
    Tuner,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2 
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy)

# Fungsi untuk melakukan inisialisasi components
def init_components(config):

    """Returns tfx components for the pipeline.
 
    Args:
        data_dir (str): Directory containing the dataset.
        transform_module (str): Path to the transform module.
        tuner_module (str): Path to the tuner module.
        training_module (str): Path to the training module.
        training_steps (int): Number of training steps.
        eval_steps (int): Number of evaluation steps.
        serving_model_dir (str): Directory to save the serving
 
    Returns:
        components: Tuple of TFX components.
    """ 
    
    # 1. Membagi dataset dengan perbandingan 9:1: Data dibagi menjadi dua bagian: 90% untuk pelatihan (train) dan 10% untuk evaluasi (eval).
    output = example_gen_pb2.Output(
        split_config = example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=9),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=1)
        ])
    )
 
    # 2. Komponen example gen: Menggunakan CsvExampleGen untuk menghasilkan contoh data dari file CSV yang terletak pada direktori yang ditentukan dalam konfigurasi.
    example_gen = CsvExampleGen(
        input_base=config["DATA_ROOT"], 
        output_config=output
    )
    
    # 3. Komponen statistics gen: StatisticsGen digunakan untuk menghasilkan statistik dari data, yang kemudian akan digunakan untuk validasi dan pembuatan schema.
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"]   
    )
    
    # 4. Komponen schema gen: SchemaGen menghasilkan schema dari statistik yang dihasilkan oleh StatisticsGen, yang berfungsi untuk menentukan struktur data yang benar.
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"]
    )
    
    # 5. Komponen example validator: ExampleValidator memastikan bahwa data yang ada sesuai dengan schema yang telah dibuat dan tidak ada data yang invalid.
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )
    
    # 6. Komponen transform: Transform digunakan untuk melakukan transformasi pada data menggunakan modul yang sudah ditentukan (misalnya modul transform.py).
    transform  = Transform(
        examples=example_gen.outputs['examples'],
        schema= schema_gen.outputs['schema'],
        module_file=os.path.abspath(config["transform_module"])
    )
    
    print("Transform Module Path:", os.path.abspath(config["transform_module"]))
    assert os.path.exists(config["transform_module"]), "Transform module file not found!"


    # 7. Komponen tuner: Tuner digunakan untuk mencari hyperparameters terbaik untuk model. Ini menggunakan modul yang sudah ditentukan (misalnya tuner.py) untuk mencari kombinasi hyperparameter yang optimal.
    tuner = Tuner(
        module_file=os.path.abspath(config["tuner_module"]),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'], 
            num_steps=config["training_steps"]),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'], 
            num_steps=config["eval_steps"]),
    )
    
    print("Tuner Module Path:", os.path.abspath(config["tuner_module"]))
    assert os.path.exists(config["tuner_module"]), "Tuner module file not found!"
    
    # 8. Komponen trainer: Trainer melatih model dengan menggunakan data yang sudah ditransformasi dan hyperparameter terbaik dari Tuner.
    trainer  = Trainer(
        module_file=os.path.abspath(config["training_module"]),
        examples = transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        hyperparameters=tuner.outputs['best_hyperparameters'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=config["training_steps"]),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'], 
            num_steps=config["eval_steps"])
    )

    print("Training Module Path:", os.path.abspath(config["training_module"]))
    assert os.path.exists(config["training_module"]), "Training module file not found!"
    
    # 9. Komponen model resolver: ModelResolver digunakan untuk memilih model terbaik yang sudah diberkati (blessed) menggunakan strategi LatestBlessedModelStrategy, yang memastikan bahwa model yang digunakan adalah yang terbaru dan terbaik.
    model_resolver = Resolver(
        strategy_class= LatestBlessedModelStrategy,
        model = Channel(type=Model),
        model_blessing = Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

    metrics_specs = [
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='AUC'),
            tfma.MetricConfig(class_name="Precision"),  # Removed invalid config
            tfma.MetricConfig(class_name="Recall"),  # Removed invalid config
            tfma.MetricConfig(class_name="ExampleCount"),
            tfma.MetricConfig(
                class_name='CategoricalAccuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value': 0.9}),
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value': 0.0001})
                )
            )
        ])
    ]



    
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="Anxiety Category Encoded")],
        slicing_specs=[
            tfma.SlicingSpec(),
            ],
        metrics_specs=metrics_specs
    )
    
    
    # 10. Komponen evaluator: Evaluator digunakan untuk mengevaluasi kinerja model dengan menggunakan metrik yang telah ditentukan seperti AUC, precision, recall, dan accuracy.
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)
    
    # 11. Komponen pusher: Pusher bertanggung jawab untuk mendorong (push) model yang telah terlatih dan dievaluasi ke sistem produksi atau direktori model yang dapat digunakan untuk penyajian.
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=config["serving_model_dir"]
            )
        ),
    )
    
    # Fungsi ini mengembalikan semua komponen yang sudah dipersiapkan untuk digunakan dalam pipeline TFX.
    components = (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher
    )
    
    # Mengembalikan komponen: Pada akhirnya, fungsi ini mengembalikan komponen-komponen tersebut dalam bentuk tuple, siap untuk digunakan dalam pipeline.
    return components
