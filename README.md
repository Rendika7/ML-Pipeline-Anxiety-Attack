# Submission 1: `Pengembangan dan Pengoperasian Sistem Machine Learning untuk Prediksi Keparahan Serangan Kecemasan` | *Development and Operation of a Machine Learning System for Predicting Anxiety Attack Severity*

**Nama**: **Rendika Nurhartanto Suharto**  
**Username dicoding**: ***RENDIKA NURHARTANTO SUHARTO***


| **Aspek**             | **Deskripsi**                                                                                                                                  |
|-----------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|
| **Dataset**           | **Name**: [Anxiety Attack: Factors, Symptoms, and Severity](https://www.kaggle.com/datasets/ashaychoudhary/anxiety-attack-factors-symptoms-and-severity/data) <br> **Data Format**: CSV (*Comma-Separated Values*) <br> **Size**: *12,000+ records* <br> **Usability in Kaggle**: *10.00* <br> **Description**: This dataset contains over **12,000 records** detailing various factors related to anxiety attacks, including demographics, lifestyle habits, stress levels, and physiological responses. It is designed for **data analysis**, **machine learning**, and **mental health research** to explore patterns, triggers, and potential correlations in anxiety disorders. |
| **Masalah**           | Dalam beberapa tahun terakhir, kecemasan telah menjadi masalah kesehatan mental utama, dengan sekitar *264 juta orang* di dunia mengalaminya, dan prevalensinya terus meningkat setiap tahunnya. Serangan kecemasan dapat terjadi mendadak dengan gejala seperti detak jantung cepat, napas pendek, berkeringat, dan rasa panik, yang dapat mengganggu aktivitas sehari-hari. Oleh karena itu, penting untuk memahami faktor-faktor yang mempengaruhi tingkat keparahan serangan, seperti demografi, gaya hidup, dan kondisi psikologis, guna mengembangkan pendekatan berbasis data yang efektif untuk memprediksi dan menangani kecemasan dengan lebih tepat sasaran. |
| **Solusi Machine Learning** | Solusi yang dikembangkan adalah model **machine learning** berbasis klasifikasi untuk memprediksi tingkat keparahan serangan kecemasan. Dengan menggunakan dataset yang mencakup faktor-faktor seperti demografi, gaya hidup, kondisi psikologis, dan indikator kesehatan, model ini dapat mengidentifikasi pola dan hubungan antara fitur-fitur tersebut dengan tingkat keparahan kecemasan. Model klasifikasi ini bertujuan untuk memberikan prediksi yang akurat mengenai tingkat keparahan serangan, mulai dari kecemasan ringan hingga serangan panik, yang dapat digunakan untuk membantu penanganan yang lebih tepat dan terarah. |
| **Metode Pengolahan** | Metode pengolahan data dimulai dengan pembersihan dataset, menghapus kolom yang tidak relevan dan memeriksa data duplikat. Kolom `Severity of Anxiety Attack (1-10)` dikategorikan menjadi empat tingkat keparahan kecemasan (*Mild*, *Moderate*, *Severe*, *Panic*), yang kemudian diubah menjadi nilai numerik melalui **label encoding**. Data yang telah diproses disimpan dalam format CSV untuk digunakan dalam pelatihan model. Modul transformasi `anxiety_transform.py` menggunakan **TensorFlow Transform (TFT)** untuk memproses fitur, dengan melakukan encoding pada data kategorikal seperti `Gender` dan `Occupation`, serta mengubah label target menjadi integer menggunakan `tf.cast`. Hasil transformasi disimpan dalam dictionary dan siap digunakan untuk tahap pelatihan, memastikan data memiliki format yang sesuai untuk model **machine learning**. |
| **Arsitektur Model**  | Arsitektur model dalam proyek ini menggunakan pendekatan **deep learning** dengan **Keras**, dioptimalkan melalui tuning hyperparameter menggunakan **Keras Tuner** dengan metode *Random Search*. Tuning dilakukan pada hyperparameter seperti jumlah unit di setiap lapisan dense (**unit_1**, **unit_2**, **unit_3**), tingkat **dropout** (*dropout_1*, *dropout_2*), dan **learning rate** (*learning_rate*). Model ini dirancang dengan tiga lapisan dense dan fungsi aktivasi **ReLU** untuk meningkatkan akurasi dan mencegah overfitting. Arsitektur ini memungkinkan model untuk mempelajari pola kompleks dalam data, menghasilkan prediksi yang lebih akurat mengenai tingkat keparahan serangan kecemasan. |
| **Evaluasi Model**    | Setelah pelatihan, model diuji menggunakan data validasi untuk mengevaluasi kinerjanya. **Accuracy**, **Precision**, **Recall**, dan **F1-Score** digunakan sebagai metrik untuk menilai performa model dalam mengklasifikasikan tingkat keparahan kecemasan. Hasil evaluasi menunjukkan bahwa model memiliki performa yang baik dengan **F1-Score** mencapai *0.85* untuk kategori *Severe* dan *Panic*, menunjukkan kemampuan model dalam menangani data yang tidak seimbang. |
| **Monitoring**        | Selama fase pemantauan, model diuji secara terus-menerus menggunakan data baru untuk memastikan performa tetap optimal. Proses ini melibatkan **cross-validation** dan penyesuaian hyperparameter jika diperlukan. Selain itu, metrik seperti **Loss** dan **Accuracy** digunakan untuk memonitor proses pembelajaran dan memastikan model tidak overfit terhadap data pelatihan. Hasil pemantauan ini digunakan untuk melakukan perbaikan lebih lanjut pada model, memastikan model tetap efektif dalam prediksi serangan kecemasan. |

# How to Use

## Cloning the Repository
To begin, clone the repository by running the following command in your terminal or command prompt:
```
git clone https://github.com/Rendika7/ML-Pipeline-Anxiety-Attack.git
```

Once the repository is cloned, navigate to the project folder:
```
cd ML-Pipeline-Anxiety-Attack
```

Now you can proceed with the steps below for setting up and running the project.

## Creating a Virtual Environment
- Create a virtual environment by running the following command:
    ```
    conda create -n tfx-beam python=3.9.21 -y
    ```

- Activate the environment with the command:
    ```
    conda activate tfx-beam
    ```

## Installing Requirements
- Make sure you are in the `proyek-akhir-mlops` environment, then run:
    ```
    pip install -r requirements.txt
    ```

- If you want to apply Python clean code practices, install with the command:
    ```
    pip install autopep8 pylint
    ```

## Running the Pipeline

- Open the `notebook.ipynb` file.

- If successful, the `output/serving model` folder will be created.

## Running the Machine Learning Model on Railway

- Create an account on Railway, install the Railway CLI, log in to your Railway account, create a new project, and connect it to the CLI using the guide [here](https://www.dicoding.com/academies/443/discussions/225535).

- After that, run the command `railway up` to push and build Docker to Railway.

- If the deploy process is successful, proceed to create a domain as per the guide.

- Access the domain by appending `/v1/models/cc-model/metadata` to the URL.

## Monitoring the Model

### Using Prometheus
- Run the following two commands to build and run the monitoring container:
    ```
    docker build -t cc-monitoring .\monitoring\
    docker run -p 9090:9090 cc-monitoring
    ```

- Access the Prometheus dashboard at the following link:
    ```
    http://localhost:9090/
    ```

- Use the following query to monitor:
    ```
    :tensorflow:serving:request_count
    ```

### Using Grafana (Windows Tutorial)
- Install Grafana by following the instructions [here](https://grafana.com/grafana/download?platform=windows).

- Start Grafana by executing `grafana-server.exe`.

- Access Grafana in your browser by visiting `http://localhost:3000/`.

- Log in with the username and password `admin`.

- Add a new data source and choose Prometheus.

- In the `connection` section, enter `http://localhost:9090`.

## Making Predictions
- Open the `test.ipynb` file.
- Run the last file in the notebook.