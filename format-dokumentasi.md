# Submission 1: `Pengembangan dan Pengoperasian Sistem Machine Learning untuk Prediksi Keparahan Serangan Kecemasan` | *Development and Operation of a Machine Learning System for Predicting Anxiety Attack Severity*

**Nama**: Rendika Nurhartanto Suharto  
**Username dicoding**: RENDIKA NURHARTANTO SUHARTO


| **Aspek**               | **Deskripsi**                                                                                                                                  |
|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| **📂 Dataset**          | **🔹 Name**: [Anxiety Attack: Factors, Symptoms, and Severity](https://www.kaggle.com/datasets/ashaychoudhary/anxiety-attack-factors-symptoms-and-severity/data)  <br> **📄 Data Format**: `CSV` (*Comma-Separated Values*)  <br> **📊 Size**: *12,000+ records*  <br> **⭐ Usability in Kaggle**: `10.00`  <br> **📌 Description**: Dataset ini berisi lebih dari **12.000 data** yang mencakup berbagai faktor terkait *serangan kecemasan*, termasuk **demografi, kebiasaan hidup, tingkat stres,** dan **respon fisiologis**. Dataset ini dirancang untuk keperluan **analisis data**, **penerapan machine learning**, serta **penelitian kesehatan mental** dalam mengidentifikasi pola dan faktor pemicu gangguan kecemasan. |
| **⚠️ Masalah**         | Dalam beberapa tahun terakhir, **kecemasan** telah menjadi salah satu masalah kesehatan mental utama, dengan sekitar **264 juta orang di dunia** mengalaminya. Prevalensinya terus meningkat setiap tahunnya. *Serangan kecemasan* dapat terjadi secara tiba-tiba, dengan gejala seperti:  **detak jantung cepat**, **napas pendek**, **berkeringat**, dan **rasa panik**, yang dapat **mengganggu aktivitas sehari-hari**. Oleh karena itu, penting untuk memahami **faktor-faktor** yang mempengaruhi **tingkat keparahan serangan kecemasan**, seperti *demografi, gaya hidup, dan kondisi psikologis*, guna mengembangkan **pendekatan berbasis data** yang efektif dalam **memprediksi dan menangani kecemasan** secara lebih tepat sasaran. |
| **🤖 Solusi Machine Learning** | Solusi yang dikembangkan adalah **model machine learning berbasis klasifikasi** untuk **memprediksi tingkat keparahan serangan kecemasan**. Dengan menggunakan **dataset** yang mencakup berbagai **faktor**, seperti **demografi, gaya hidup, kondisi psikologis,** dan **indikator kesehatan**, model ini dapat **mengidentifikasi pola serta hubungan antar fitur** dengan **tingkat keparahan kecemasan**. Model klasifikasi ini bertujuan untuk memberikan **prediksi yang akurat** mengenai tingkat keparahan serangan, mulai dari **kecemasan ringan** hingga **serangan panik**, yang dapat digunakan untuk **membantu penanganan** yang lebih **tepat dan terarah**. |
| **🛠️ Metode Pengolahan** | Metode pengolahan data dimulai dengan *pembersihan dataset*, yaitu menghapus kolom yang tidak relevan dan memeriksa data duplikat. Kolom `Severity of Anxiety Attack (1-10)` dikategorikan menjadi **empat tingkat keparahan kecemasan** (`Mild`, `Moderate`, `Severe`, `Panic`), yang kemudian diubah menjadi nilai numerik melalui **label encoding**. Data yang telah diproses disimpan dalam format `CSV` untuk digunakan dalam pelatihan model. Modul transformasi `anxiety_transform.py` menggunakan **TensorFlow Transform (TFT)** untuk memproses fitur, dengan melakukan encoding pada data kategorikal seperti `Gender` dan `Occupation`, serta mengubah label target menjadi integer menggunakan `tf.cast`. Hasil transformasi disimpan dalam `dictionary` dan siap digunakan untuk tahap pelatihan, memastikan data memiliki format yang sesuai untuk model machine learning. |
| **🏗️ Arsitektur Model**  | Arsitektur model dalam proyek ini menggunakan **pendekatan deep learning dengan Keras**, yang dioptimalkan melalui **tuning hyperparameter** menggunakan **Keras Tuner** dengan metode `Random Search`. Tuning dilakukan pada hyperparameter seperti jumlah unit di setiap lapisan dense (`unit_1`, `unit_2`, `unit_3`), tingkat dropout (`dropout_1`, `dropout_2`, `dropout_3`), dan laju pembelajaran (`learning_rate`) untuk meningkatkan performa model. Model terdiri dari tiga lapisan dense dengan fungsi aktivasi **ReLU**, diikuti oleh lapisan **dropout** untuk mengurangi **overfitting**, dan lapisan output dengan aktivasi **softmax** untuk klasifikasi empat kategori keparahan kecemasan. Hyperparameter terbaik yang ditemukan meliputi `unit_1 = 384`, `dropout_1 = 0.4`, `unit_2 = 192`, `dropout_2 = 0.2`, `unit_3 = 96`, `dropout_3 = 0.2`, dan `learning_rate = 0.0001`. Model dikompilasi menggunakan **Optimizer Adam** dengan *learning rate* `0.0001`, serta fungsi *loss* `SparseCategoricalCrossentropy` untuk klasifikasi multi-kelas. |
| **📊 Metrik Evaluasi**   | Metrik evaluasi yang digunakan dalam proyek ini meliputi **Sparse Categorical Accuracy** untuk mengukur akurasi model dalam memprediksi kategori yang benar dalam klasifikasi multi-kelas, serta **Sparse Categorical Crossentropy (Loss)** untuk menghitung selisih antara prediksi dan nilai target. Selain itu, digunakan **AUC (Area Under the Curve)** untuk menilai kemampuan model dalam membedakan kelas positif dan negatif, **Precision** untuk mengukur proporsi prediksi positif yang benar, dan **Recall** untuk mengukur proporsi data positif yang benar-benar teridentifikasi oleh model. Terakhir, **Example Count** digunakan untuk menghitung jumlah contoh yang dievaluasi dalam dataset. Metrik-metrik ini memberikan gambaran yang lebih komprehensif tentang performa model baik dalam pelatihan maupun pada data uji. |
| **📈 Performa Model**    | Performa model ini menunjukkan bahwa meskipun terjadi penurunan *loss*, akurasi tetap stagnan di kisaran **30% sepanjang beberapa epoch pertama**. Pada epoch pertama, *loss* tercatat **1.8750** dengan **sparse_categorical_accuracy 30.20%**, dan meskipun ada penurunan pada epoch kedua dengan *loss* menjadi **1.3150** dan akurasi **30.79%**, tidak ada perubahan signifikan dalam akurasi atau *loss*. Pada tahap evaluasi, *validation accuracy* hanya sedikit lebih rendah dari *training accuracy*, yang menunjukkan bahwa model belum cukup optimal dan masih stagnan. Hal ini mengindikasikan perlunya **penyesuaian lebih lanjut pada arsitektur, pemilihan fitur, atau teknik pelatihan** untuk mencapai performa yang lebih baik. |
| **🚀 Opsi Deployment**   | Opsi deployment untuk model ini meliputi **dua pendekatan utama**, yaitu menggunakan **Railway** untuk deployment API publik dan **Docker** untuk deployment lokal. **Railway** memungkinkan model untuk di-*deploy* sebagai **API** yang dapat diakses publik melalui **endpoint HTTP**, memudahkan **prediksi real-time** dengan skalabilitas cloud yang tinggi. Sementara itu, **Docker** menawarkan solusi lokal dengan membangun **container** yang mencakup semua dependensi model, memungkinkan deployment di mesin lokal atau server pribadi. **Docker** ideal untuk pengujian cepat atau deployment skala kecil tanpa tergantung pada platform cloud. Kedua opsi ini memberikan fleksibilitas, dengan **Railway** lebih cocok untuk aplikasi publik dan **Docker** untuk prototyping atau deployment di server pribadi. |
| **🌐 Web App**           | Tautan web app yang digunakan untuk mengakses model serving: [**Anxiety-Model**](https://ml-pipeline-anxiety-attack-production.up.railway.app/v1/models/anxiety-model/metadata) |
| **📡 Monitoring**        | Monitoring model dilakukan dengan **Prometheus** untuk mengumpulkan **metrik secara real-time** dan **Grafana** untuk visualisasinya dalam bentuk **dashboard interaktif**. **Prometheus** mengumpulkan data seperti **akurasi, loss, dan waktu respon**, sementara **Grafana** memungkinkan pemantauan **kinerja model secara langsung** dan **analisis tren**, sehingga mempermudah deteksi masalah dan perbaikan model. |

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