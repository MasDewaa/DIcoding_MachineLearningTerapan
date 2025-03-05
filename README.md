# Laporan Proyek Machine Learning - Rama Syailana Dewa

---

## Domain Proyek

Obesitas merupakan masalah kesehatan global yang semakin meningkat, terutama di negara-negara berkembang. Di Amerika Latin—khususnya di Meksiko, Peru, dan Kolombia—prevalensi obesitas terus naik dan menjadi faktor risiko utama bagi penyakit kronis seperti diabetes, hipertensi, dan penyakit kardiovaskular. Proyek ini bertujuan untuk memahami keterkaitan antara gaya hidup (seperti kebiasaan makan, aktivitas fisik, dan riwayat keluarga) dengan tingkat obesitas. Dengan menggunakan machine learning, diharapkan dapat dikembangkan model prediktif yang mampu mengklasifikasikan tingkat obesitas secara akurat, membantu profesional kesehatan dalam pengambilan keputusan berbasis data, dan mengidentifikasi pola-pola tersembunyi dalam data gaya hidup yang berkontribusi terhadap obesitas. Pendekatan ini didukung oleh berbagai penelitian, seperti pada studi "Obesity: Preventing and Managing the Global Epidemic."

---

## Business Understanding

### Problem Statement
Tingkat obesitas yang terus meningkat merupakan salah satu masalah kesehatan yang berdampak signifikan pada produktivitas dan biaya perawatan kesehatan. Secara bisnis, peningkatan prevalensi obesitas menimbulkan konsekuensi besar, seperti meningkatnya biaya rawat inap, berkurangnya produktivitas kerja, dan peningkatan risiko penyakit kronis. Oleh karena itu, terdapat kebutuhan mendesak untuk mengidentifikasi faktor-faktor gaya hidup yang berkontribusi terhadap obesitas serta mengembangkan sistem prediktif yang dapat membantu mengidentifikasi individu berisiko tinggi. Dengan demikian, solusi yang dikembangkan tidak hanya bersifat teknis, tetapi juga berfokus pada pengurangan biaya kesehatan dan peningkatan kualitas hidup masyarakat.

### Goals
- **Identifikasi Faktor Risiko:** Menentukan variabel-variabel utama (misalnya, kebiasaan makan, aktivitas fisik, dan riwayat keluarga) yang secara signifikan mempengaruhi tingkat obesitas.  
- **Pembangunan Model Prediktif:** Mengembangkan model machine learning yang dapat mengklasifikasikan tingkat obesitas dengan akurasi tinggi, sehingga memungkinkan identifikasi dini pada individu berisiko.  
- **Dukungan Keputusan Bisnis:** Menghasilkan output yang terukur dan dapat diintegrasikan ke dalam sistem pendukung keputusan, guna membantu lembaga kesehatan dan perusahaan asuransi dalam merancang program pencegahan dan intervensi yang efektif.

### Solution Statement (Opsional)
Solusi yang diusulkan melibatkan pengembangan pipeline data end-to-end, mulai dari pembersihan data dan transformasi fitur hingga penerapan beberapa algoritma machine learning untuk mengklasifikasikan tingkat obesitas. Hasil prediksi akan dievaluasi dengan menggunakan metrik evaluasi yang komprehensif (akurasi, precision, recall, F1-score, dan confusion matrix). Model terbaik akan diintegrasikan ke dalam sistem pendukung keputusan untuk membantu instansi terkait dalam mengoptimalkan intervensi kesehatan, mengurangi biaya perawatan, dan meningkatkan kualitas hidup masyarakat.

---
## Data Understanding

Dataset yang digunakan adalah Obesity Data Set yang terdiri dari 2111 baris dan 17 kolom. Data ini diperoleh dari UCI Machine Learning Repository dan dapat diakses melalui [Obesity Data Set](https://archive.ics.uci.edu/ml/datasets/Obesity+Data+Set).

### Ringkasan Data
- **Jumlah Data:**  
  Dataset ini mencakup 2111 baris (record) dan 17 kolom (fitur).

- **Kondisi Data:**  
  - **Missing Value:**  
    Pemeriksaan awal menunjukkan bahwa dataset tidak mengandung missing value, sehingga semua kolom memiliki entri yang lengkap.
  - **Duplikat:**  
    Ditemukan adanya baris duplikat yang kemudian dihapus untuk memastikan keunikan setiap record.
  - **Outlier:**  
    Analisis visual menggunakan boxplot mengidentifikasi adanya outlier pada beberapa variabel numerik, seperti pada fitur Height dan Weight, yang perlu dipertimbangkan dalam proses preprocessing.

### Uraian Fitur pada Data
- **Gender:**  
  Variabel kategorikal yang menunjukkan jenis kelamin individu (Male/Female).
- **Age:**  
  Variabel numerik yang menunjukkan usia individu dalam tahun.
- **Height:**  
  Variabel numerik yang mengukur tinggi badan dalam satuan meter.
- **Weight:**  
  Variabel numerik yang mengukur berat badan dalam kilogram.
- **family_history_with_overweight:**  
  Variabel kategorikal (yes/no) yang menunjukkan apakah terdapat riwayat keluarga dengan kecenderungan overweight.
- **FAVC:**  
  Variabel kategorikal (yes/no) yang menunjukkan frekuensi konsumsi makanan tinggi kalori.
- **FCVC:**  
  Variabel numerik dengan skala 1 hingga 3 yang menunjukkan frekuensi konsumsi sayuran.
- **NCP:**  
  Variabel numerik yang menunjukkan jumlah makan utama per hari.
- **CAEC:**  
  Variabel kategorikal yang menggambarkan frekuensi konsumsi makanan ringan di antara waktu makan (Never, Sometimes, Frequently, Always).
- **SMOKE:**  
  Variabel kategorikal (yes/no) yang menunjukkan status perokok individu.
- **CH2O:**  
  Variabel numerik dengan skala 1 hingga 3 yang mengukur asupan air harian.
- **SCC:**  
  Variabel kategorikal (yes/no) yang menunjukkan apakah individu memonitor asupan kalori.
- **FAF:**  
  Variabel numerik dengan skala 0 hingga 3 yang mengukur frekuensi aktivitas fisik.
- **TUE:**  
  Variabel numerik dengan skala 0 hingga 3 yang menunjukkan waktu yang dihabiskan menggunakan teknologi.
- **CALC:**  
  Variabel kategorikal yang menggambarkan frekuensi konsumsi alkohol (Never, Sometimes, Frequently, Always).
- **MTRANS:**  
  Variabel kategorikal yang menyatakan moda transportasi utama (Automobile, Bike, Motorbike, Public Transportation, Walking).
- **NObeyesdad:**  
  Variabel kategorikal yang mengklasifikasikan tingkat obesitas ke dalam kategori: Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, dan Obesity Type III.

Penjelasan identifikasi ini memberikan dasar untuk EDA

## Exploratory Data Analysis (EDA)

Tahap EDA dilakukan untuk menggali informasi lebih dalam dari data melalui analisis univariate dan multivariate.

### Univariate Analysis

Dalam analisis univariate, dilakukan:
- **Count Plot:** Untuk variabel kategorikal, count plot digunakan guna memvisualisasikan frekuensi setiap kategori. Hal ini membantu melihat distribusi dan dominasi kategori tertentu.
- **Histogram:** Untuk variabel numerik, histogram dengan overlay density (KDE) digunakan untuk melihat distribusi nilai serta mendeteksi kemungkinan adanya outlier.
- **Pie Chart:** Data numerik juga dikelompokkan ke dalam beberapa bin, dan pie chart digunakan untuk menampilkan proporsi tiap bin, memberikan gambaran tentang sebaran nilai.

### Multivariate Analysis

Analisis multivariate dilakukan dengan:
- **Pairplot:** Untuk mengeksplorasi hubungan dan distribusi antar variabel numerik secara berpasangan, sehingga dapat mengidentifikasi pola dan hubungan antar fitur.
- **Correlation Heatmap:** Matriks korelasi dibuat untuk menilai hubungan linier antar fitur numerik. Visualisasi ini membantu mengidentifikasi fitur yang sangat berkorelasi, yang penting dalam pemilihan fitur dan pengembangan model.
---

## Data Preparation

Proses persiapan data dilakukan secara runtut untuk memastikan bahwa data bersih, konsisten, dan siap digunakan dalam tahap pemodelan. Berikut adalah langkah-langkah yang dilakukan:

1. **Pembersihan Data:**
   - Dilakukan pemeriksaan seluruh kolom untuk mendeteksi adanya missing values. Hasil pemeriksaan menunjukkan bahwa dataset tidak mengandung nilai yang hilang sehingga tidak diperlukan imputasi.
   - Deteksi baris duplikat dilakukan untuk memastikan setiap record unik. Baris duplikat yang teridentifikasi kemudian dihapus untuk menjaga integritas data.

2. **Transformasi Data:**
   - **Encoding Variabel Kategorikal:**  
     Variabel seperti *Gender*, *family_history_with_overweight*, *FAVC*, *CAEC*, *SMOKE*, *SCC*, *CALC*, *MTRANS*, dan *NObeyesdad* diubah ke format numerik menggunakan teknik Label Encoding. Hal ini memastikan bahwa seluruh variabel dapat diproses oleh algoritma machine learning yang hanya menerima data numerik.
   - **Normalisasi/Standarisasi Variabel Numerik:**  
     Variabel numerik, misalnya *Age*, *Height*, dan *Weight*, distandarisasi menggunakan StandardScaler untuk memastikan setiap fitur memiliki skala yang konsisten. Langkah ini membantu mencegah model terpengaruh oleh perbedaan skala antar variabel.

3. **Splitting Data:**
   - Dataset dibagi menjadi data pelatihan (training set) dan data pengujian (testing set) dengan proporsi yang telah ditentukan (misalnya, 80:20 atau 75:25). Pembagian ini penting untuk mengevaluasi kinerja model secara objektif pada data yang belum pernah digunakan saat pelatihan.
---

## Model Development

Pada tahap pengembangan model, berbagai algoritma klasifikasi diterapkan untuk memprediksi tingkat obesitas berdasarkan data gaya hidup dan kesehatan. Berikut adalah penjelasan mengenai cara kerja masing-masing algoritma beserta parameter yang digunakan:

### Random Forest Classifier
Random Forest merupakan algoritma ensemble yang membangun banyak pohon keputusan (decision trees) dan menggabungkan hasil prediksi dari masing-masing pohon untuk menghasilkan keputusan akhir. Algoritma ini efektif untuk mengatasi overfitting dan menangani dataset dengan banyak fitur.  
- **Parameter Utama:**  
  - *n_estimators:* Jumlah pohon yang dibangun (default = 100).  
  - *max_depth:* Kedalaman maksimum pohon (default = None, yaitu tidak ada batasan).  
  - *max_features:* Jumlah fitur yang dipertimbangkan pada setiap split (default = "auto").

### Decision Tree Classifier
Decision Tree menggunakan struktur pohon untuk mengambil keputusan berdasarkan pembagian fitur secara berurutan. Model ini mudah diinterpretasikan, namun rawan overfitting jika tidak diatur dengan baik.  
- **Parameter Utama:**  
  - *criterion:* Metode untuk mengukur kualitas split (default = "gini", alternatif "entropy").  
  - *max_depth:* Kedalaman maksimum pohon (default = None).  
  - Parameter lain digunakan sesuai nilai default.

### AdaBoost Classifier
AdaBoost (Adaptive Boosting) merupakan metode ensemble yang menggabungkan beberapa model lemah untuk membentuk model yang kuat. Setiap model lemah diperbaiki berdasarkan kesalahan model sebelumnya.  
- **Parameter Utama:**  
  - *n_estimators:* Jumlah model lemah yang digabungkan (default = 50).  
  - *learning_rate:* Mengontrol kontribusi masing-masing model (default = 1.0).  
  - Parameter lain mengikuti nilai default.

### K-Nearest Neighbors (KNN)
KNN adalah algoritma non-parametrik yang mengklasifikasikan instance baru berdasarkan kedekatan (jarak) dengan data training. Model ini bergantung pada pemilihan jumlah tetangga terdekat (neighbors).  
- **Parameter Utama:**  
  - *n_neighbors:* Jumlah tetangga yang dipertimbangkan untuk prediksi (default = 5).  
  - Parameter lainnya menggunakan nilai default.

### Gradient Boosting Classifier
Gradient Boosting merupakan metode ensemble yang membangun model secara iteratif untuk mengurangi error residual dari model sebelumnya. Metode ini cenderung memberikan performa tinggi dan stabil.  
- **Parameter Utama:**  
  - *n_estimators:* Jumlah iterasi (model) yang dibangun (default = 100).  
  - *learning_rate:* Laju pembelajaran untuk setiap iterasi (default = 0.1).  
  - *max_depth:* Kedalaman maksimum pohon (default = 3).  
  - Parameter lain mengikuti nilai default.

### Logistic Regression
Logistic Regression adalah model klasifikasi linier yang menggunakan fungsi sigmoid untuk mengestimasi probabilitas suatu kelas. Model ini sederhana dan efektif terutama jika hubungan antara fitur dan target bersifat linier.  
- **Parameter Utama:**  
  - *max_iter:* Jumlah iterasi maksimum yang diijinkan untuk konvergensi (diatur ke 1000 untuk memastikan konvergensi).  
  - Parameter lainnya menggunakan nilai default.

Setiap model dilatih menggunakan data pelatihan yang telah dipreproses, dan performanya dievaluasi dengan menggunakan metrik seperti akurasi, precision, recall, F1 score, dan confusion matrix. Pemilihan model terbaik didasarkan pada hasil evaluasi yang komprehensif untuk memastikan model yang paling sesuai dengan kebutuhan prediksi tingkat obesitas.

# Evaluation

Bagian evaluasi menyajikan hasil metrik evaluasi dari setiap skema pelatihan dan melakukan komparasi untuk menentukan model terbaik. Hasil evaluasi ini kemudian dihubungkan dengan aspek bisnis dan tujuan proyek untuk memastikan bahwa solusi yang dikembangkan menjawab setiap problem statement dan mencapai goals yang diharapkan.

## Hasil Evaluasi Model

Beberapa model telah dilatih menggunakan data pelatihan yang telah dipreproses, dan evaluasi dilakukan pada data pengujian. Hasil evaluasi diperoleh berdasarkan metrik utama, yaitu:

- **Accuracy:** Persentase prediksi yang benar secara keseluruhan.
- **Precision:** Ketepatan prediksi positif.
- **Recall:** Kemampuan model mendeteksi semua instance positif.
- **F1 Score:** Rata-rata harmonis antara precision dan recall.
- **Confusion Matrix:** Distribusi prediksi terhadap label asli untuk masing-masing kelas.

Dari evaluasi tersebut, ditemukan bahwa:
- **Gradient Boosting Classifier** menghasilkan akurasi tertinggi (sekitar 96.55%) dengan nilai precision, recall, dan F1 score yang sangat tinggi. 
- **Random Forest Classifier** dan **Decision Tree Classifier** juga menunjukkan performa yang baik, meskipun sedikit di bawah model Gradient Boosting.
- **Ada Boost Classifier** dan **KNN** menunjukkan performa yang jauh lebih rendah dan tidak konsisten.

## Komparasi Skema Pelatihan dan Pemilihan Model Terbaik

Dua skema pelatihan yang diterapkan menggunakan berbagai algoritma menunjukkan perbedaan yang signifikan. Model dengan evaluasi terbaik adalah Gradient Boosting Classifier, yang secara konsisten unggul pada semua metrik evaluasi. Pemilihan model terbaik didasarkan pada nilai akurasi tertinggi serta keseimbangan antara precision, recall, dan F1 score. Hal ini menandakan bahwa model tersebut lebih andal dalam mengklasifikasikan tingkat obesitas pada berbagai kelas dibandingkan model-model lainnya.

## Hubungan dengan Business Understanding

Evaluasi model harus dihubungkan dengan aspek bisnis dan kebutuhan yang telah diidentifikasi pada Business Understanding. Beberapa poin penting adalah sebagai berikut:

- **Jawaban Problem Statement:**  
  Evaluasi model menunjukkan bahwa pendekatan yang digunakan mampu mengidentifikasi pola-pola penting dalam data gaya hidup, sehingga membantu menjawab permasalahan utama yaitu mengidentifikasi faktor-faktor risiko obesitas. Model prediktif yang dihasilkan (terutama Gradient Boosting Classifier) dapat digunakan untuk mendeteksi individu berisiko tinggi, sehingga berpotensi mengurangi biaya perawatan dan meningkatkan efektivitas intervensi kesehatan.

- **Pencapaian Goals:**  
  Setiap goals yang telah ditetapkan tercapai secara terukur. Akurasi yang tinggi dan nilai evaluasi lain (precision, recall, dan F1 score) mengindikasikan bahwa model telah mencapai tujuan untuk mengklasifikasikan tingkat obesitas secara akurat. Hasil evaluasi mendukung pemanfaatan model sebagai alat bantu bagi profesional kesehatan untuk diagnosis dini dan pengambilan keputusan berbasis data.

- **Dampak dari Solution Statement:**  
  Proses data preparation yang mencakup pembersihan data, transformasi melalui encoding dan normalisasi, serta pembagian data pelatihan dan pengujian, semuanya berkontribusi pada kinerja model yang optimal. Penggunaan teknik ensemble dan hyperparameter tuning menunjukkan dampak positif terhadap performa model, sehingga solusi yang direncanakan terbukti berdampak signifikan terhadap hasil akhir. Selain itu, evaluasi komprehensif dengan berbagai metrik memastikan bahwa model tidak hanya berperforma tinggi secara teknis, tetapi juga relevan secara bisnis dengan membantu mengidentifikasi dan mengurangi risiko obesitas.

## Kesimpulan

Evaluasi model yang dilakukan menggunakan metrik akurasi, precision, recall, F1 score, dan confusion matrix menunjukkan bahwa:
- Model terbaik adalah Gradient Boosting Classifier, yang menunjukkan performa tertinggi.
- Model lain seperti Random Forest dan Decision Tree juga menunjukkan hasil yang baik, namun tidak seoptimal model Gradient Boosting.
- Evaluasi menyeluruh mendukung pemenuhan setiap problem statement yang telah diidentifikasi dan pencapaian goals yang diharapkan.

Secara keseluruhan, solusi yang diimplementasikan telah memberikan dampak positif terhadap pemahaman faktor-faktor gaya hidup yang berkontribusi pada obesitas, dan hasil evaluasi model mendukung pengembangan sistem pendukung keputusan berbasis data untuk intervensi dan diagnosis dini dalam konteks kesehatan. Model terbaik yang dihasilkan dapat dijadikan dasar untuk meningkatkan efektivitas program pencegahan dan pengelolaan obesitas secara profesional.
----