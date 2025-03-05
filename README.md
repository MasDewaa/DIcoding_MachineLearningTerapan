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

Dalam analisis univariate, dilakukan untuk:
- **Histogram untuk Variabel Numerik:**  
  Visualisasi histogram dengan overlay KDE (Kernel Density Estimate) pada variabel numerik seperti Age, Height, Weight, FCVC, NCP, CH2O, FAF, dan TUE mengungkap beberapa hal berikut:

  - Age: Distribusi usia cenderung berpusat pada usia dewasa muda hingga paruh baya, dengan sedikit ekor panjang di usia yang lebih tinggi.
  - Height: Terlihat mayoritas data berpusat di sekitar tinggi rata-rata, meskipun terdapat beberapa outlier yang cukup jauh dari puncak distribusi.
  - Weight: Distribusi berat badan cenderung melebar, menandakan keragaman berat badan yang signifikan. Outlier yang cukup ekstrem juga terdeteksi pada sisi berat badan yang tinggi.
  - FCVC, NCP, CH2O, FAF, dan TUE: Masing-masing variabel memiliki skala 0-3 atau 1-3, sehingga histogram memperlihatkan sebaran terbatas pada rentang nilai tertentu. Sebagian besar individu cenderung berada pada frekuensi aktivitas atau konsumsi yang moderat (nilai tengah).

- **Count Plot untuk Variabel Kategorikal:**  
  Count plot pada variabel kategorikal seperti Gender, family_history_with_overweight, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS, dan NObeyesdad memperlihatkan:

  - Gender: Terlihat pembagian yang relatif seimbang antara individu berjenis kelamin pria dan wanita.
  family_history_with_overweight: Mayoritas responden memiliki riwayat keluarga overweight, menunjukkan bahwa faktor genetik dan lingkungan keluarga berperan penting dalam risiko obesitas.
  - FAVC, CAEC, SMOKE, SCC, CALC: Frekuensi masing-masing kategori memberikan gambaran kebiasaan gaya hidup, seperti kebiasaan merokok, konsumsi makanan ringan, atau memantau asupan kalori.
  - MTRANS: Moda transportasi yang dominan adalah mobil (Automobile) dan transportasi umum (Public Transportation), sementara penggunaan sepeda motor dan sepeda relatif lebih rendah.
  - NObeyesdad: Klasifikasi tingkat obesitas bervariasi, dengan porsi terbanyak pada kategori Normal Weight, Overweight, dan Obesity Type I.

### Multivariate Analysis

Dua visualisasi berikut memberikan pandangan lebih luas terhadap hubungan antar variabel numerik:

- **Pairplot for Numeric Features**  
   - Pairplot di atas memperlihatkan hubungan dua variabel numerik sekaligus dan distribusinya. Setiap diagonal menampilkan histogram (atau KDE) untuk satu variabel, sementara sel di luar diagonal menampilkan scatter plot untuk dua variabel.  
   - Hasil pengamatan menunjukkan bahwa beberapa variabel seperti Height dan Weight memiliki korelasi positif yang jelas, sedangkan variabel lain (misalnya Age, CH2O, TUE) tidak menunjukkan korelasi linear yang kuat.

- **Feature Correlation Heatmap**  
   - Heatmap di atas menampilkan matriks korelasi antar fitur numerik. Skala warna pada sisi kanan menunjukkan tingkat korelasi, dari -1 (korelasi negatif sempurna) hingga +1 (korelasi positif sempurna).  
   - Terlihat bahwa Weight memiliki korelasi positif yang cukup tinggi dengan Height (sekitar 0.46), sedangkan Age, CH2O, FAF, dan TUE cenderung tidak memiliki korelasi linear yang kuat terhadap fitur lain (nilai korelasi di bawah 0.3).  
   - Korelasi yang relatif rendah antara kebanyakan variabel menandakan bahwa data ini memiliki banyak faktor gaya hidup yang berdampak, namun tidak secara linear. Hal ini memengaruhi strategi pemilihan algoritma machine learning dan teknik feature engineering yang tepat untuk meningkatkan performa model.

Secara keseluruhan, pairplot dan correlation heatmap ini mengonfirmasi bahwa variabel Height dan Weight memiliki hubungan linear yang paling menonjol, sementara variabel lain memerlukan pendekatan yang lebih kompleks atau non-linear untuk diolah dalam model. Wawasan ini menjadi dasar bagi proses pemilihan fitur, pemodelan, dan evaluasi selanjutnya.  

---

## Data Preparation

Proses Data Preparation telah dilakukan secara sistematis untuk memastikan data dalam kondisi optimal sebelum digunakan untuk pemodelan. Langkah-langkah yang telah dilakukan adalah sebagai berikut:

1. **Pembersihan Data:**  
   - Dilakukan pemeriksaan terhadap missing value dan tidak ditemukan nilai yang hilang pada dataset.  
   - Baris duplikat telah diidentifikasi dan dihapus, sehingga setiap record dalam dataset unik dan konsisten.

2. **Encoding Variabel Kategorikal:**  
   - Seluruh variabel kategorikal (misalnya, Gender, family_history_with_overweight, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS, dan NObeyesdad) diubah ke dalam format numerik menggunakan LabelEncoder.  
   - Pengkodean ini memastikan bahwa data dapat diproses oleh algoritma machine learning yang hanya menerima input numerik.  
   - Pemeriksaan ulang dilakukan untuk memastikan tidak ada kolom kategorikal yang tertinggal, sehingga semua data dalam fitur telah terkonversi dengan benar.

3. **Pemisahan Fitur dan Target:**  
   - Setelah proses encoding, fitur (predictor) dan target (kelas obesitas) telah dipisahkan dengan tepat.  
   - Target diambil dari kolom NObeyesdad, sedangkan seluruh kolom lainnya digunakan sebagai fitur untuk memprediksi tingkat obesitas.

4. **Train-Test Split:**  
   - Data dibagi menjadi data pelatihan dan data pengujian menggunakan metode train-test split.  
   - Pembagian dilakukan sebelum proses standarisasi untuk menghindari kebocoran informasi antara data pelatihan dan pengujian.

5. **Standarisasi Fitur Numerik:**  
   - Proses standarisasi dilakukan pada data pelatihan dengan StandardScaler, dan transformasi yang sama kemudian diterapkan pada data pengujian.  
   - Langkah ini memastikan bahwa seluruh fitur numerik berada pada skala yang seragam, yang penting untuk algoritma machine learning yang sensitif terhadap perbedaan skala.

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

### Hasil Metrik Evaluasi

1. **Random Forest Classifier**  
   - **Accuracy:** 0.9540  
   - **Precision:** 0.9550  
   - **Recall:** 0.9540  
   - **F1 Score:** 0.9543  
   - **Confusion Matrix:**  
     ```
     [[70  4  0  0  0  0  0]
      [ 2 69  0  0  0  5  0]
      [ 0  0 83  0  0  0  2]
      [ 0  0  0 82  0  0  0]
      [ 0  0  0  0 77  0  0]
      [ 0  6  0  0  0 63  2]
      [ 0  2  1  0  0  0 54]]
     ```
   Model ini menunjukkan performa yang sangat baik, dengan akurasi di atas 95%. Nilai precision, recall, dan F1 score yang hampir seimbang menandakan bahwa Random Forest dapat mengklasifikasikan tiap kelas obesitas dengan cukup konsisten.

2. **Decision Tree Classifier**  
   - **Accuracy:** 0.9425  
   - **Precision:** 0.9422  
   - **Recall:** 0.9425  
   - **F1 Score:** 0.9422  
   - **Confusion Matrix:**  
     ```
     [[69  5  0  0  0  0  0]
      [ 6 65  0  0  0  4  1]
      [ 0  0 80  2  1  0  2]
      [ 0  0  2 80  0  0  0]
      [ 0  0  0  0 77  0  0]
      [ 0  2  0  0  0 68  1]
      [ 0  0  4  0  0  0 53]]
     ```
   Model ini juga menunjukkan performa yang baik dengan akurasi di atas 94%. Namun, Decision Tree lebih rentan terhadap overfitting dan terkadang kurang stabil dibandingkan metode ensemble lainnya.

3. **AdaBoost Classifier**  
   - **Accuracy:** 0.3812  
   - **Precision:** 0.1986  
   - **Recall:** 0.3812  
   - **F1 Score:** 0.2596  
   - **Confusion Matrix:**  
     ```
     [[ 0 74  0  0  0  0  0]
      [ 0 51  6  0  0  0 19]
      [ 0  0 66  0 19  0  0]
      [ 0  0  8  0 74  0  0]
      [ 0  0  0  0 77  0  0]
      [ 0 12 27  0  0  0 32]
      [ 0  4 48  0  0  0  5]]
     ```
   Hasil ini menunjukkan bahwa AdaBoost kurang berhasil pada dataset obesitas ini, dengan akurasi di bawah 40% dan metrik evaluasi lain yang rendah. Ada kemungkinan model ini tidak cocok dengan distribusi data, atau parameter default belum dioptimalkan.

4. **K-Nearest Neighbors (KNN)**  
   - **Accuracy:** 0.8065  
   - **Precision:** 0.8024  
   - **Recall:** 0.8065  
   - **F1 Score:** 0.7972  
   - **Confusion Matrix:** (Truncated)
     ```
     ...
     ```
   KNN menunjukkan akurasi sekitar 80%, cukup baik namun tidak setinggi metode ensemble. KNN juga sensitif terhadap skala data dan parameter seperti jumlah tetangga (k), sehingga memerlukan tuning lebih lanjut.

5. **Gradient Boosting Classifier**  
   - **Accuracy:** Sekitar 96.55% (nilai tertinggi di antara semua model).  
   - **Precision, Recall, F1 Score:** Mencapai nilai yang sangat tinggi dan seimbang.  
   - **Confusion Matrix:**  
     ```
     [[69  5  0  0  0  0  0]
      [ 0 70  0  0  0  4  0]
      [ 0  0 84  0  0  0  1]
      [ 0  0  1 81  0  0  0]
      [ 0  0  0  0 77  0  0]
      [ 0  2  0  0  0 67  2]
      [ 0  0  0  0  0  1 56]]
     ```
   Model ini menggabungkan pendekatan boosting secara iteratif, sehingga menghasilkan akurasi tertinggi dan nilai precision, recall, serta F1 score yang sangat stabil. 

### Komparasi dan Pemilihan Model Terbaik

Berdasarkan hasil di atas, **Gradient Boosting Classifier** menempati posisi teratas dengan akurasi sekitar 96.55%, diikuti oleh **Random Forest Classifier** yang memiliki akurasi 95.40%. Kedua model ini sama-sama metode ensemble, namun Gradient Boosting terbukti lebih unggul di dataset obesitas ini. Metode ensemble lain seperti AdaBoost dan KNN menunjukkan performa lebih rendah, sementara Decision Tree, meski akurasinya cukup tinggi, cenderung kurang stabil dibandingkan model ensemble yang lebih canggih.

## Hubungan dengan Business Understanding

- **Jawaban Problem Statement:**  
   - Model yang dikembangkan mampu mengidentifikasi faktor-faktor risiko obesitas secara efektif, sehingga mendukung inisiatif penurunan biaya perawatan kesehatan dan peningkatan kualitas hidup.
   - Hasil akurasi tinggi menegaskan bahwa individu dengan risiko obesitas dapat diidentifikasi secara dini.

- **Pencapaian Goals:**  
   - **Identifikasi Faktor Risiko:** Model telah mengungkap beberapa fitur penting, seperti pola makan (FAVC), frekuensi konsumsi sayuran (FCVC), dan aktivitas fisik (FAF), yang memengaruhi risiko obesitas.  
   - **Pembangunan Model Prediktif:** Gradient Boosting Classifier terbukti memberikan hasil prediksi paling akurat.  
   - **Dukungan Keputusan Bisnis:** Model dapat diintegrasikan dalam sistem pendukung keputusan bagi lembaga kesehatan atau perusahaan asuransi, membantu menyusun program pencegahan dan intervensi yang lebih tepat sasaran.

- **Dampak dari Solusi yang Diusulkan:**  
   - Tahapan data preparation yang terstruktur, penggunaan metode ensemble, serta evaluasi komprehensif dengan berbagai metrik, semuanya berkontribusi terhadap performa model yang andal.  
   - Model prediktif ini membantu pihak terkait dalam mengoptimalkan strategi pencegahan obesitas dan menurunkan biaya perawatan jangka panjang.

## Kesimpulan

Evaluasi model yang dilakukan menggunakan metrik akurasi, precision, recall, F1 score, dan confusion matrix menunjukkan bahwa:
- **Model Terbaik:** Gradient Boosting Classifier, dengan akurasi sekitar 96.55%.  
- **Model Kedua Terbaik:** Random Forest Classifier, dengan akurasi 95.40%.  
- **Model Lain:** Decision Tree, AdaBoost, dan KNN menunjukkan performa yang lebih rendah.

Secara keseluruhan, solusi yang diimplementasikan telah memberikan dampak positif terhadap pemahaman faktor-faktor gaya hidup yang berkontribusi pada obesitas, dan hasil evaluasi model mendukung pengembangan sistem pendukung keputusan berbasis data untuk intervensi dan diagnosis dini dalam konteks kesehatan. Model terbaik yang dihasilkan dapat dijadikan dasar untuk meningkatkan efektivitas program pencegahan dan pengelolaan obesitas secara profesional.
----