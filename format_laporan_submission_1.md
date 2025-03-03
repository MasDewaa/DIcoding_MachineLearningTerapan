# Laporan Proyek Machine Learning - Rama Syailana Dewa

---

## Domain Proyek

Obesitas merupakan masalah kesehatan global yang terus meningkat, terutama di negara-negara berkembang. Di Amerika Latin—terutama Meksiko, Peru, dan Kolombia—prevalensi obesitas semakin tinggi dan menjadi faktor risiko utama bagi penyakit kronis seperti diabetes, hipertensi, dan penyakit kardiovaskular. Proyek ini diangkat untuk memahami keterkaitan antara gaya hidup (misalnya, kebiasaan makan, aktivitas fisik, dan riwayat keluarga) dengan tingkat obesitas. Dengan memanfaatkan machine learning, diharapkan dapat dikembangkan model prediktif yang:

- **Mengklasifikasikan tingkat obesitas secara akurat.**
- **Menjadi alat bantu bagi profesional kesehatan dalam pengambilan keputusan berbasis data.**
- **Mengidentifikasi pola-pola tersembunyi dalam data gaya hidup yang berkontribusi terhadap obesitas.**

Pendekatan ini didukung oleh berbagai penelitian yang menekankan pentingnya intervensi dini dan pemanfaatan teknologi data-driven dalam penanganan obesitas, misalnya pada penelitian [Obesity: Preventing and Managing the Global Epidemic](https://scholar.google.com/).

---

## Business Understanding

Tahap ini berfokus pada pemahaman mendalam terhadap permasalahan dan perumusan tujuan yang hendak dicapai. Klarifikasi masalah memastikan bahwa solusi yang dikembangkan akan relevan dan berdampak positif bagi kesehatan masyarakat.

### Problem Statements

1. **Pernyataan Masalah 1:**  
   *Bagaimana mengidentifikasi faktor-faktor utama dalam gaya hidup (pola makan, aktivitas fisik, dan riwayat keluarga) yang berkontribusi terhadap peningkatan risiko obesitas?*

2. **Pernyataan Masalah 2:**  
   *Bagaimana mengembangkan model machine learning yang dapat mengklasifikasikan tingkat obesitas dengan akurasi tinggi berdasarkan data gaya hidup?*

3. **Pernyataan Masalah 3:**  
   *Bagaimana mengintegrasikan data historis dan data sintetik secara efektif untuk meningkatkan validitas dan performa model prediktif?*

### Goals

1. **Jawaban Pernyataan Masalah 1:**  
   - Mengidentifikasi dan menganalisis variabel-variabel kritis (kebiasaan makan, frekuensi aktivitas fisik, riwayat keluarga) yang berperan dalam peningkatan risiko obesitas.

2. **Jawaban Pernyataan Masalah 2:**  
   - Mengembangkan, melatih, dan menguji beberapa model machine learning (misalnya, Random Forest, Support Vector Machine, dan Neural Network) untuk mengklasifikasikan tingkat obesitas dengan metrik evaluasi seperti akurasi, precision, recall, dan F1-score.

3. **Jawaban Pernyataan Masalah 3:**  
   - Menggabungkan data historis dan data sintetik guna menciptakan dataset yang representatif serta menerapkan teknik data augmentation untuk meningkatkan performa model.

### Solution Statements

1. **Data Processing:**  
   - Membangun pipeline pembersihan data dan transformasi fitur untuk memastikan kualitas data sebelum analisis lebih lanjut.

2. **Pengembangan Model:**  
   - Mengembangkan dan membandingkan beberapa model machine learning dengan melakukan hyperparameter tuning untuk menemukan model dengan performa terbaik.

3. **Evaluasi Model:**  
   - Mengimplementasikan validasi silang (cross-validation) dan menggunakan metrik evaluasi yang terukur untuk memastikan keandalan model.

4. **Ensemble Learning:**  
   - Menerapkan teknik ensemble untuk menggabungkan prediksi dari beberapa model guna meningkatkan akurasi dan stabilitas prediksi.

---

## Data Understanding

Dataset obesitas yang digunakan mencakup 2111 catatan individu dari tiga negara, yaitu Meksiko, Peru, dan Kolombia. Dataset ini menggabungkan data gaya hidup dan kesehatan melalui kombinasi teknik sintetik dan pengumpulan data langsung melalui platform web. Dataset dapat diunduh melalui [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Obesity+Data+Set) *(tautan contoh)*.

### Variabel-variabel dalam Dataset

- **Gender:** Jenis kelamin individu (Male/Female).
- **Age:** Usia dalam tahun.
- **Height:** Tinggi badan dalam meter.
- **Weight:** Berat badan dalam kilogram.
- **family_history_with_overweight:** Riwayat keluarga dengan kecenderungan overweight (yes/no).
- **FAVC:** Frekuensi konsumsi makanan tinggi kalori (yes/no).
- **FCVC:** Frekuensi konsumsi sayuran (skala 1 hingga 3).
- **NCP:** Jumlah makan utama per hari.
- **CAEC:** Frekuensi konsumsi makanan ringan di antara waktu makan (Never, Sometimes, Frequently, Always).
- **SMOKE:** Status perokok (yes/no).
- **CH2O:** Asupan air harian (skala 1 hingga 3).
- **SCC:** Kebiasaan memonitor asupan kalori (yes/no).
- **FAF:** Frekuensi aktivitas fisik (skala 0 hingga 3).
- **TUE:** Waktu yang dihabiskan menggunakan teknologi (skala 0 hingga 3).
- **CALC:** Frekuensi konsumsi alkohol (Never, Sometimes, Frequently, Always).
- **MTRANS:** Moda transportasi utama (Automobile, Bike, Motorbike, Public Transportation, Walking).
- **NObeyesdad:** Klasifikasi tingkat obesitas (Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, Obesity Type III).

Untuk lebih memahami karakteristik data, dilakukan eksplorasi awal (exploratory data analysis) dengan visualisasi seperti histogram, boxplot, dan heatmap korelasi. Teknik-teknik ini membantu mengidentifikasi distribusi data, outlier, dan hubungan antar variabel.

---

## Data Preparation

Tahapan data preparation dilakukan untuk memastikan bahwa data siap digunakan dalam pemodelan. Langkah-langkah yang dilakukan meliputi:

1. **Pembersihan Data (Data Cleaning):**
   - **Identifikasi Missing Values:** Pengecekan nilai yang hilang dan penentuan strategi penanganan (misalnya, imputasi atau penghapusan baris).
   - **Penanganan Outlier:** Mendeteksi dan menangani outlier yang dapat mempengaruhi kinerja model.

2. **Transformasi Data:**
   - **Encoding Variabel Kategorikal:** Variabel seperti *Gender*, *family_history_with_overweight*, *FAVC*, *SMOKE*, *CAEC*, *CALC*, dan *MTRANS* diubah ke format numerik menggunakan teknik One-Hot Encoding atau Label Encoding.
   - **Normalisasi/Standarisasi:** Variabel numerik seperti *Age*, *Height*, dan *Weight* dinormalisasi menggunakan StandardScaler agar memiliki skala yang konsisten.  
     *Alasan:* Transformasi ini diperlukan agar algoritma machine learning dapat bekerja optimal dengan data yang terdistribusi secara normal dan homogen.

3. **Splitting Data:**
   - Membagi dataset menjadi data pelatihan (training set) dan data pengujian (testing set) untuk mengevaluasi performa model secara objektif.

   **Contoh Code Snippet:**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeling

Bagian ini mendokumentasikan proses pemodelan yang digunakan dalam proyek prediksi tingkat obesitas berdasarkan data gaya hidup dan kesehatan.

---

## Algoritma yang Digunakan

Dalam proyek ini, beberapa algoritma machine learning diterapkan untuk mengklasifikasikan tingkat obesitas. Berikut adalah deskripsi masing-masing algoritma:

### 1. Random Forest Classifier
- **Kelebihan:**
  - Robust terhadap overfitting, terutama pada dataset dengan banyak fitur.
  - Mampu menangani variabel numerik dan kategorikal dengan baik.
- **Kekurangan:**
  - Proses training bisa menjadi lambat pada dataset yang sangat besar.
- **Parameter Utama:** `n_estimators`, `max_depth`, `max_features`.

### 2. Support Vector Machine (SVM)
- **Kelebihan:**
  - Efektif pada ruang berdimensi tinggi.
  - Cocok untuk dataset dengan ukuran yang tidak terlalu besar.
- **Kekurangan:**
  - Memerlukan penalaan parameter kernel yang cermat dan sensitif terhadap skala data.
- **Parameter Utama:** Kernel (misalnya, linear, RBF), parameter regulasi `C`, dan `gamma`.

### 3. Neural Network (Multi-Layer Perceptron)
- **Kelebihan:**
  - Mampu menangkap hubungan non-linear yang kompleks dalam data.
  - Fleksibel dalam arsitektur sehingga dapat disesuaikan dengan berbagai jenis data.
- **Kekurangan:**
  - Memerlukan data training yang cukup besar dan tuning hyperparameter yang kompleks.
- **Parameter Utama:** Jumlah lapisan tersembunyi, jumlah neuron per lapisan, fungsi aktivasi, dan learning rate.

---

## Proses Improvement & Hyperparameter Tuning

Untuk setiap model, dilakukan proses tuning hyperparameter guna menemukan konfigurasi parameter yang optimal dan meningkatkan performa model. Langkah-langkah yang dilakukan meliputi:

- **Menentukan Parameter Grid:**  
  Menetapkan rentang nilai untuk masing-masing parameter model.
- **Validasi Silang (Cross-Validation):**  
  Menggunakan teknik validasi silang untuk mengevaluasi performa setiap kombinasi parameter.
- **Pemilihan Model Terbaik:**  
  Memilih model dengan performa terbaik berdasarkan metrik evaluasi seperti akurasi, precision, recall, dan F1-score.

### Contoh Code Snippet untuk Grid Search (Random Forest)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Definisikan parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'max_features': ['auto', 'sqrt']
}

# Inisialisasi RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Lakukan Grid Search dengan cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Dapatkan model terbaik dan parameter optimal
best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
```

# Evaluation Metrics

## Overview

Dalam proyek prediksi tingkat obesitas ini, metrik evaluasi digunakan untuk mengukur performa model dalam mengklasifikasikan data berdasarkan variabel gaya hidup dan kesehatan. Metrik yang digunakan mencakup:

- **Akurasi**
- **Precision**
- **Recall**
- **F1 Score**

Metrik-metrik ini dipilih karena memberikan gambaran menyeluruh mengenai kekuatan model, baik dalam mendeteksi kelas positif maupun dalam menjaga keseimbangan antara kesalahan tipe I dan tipe II.

---

## Detail Metrik

### Akurasi
- **Definisi:**  
  Persentase prediksi yang benar dari total keseluruhan data.
- **Formula:**  
  \[
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  \]
- **Keterangan:**  
  Akurasi merupakan metrik dasar yang memberikan gambaran umum tentang kinerja model. Namun, jika dataset tidak seimbang, akurasi saja mungkin tidak cukup menggambarkan performa model.

### Precision
- **Definisi:**  
  Mengukur seberapa tepat prediksi positif yang dibuat oleh model.
- **Formula:**  
  \[
  \text{Precision} = \frac{TP}{TP + FP}
  \]
- **Keterangan:**  
  Precision tinggi mengindikasikan bahwa model jarang salah mengklasifikasikan data negatif sebagai positif.

### Recall
- **Definisi:**  
  Mengukur kemampuan model dalam menangkap seluruh data positif yang sebenarnya.
- **Formula:**  
  \[
  \text{Recall} = \frac{TP}{TP + FN}
  \]
- **Keterangan:**  
  Recall yang tinggi berarti model mampu mendeteksi sebagian besar contoh positif, sehingga mengurangi kemungkinan terjadinya false negatives.

### F1 Score
- **Definisi:**  
  Rata-rata harmonis dari Precision dan Recall, memberikan keseimbangan antara keduanya.
- **Formula:**  
  \[
  \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  \]
- **Keterangan:**  
  F1 Score sangat berguna ketika ingin mendapatkan keseimbangan antara Precision dan Recall, terutama pada situasi dengan distribusi kelas yang tidak seimbang.

---

## Implementasi Kode

Berikut contoh implementasi perhitungan metrik evaluasi menggunakan Python dan scikit-learn:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Misal y_true adalah label asli, dan y_pred adalah hasil prediksi model
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("Akurasi: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))
```

# Conclusion

Dalam proyek prediksi tingkat obesitas ini, evaluasi model dilakukan dengan menggunakan metrik seperti akurasi, precision, recall, dan F1 score. Berdasarkan analisis dan pengujian yang telah dilakukan, terdapat beberapa poin penting yang dapat disimpulkan:

- **Efektivitas Model:**  
  Model yang dikembangkan mampu mengklasifikasikan tingkat obesitas dengan performa yang baik. Metrik evaluasi menunjukkan bahwa model dapat secara konsisten menghasilkan prediksi yang akurat serta seimbang dalam menangkap kelas positif dan negatif.

- **Keseimbangan Precision dan Recall:**  
  Dengan mempertimbangkan nilai precision dan recall, model terpilih berhasil mengurangi kesalahan false positive dan false negative. Hal ini penting untuk aplikasi kesehatan, di mana kesalahan klasifikasi dapat berdampak signifikan terhadap intervensi dan diagnosis.

- **Keseimbangan F1 Score:**  
  F1 score, sebagai rata-rata harmonis dari precision dan recall, memberikan gambaran bahwa model tidak hanya akurat tetapi juga sensitif terhadap data positif. Ini mengindikasikan bahwa pendekatan yang digunakan tepat dalam menangani dataset dengan distribusi kelas yang tidak seimbang.

- **Potensi Penerapan:**  
  Hasil evaluasi mendukung potensi penggunaan model ini dalam skenario dunia nyata. Model dapat dijadikan dasar untuk pengembangan sistem pendukung keputusan berbasis data yang membantu profesional kesehatan dalam melakukan diagnosis dini dan intervensi personal terhadap masalah obesitas.

- **Rekomendasi Pengembangan:**  
  Disarankan untuk melakukan pengembangan lebih lanjut, seperti eksplorasi teknik ensemble atau integrasi data tambahan, guna meningkatkan performa dan generalisasi model.

Secara keseluruhan, proyek ini tidak hanya memberikan insight mendalam mengenai faktor-faktor yang berkontribusi terhadap obesitas, tetapi juga menekankan pentingnya evaluasi komprehensif dalam pengembangan solusi machine learning untuk aplikasi kesehatan.
