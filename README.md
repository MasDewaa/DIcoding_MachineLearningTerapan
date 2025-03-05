# Laporan Proyek Machine Learning - Rama Syailana Dewa

---

## Domain Proyek

Obesitas merupakan masalah kesehatan global yang semakin meningkat, terutama di negara-negara berkembang. Di Amerika Latin—khususnya Meksiko, Peru, dan Kolombia—prevalensi obesitas terus naik dan menjadi faktor risiko utama bagi penyakit kronis seperti diabetes, hipertensi, dan penyakit kardiovaskular. Proyek ini diangkat untuk memahami keterkaitan antara gaya hidup (misalnya, kebiasaan makan, aktivitas fisik, dan riwayat keluarga) dengan tingkat obesitas. Dengan memanfaatkan machine learning, diharapkan dapat dikembangkan model prediktif yang:

- **Mengklasifikasikan tingkat obesitas secara akurat.**
- **Menjadi alat bantu bagi profesional kesehatan dalam pengambilan keputusan berbasis data.**
- **Mengidentifikasi pola-pola tersembunyi dalam data gaya hidup yang berkontribusi terhadap obesitas.**

Pendekatan ini didukung oleh berbagai penelitian, seperti pada studi [Obesity: Preventing and Managing the Global Epidemic](https://onlinelibrary.wiley.com/doi/abs/10.1111/obr.12872).

---

## Business Understanding

Pada tahap ini, fokus utama adalah memahami permasalahan serta merumuskan tujuan dan solusi yang relevan.

### Problem Statements

1. **Identifikasi Faktor Gaya Hidup:**  
   Bagaimana mengidentifikasi faktor-faktor utama (pola makan, aktivitas fisik, dan riwayat keluarga) yang berkontribusi terhadap peningkatan risiko obesitas?

2. **Pengembangan Model Prediktif:**  
   Bagaimana mengembangkan model machine learning yang dapat mengklasifikasikan tingkat obesitas dengan akurasi tinggi berdasarkan data gaya hidup?

3. **Integrasi Data:**  
   Bagaimana mengintegrasikan data historis dan data sintetik secara efektif untuk meningkatkan validitas dan performa model prediktif?

### Goals

- **Analisis Variabel:** Mengidentifikasi dan menganalisis variabel kritis yang mempengaruhi tingkat obesitas.
- **Modeling:** Membangun dan menguji beberapa model (misalnya, Random Forest, Decision Tree, AdaBoost, KNN, Gradient Boosting, Logistic Regression) dengan evaluasi menggunakan metrik seperti akurasi, precision, recall, F1-score, dan confusion matrix.
- **Data Representatif:** Menggabungkan data historis dan sintetik serta menerapkan teknik data augmentation untuk meningkatkan performa model.

### Solution Statements

1. **Data Processing:**  
   Membangun pipeline pembersihan data dan transformasi fitur (encoding, normalisasi) untuk memastikan data berkualitas.

2. **Pengembangan Model:**  
   Mengembangkan dan membandingkan beberapa model dengan hyperparameter tuning guna mendapatkan performa terbaik.

3. **Evaluasi Model:**  
   Mengimplementasikan validasi silang dan menggunakan berbagai metrik evaluasi untuk memastikan keandalan model.

4. **Ensemble Learning:**  
   Menerapkan teknik ensemble untuk menggabungkan prediksi beberapa model guna meningkatkan akurasi dan stabilitas prediksi.

---

## Data Understanding

Dataset obesitas yang digunakan mencakup 2111 catatan individu dari tiga negara (Meksiko, Peru, dan Kolombia). Data ini menggabungkan informasi gaya hidup dan kesehatan yang diperoleh melalui metode sintetik dan pengumpulan langsung via platform web. Dataset dapat diunduh melalui [Kaggle](https://www.kaggle.com/datasets/adeniranstephen/obesity-prediction-dataset).

### Variabel dalam Dataset

- **Gender:** Jenis kelamin (Male/Female)
- **Age:** Usia (tahun)
- **Height:** Tinggi badan (meter)
- **Weight:** Berat badan (kg)
- **family_history_with_overweight:** Riwayat keluarga dengan kecenderungan overweight (yes/no)
- **FAVC:** Frekuensi konsumsi makanan tinggi kalori (yes/no)
- **FCVC:** Frekuensi konsumsi sayuran (skala 1-3)
- **NCP:** Jumlah makan utama per hari
- **CAEC:** Frekuensi konsumsi makanan ringan (Never, Sometimes, Frequently, Always)
- **SMOKE:** Status perokok (yes/no)
- **CH2O:** Asupan air harian (skala 1-3)
- **SCC:** Kebiasaan memonitor asupan kalori (yes/no)
- **FAF:** Frekuensi aktivitas fisik (skala 0-3)
- **TUE:** Waktu penggunaan teknologi (skala 0-3)
- **CALC:** Frekuensi konsumsi alkohol (Never, Sometimes, Frequently, Always)
- **MTRANS:** Moda transportasi (Automobile, Bike, Motorbike, Public Transportation, Walking)
- **NObeyesdad:** Klasifikasi tingkat obesitas (Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, Obesity Type III)

Eksplorasi awal dilakukan menggunakan fungsi seperti `head()`, `info()`, dan `describe()`. Visualisasi awal (histogram, boxplot, heatmap) membantu mengidentifikasi distribusi, outlier, dan hubungan antar variabel.

---

## Data Preparation

Langkah-langkah persiapan data meliputi:

1. **Pembersihan Data:**  
   - Mengecek missing values dan menghapus duplikasi.

2. **Transformasi Data:**  
   - **Encoding:** Variabel kategorikal (seperti Gender, FAVC, dll.) diubah ke format numerik menggunakan LabelEncoder.
   - **Normalisasi:** Variabel numerik (Age, Height, Weight) dinormalisasi dengan StandardScaler untuk memastikan skala yang konsisten.

3. **Splitting Data:**  
   - Dataset dibagi menjadi data pelatihan dan pengujian menggunakan `train_test_split`.

Contoh:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## Exploratory Data Analysis (EDA)

### Univariate Analysis

#### Count Plot for Categorical Variables
Visualisasi frekuensi masing-masing kategori pada variabel kategorikal.
```python
plt.figure(figsize=(15, 10))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=col, data=df, palette="viridis")
    plt.title(f"Count of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
plt.tight_layout()
plt.show()
```
#### Histogram for Numeric Variables
Menganalisis distribusi setiap variabel numerik.
```python
plt.figure(figsize=(15, num_rows * 5))
for i, column in enumerate(numeric_cols, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.histplot(df[column], bins=30, kde=True, color='darkblue')
    plt.title(f"Histogram: {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
```
#### Pie Charts for Binned Numeric Data
Menganalisis distribusi setiap variabel numerik.
```python
plt.figure(figsize=(15, num_rows * 5))
for i, column in enumerate(numeric_cols, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.histplot(df[column], bins=30, kde=True, color='darkblue')
    plt.title(f"Histogram: {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
```
### Multivariate Analysis
#### Pairplot for Numeric Features
Menggambarkan hubungan antar variabel numerik serta distribusinya.
```python
sns.pairplot(df[numeric_cols], diag_kind='kde', corner=True)
plt.suptitle("Pairplot for Numeric Features", y=1.02)
plt.show()
```
#### Correlation Heatmap
Menampilkan matriks korelasi antar fitur numerik untuk mengidentifikasi hubungan linier.
```python
plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
```

### Model Development & Evaluation
#### Model Training and Evaluation
Kode berikut melatih beberapa model dan menghitung metrik evaluasi seperti accuracy, precision, recall, F1 score, serta menampilkan confusion matrix:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Define models dictionary
models = {
    "Random Forest Classifier": RandomForestClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Ada Boost Classifier": AdaBoostClassifier(),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

print("Defined Models:")
print(models)

# Dictionary untuk menyimpan hasil evaluasi
results = {}

for name, model in models.items():
    # Training model
    model.fit(x_train, y_train)
    
    # Prediksi pada data testing
    y_pred = model.predict(x_test)
    
    # Hitung metrik evaluasi
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    # Simpan hasil evaluasi
    results[name] = {
       "accuracy": acc,
       "precision": prec,
       "recall": rec,
       "f1_score": f1,
       "confusion_matrix": cm
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print("  Confusion Matrix:")
    print(cm)

# Menentukan model terbaik berdasarkan akurasi
best_model_name = max(results, key=lambda x: results[x]["accuracy"])
print(f"\nBest Model by Accuracy: {best_model_name}")

# Visualisasi Confusion Matrix untuk model terbaik
best_cm = results[best_model_name]["confusion_matrix"]

plt.figure(figsize=(8, 6))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix for {best_model_name}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
```
#### Visualizing Model Accuracy Comparison
Bar chart berikut mengurutkan dan membandingkan akurasi dari setiap model:
```python
# Urutkan model berdasarkan akurasi
sorted_accuracy = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
accuracy_values = [item[1]["accuracy"] for item in sorted_accuracy]
model_names = [item[0] for item in sorted_accuracy]

plt.figure(figsize=(10, 6))
sns.barplot(x=accuracy_values, y=model_names, palette="Blues_r")
plt.xlabel("Accuracy Score")
plt.ylabel("Model")
plt.title("Model Accuracy Comparison")
plt.xlim(0.5, 1)
for index, value in enumerate(accuracy_values):
    plt.text(value - 0.05, index, f"{value:.4f}", ha='center', fontsize=12, color='black')
plt.show()
```
### Evaluation Metriks
#### Overview
# Evaluation Metrics

Pada tahap ini, kita akan membahas metrik yang digunakan untuk mengevaluasi performa model dalam melakukan klasifikasi tingkat obesitas. Metrik yang dipilih memberikan gambaran menyeluruh mengenai kekuatan model dalam mendeteksi kelas positif maupun negatif secara seimbang.

---

## Overview

Dalam proyek klasifikasi tingkat obesitas, beberapa metrik evaluasi digunakan untuk mengukur seberapa baik model memprediksi label yang benar. Metrik-metrik tersebut mencakup:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

Metrik-metrik ini membantu memastikan bahwa model tidak hanya akurat secara keseluruhan, tetapi juga memiliki keseimbangan dalam mendeteksi kelas positif dan negatif.

---

## Detail Metrik

### 1. Accuracy

- **Definisi:**  
  Persentase prediksi yang benar dari total keseluruhan data.
- **Formula:**  
  \[
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  \]
- **Penjelasan:**  
  - TP: True Positive  
  - TN: True Negative  
  - FP: False Positive  
  - FN: False Negative  
  Meskipun akurasi adalah metrik yang umum digunakan, pada kasus dengan distribusi kelas yang tidak seimbang, akurasi saja belum cukup untuk mengevaluasi kinerja model.

### 2. Precision

- **Definisi:**  
  Mengukur seberapa tepat prediksi positif yang dibuat oleh model.
- **Formula:**  
  \[
  \text{Precision} = \frac{TP}{TP + FP}
  \]
- **Penjelasan:**  
  Precision yang tinggi menandakan bahwa model jarang salah mengklasifikasikan data negatif menjadi positif. Hal ini penting pada kasus di mana kesalahan *false positive* harus diminimalkan.

### 3. Recall

- **Definisi:**  
  Mengukur kemampuan model dalam mendeteksi seluruh instance positif yang sebenarnya.
- **Formula:**  
  \[
  \text{Recall} = \frac{TP}{TP + FN}
  \]
- **Penjelasan:**  
  Recall yang tinggi menandakan bahwa model jarang gagal mendeteksi data positif (*false negative*). Hal ini penting pada kasus di mana setiap instance positif harus terdeteksi, misalnya pada diagnosa penyakit.

### 4. F1 Score

- **Definisi:**  
  Rata-rata harmonis dari Precision dan Recall, memberikan keseimbangan di antara keduanya.
- **Formula:**  
  \[
  \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  \]
- **Penjelasan:**  
  F1 Score sangat berguna ketika kita ingin mendapatkan keseimbangan antara Precision dan Recall, terutama pada data dengan distribusi kelas yang tidak seimbang.

### 5. Confusion Matrix

- **Definisi:**  
  Matriks yang menampilkan jumlah prediksi benar dan salah untuk masing-masing kelas.  
- **Penjelasan:**  
  - Baris pada confusion matrix mewakili label sebenarnya (actual class).  
  - Kolom pada confusion matrix mewakili prediksi model (predicted class).  
  Dengan menganalisis confusion matrix, kita dapat mengetahui secara spesifik kelas mana yang sering salah diprediksi oleh model.

---

## Implementasi Kode

Berikut contoh penggunaan metrik evaluasi pada data testing (y_test) dan hasil prediksi model (y_pred):

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Contoh: y_test dan y_pred sudah tersedia
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

cm = confusion_matrix(y_test, y_pred)

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))
print("\nConfusion Matrix:")
print(cm)
```
#### Penjelasan:
- accuracy_score: Menghitung persentase prediksi yang benar.
- precision_score: Mengukur ketepatan prediksi positif (menggunakan average='weighted' untuk multi-class).
- recall_score: Mengukur seberapa baik model mendeteksi semua instance positif.
- f1_score: Menggabungkan Precision dan Recall dalam satu metrik.
confusion_matrix: Menunjukkan distribusi prediksi untuk setiap kelas secara rinci

#### Conclusion
Penggunaan metrik seperti Accuracy, Precision, Recall, dan F1 Score memungkinkan kita untuk mengevaluasi model secara menyeluruh. Setiap metrik memiliki fokus yang berbeda:

- Accuracy memberikan pandangan umum seberapa sering prediksi benar.
- Precision berfokus pada ketepatan prediksi positif.
- Recall menyoroti kemampuan model dalam menangkap seluruh instance positif.
- F1 Score menggabungkan kedua aspek tersebut (Precision dan Recall).
- Confusion Matrix membantu memahami detail kesalahan model di setiap kelas.

Dengan menggabungkan semua metrik ini, kita dapat menilai kinerja model secara komprehensif dan menentukan langkah selanjutnya, seperti melakukan hyperparameter tuning atau pemilihan model lain yang lebih sesuai. Model terbaik dapat dijadikan dasar untuk sistem pendukung keputusan dalam penanganan obesitas secara lebih efektif dan tepat sasaran.