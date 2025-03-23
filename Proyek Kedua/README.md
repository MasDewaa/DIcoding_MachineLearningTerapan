# Laporan Proyek Machine Learning - Rama Syailana Dewa

## Project Overview

Topik yang saya pilih untuk proyek akhir ini adalah mengenai rekomendasi buku, dengan judul proyek **Book Recommendation System**.

<img width="773" alt="Book" src="https://github.com/user-attachments/assets/12764ab5-9872-4bd5-b498-cc93190e7c6f">



Perkembangan teknologi dan internet dalam beberapa tahun terakhir telah mengubah kebiasaan membaca masyarakat Indonesia. Kini, banyak orang lebih tertarik menggunakan media sosial seperti Instagram dan TikTok, atau menikmati hiburan di platform streaming seperti Netflix dan YouTube. Misalnya, seseorang yang dulu mungkin menghabiskan waktu membaca buku sejarah untuk belajar, sekarang lebih sering menonton video dokumenter di YouTube. Begitu pula dengan novel, yang dulu menjadi pilihan utama untuk hiburan, kini mulai tergantikan oleh serial drama di Netflix. Pergeseran ini menunjukkan bagaimana teknologi telah mengubah cara masyarakat mencari informasi dan menikmati hiburan. Membaca buku adalah salah satu cara efektif untuk memperoleh ilmu pengetahuan dan memperluas wawasan. Sebagai sumber informasi, buku memainkan peran penting dalam meningkatkan pemahaman tentang berbagai topik. Namun, di Indonesia, minat baca masyarakat tergolong rendah meskipun informasi tentang buku semakin mudah diakses melalui internet. Salah satu alasan utamanya adalah banyaknya pilihan buku yang tersedia, sehingga pembaca sering kesulitan menemukan bacaan yang sesuai dengan minat atau kebutuhannya. Faktor lain yang memengaruhi rendahnya minat baca adalah kurang menariknya kualitas banyak buku lokal dan keterbatasan akses ke buku berkualitas. Banyak buku yang tidak relevan dengan minat pembaca atau kurang menggugah rasa ingin tahu. Dalam beberapa kasus, masyarakat cenderung lebih tertarik pada buku terjemahan yang menawarkan gaya penulisan dan konten yang lebih menarik [[1](https://tirto.id/6-alasan-mengapa-minat-baca-masyarakat-indonesia-masih-rendah-gCNE)]. 

Hasil survei Programme for International Student Assessment (PISA) 2022 yang dipublikasikan oleh Databoks Katadata, skor kemampuan membaca pelajar Indonesia berada di angka 359, yang lebih rendah dibandingkan negara-negara ASEAN lainnya seperti Thailand (379), Malaysia (388), dan Brunei Darussalam (429). Indonesia hanya unggul dari Filipina (347) dan Kamboja (329), sementara Singapura menempati peringkat tertinggi dengan skor 543 [[2](https://databoks.katadata.co.id/demografi/statistik/871e4e286982d42/pisa-2022-kemampuan-membaca-pelajar-indonesia-tergolong-rendah-di-asean)]. Data ini menunjukkan bahwa tingkat literasi di Indonesia masih perlu ditingkatkan secara signifikan agar dapat bersaing dengan negara lain di kawasan Asia Tenggara. menunjukkan bahwa Indonesia berada di peringkat bawah dalam hal literasi membaca dibandingkan negara-negara lain.

Di sisi lain, perkembangan teknologi digital juga memberikan peluang baru dalam meningkatkan budaya literasi. Platform seperti e-book, audiobooks, dan aplikasi berbasis komunitas pembaca mulai mendapatkan perhatian lebih. Salah satu contohnya adalah Wattpad, yang memungkinkan penulis dan pembaca untuk berinteraksi secara langsung. Banyak cerita yang awalnya diterbitkan secara digital di Wattpad akhirnya mendapatkan popularitas dan diterbitkan dalam bentuk buku fisik, menunjukkan adanya potensi dalam memanfaatkan teknologi untuk meningkatkan minat baca [[3](https://id.quora.com/Apakah-kamu-lebih-suka-baca-buku-di-situs-Wattpad-atau-toko-buku-offline)].

Salah satu cara untuk meningkatkan minat baca di masyarakat adalah dengan mengembangkan sistem rekomendasi buku yang dapat membantu pembaca menemukan buku sesuai preferensi mereka. Sistem ini dapat dirancang menggunakan metode seperti Content-Based Filtering, yang menyarankan buku berdasarkan kesamaan konten dengan pilihan pembaca sebelumnya, serta Collaborative Filtering, yang merekomendasikan buku berdasarkan preferensi pengguna lain dengan minat serupa. Dengan menyediakan rekomendasi yang relevan dan personal, sistem ini diharapkan mampu memberikan pengalaman yang lebih memuaskan bagi pembaca, sekaligus memotivasi mereka untuk terus membaca dan mengeksplorasi lebih banyak buku yang menarik. Hal ini berpotensi meningkatkan minat baca secara keseluruhan di masyarakat.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang yang telah dijelaskan di atas, berikut ini merupakan permasalahan yang akan diselesaikan dari proyek ini : 
- Bagaimana merancang proses pengolahan data meliputi data buku, pengguna, dan penilaian agar data tersebut siap digunakan untuk membangun sistem rekomendasi berbasis machine learning ? 
- Bagaimana cara membuat model machine learning untuk sistem rekomendasi buku ?

### Goals
Berdasarkan permasalahan yang telah dirumuskan sebelumnya, tujuan dari proyek ini adalah : 
- Melakukan proses pengolahan data sehingga data dapat diproses dan siap digunakan dalam pengembangan model machine learning untuk sistem rekomendasi.  
- Merancang dan membangun model machine learning yang mampu memberikan rekomendasi buku terbaik sesuai dengan kebutuhan dan preferensi pengguna.  

### Solution statements
Untuk mencapai tujuan yang telah diuraikan di atas, maka berikut adalah beberapa solusi yang dapat dilakukan agar dapat mencapai tujuan dari proyek ini, yaitu:
- **Content-based Filtering Recommendation**

  Sistem rekomendasi yang berbasis konten (content-based filtering) merupakan sistem rekomendasi yang memberikan rekomendasi item yang hampir sama dengan item yang disukai oleh pengguna di masa lalu. Content-based filtering akan mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai oleh pengguna lain sebelumnya. Pada pendekatan menggunakan content-based filtering akan menggunakan algoritma TF-IDF Vectorizer dan Cosine Similarity.
  - TF-IDF Vectorizer
    
    Algoritma Term Frequency Inverse Document Frequency Vectorizer (TF-IDF Vectorizer) adalah algoritma yang dapat melakukan kalkulasi dan transformasi dari teks mentah menjadi representasi angka yang memiliki makna tertentu dalam bentuk matriks serta dapat digunakan dan dimengerti oleh model machine learning [[4](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)].

    Kelebihan dari teknik ini adalah tidak membutuhkan data yang diperoleh dari pengguna lain karena rekomendasi yang akan diberikan akan spesifik hanya untuk pengguna tersebut. Sedangkan kekurangan dengan menggunakan teknik ini ialah hasil rekomendasi yang hanya terbatas dari pengguna itu saja dan tidak dapat memperluas data dari penilaian pengguna lain [[5](https://towardsdatascience.com/tf-idf-simplified-aba19d5f5530)].
  - Cosine Similarity

    Teknik cosine similarity digunakan untuk melakukan perhitungan derajat kesamaan (similarity degree) antara dua sampel [[6](https://www.sciencedirect.com/topics/computer-science/cosine-similarity)].
    
- **Collaborative Filtering Recommendation**

  Sistem rekomendasi berbasis **Collaborative Filtering** berfungsi untuk memberikan saran item yang sesuai dengan preferensi pengguna di masa lalu, dengan mengandalkan data dari pengguna lain yang memiliki preferensi serupa, misalnya berdasarkan rating atau penilaian yang diberikan sebelumnya [[7](https://realpython.com/build-recommendation-engine-collaborative-filtering)]. Namun, pendekatan ini memiliki kekurangan, yakni tidak dapat merekomendasikan item yang belum pernah dinilai atau memiliki riwayat transaksi.

  Untuk menerapkan metode ini, diperlukan proses penyandian (encoding) terhadap fitur-fitur dalam dataset, mengubahnya menjadi indeks integer, dan kemudian memetakan informasi tersebut ke dalam dataframe yang sesuai. Setelah itu, dataset akan dibagi menggunakan rasio tertentu untuk memisahkan data yang digunakan untuk pelatihan (training data) dan data yang digunakan untuk pengujian (validation data), sebelum melanjutkan ke tahap pemodelan. Pada tahap ini, pembuatan model akan menggunakan kelas **RecommenderNet** dengan keras model class.

## Data Understanding
Data yang digunakan dalam proyek ini adalah *dataset* yang diambil dari Kaggle Dataset. Di bawah ini adalah informasi detail tentang *dataset* yang digunakan.

|                         | Keterangan                                                                                                                                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sumber                  | [Kaggle Dataset : Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset 'Build state-of-the-art models for book recommendation system') |
| *Usability*             | 10.00                                                                                                                                                                              |
| Lisensi                 | [CC0 : Public Domain](https://creativecommons.org/publicdomain/zero/1.0 'Creative Common - CC0 1.0 Universal')                                                                      |
| Penilaian/*Rating*      | Silver                                                                                                                                                                             |
| Jenis dan Ukuran Berkas | zip (25 MB)                                                                                                                                                                        |
| Tags              | Online Communities, Literature, Art, Recommender Systems, Culture and Humanities                                                                                                                                            |

Dalam dataset tersebut berisi tiga 3 data CSV yaitu `Books.csv`, `Ratings.csv`, `Users.csv`.

- **Books.csv**, memiliki atribut sebagai berikut :
  
  <img width="364" alt="1" src="assets\1.png">
  
  **Penjelasan :**
  - `RangeIndex` : Dataset memiliki 271,360 baris, dari indeks 0 hingga 271,359.
  - `Data Columns` : Terdapat 8 kolom dalam dataset.
  - `Non-Null Count` : Menunjukkan jumlah nilai yang tidak kosong dalam setiap kolom.
  - `Dtype` : Menunjukan Tipe data dari setiap kolom.
  - `Memory Usage` : Dataset menggunakan sekitar 16.6 MB memori di RAM.

  Penjelasan kolom : 
  - `ISBN` : Merupakan kode unik *International Standard Book Number* berupa 10 atau 13 digit yang digunakan untuk mengidentifikasi buku secara internasional. Setiap buku memiliki ISBN yang berbeda.
  - `Book-Title` : Berisi judul buku yang dimasukkan dalam dataset. Judul ini digunakan untuk mengidentifikasi isi atau nama buku.
  - `Book-Author` : Nama penulis buku. Bisa berupa satu penulis atau lebih jika buku ditulis oleh beberapa orang.
  - `Year-of-Publication` : Tahun di mana buku diterbitkan untuk pertama kalinya. Informasi ini membantu menentukan usia buku dan relevansinya.
  - `Publisher` : Nama penerbit yang bertanggung jawab atas publikasi buku tersebut. Penerbit biasanya mengelola produksi, distribusi, dan pemasaran buku.
  - `Image-URL-S` : URL untuk gambar sampul buku dengan ukuran kecil. Biasanya digunakan untuk pratinjau cepat atau thumbnail.
  - `Image-URL-M` : URL untuk gambar sampul buku dengan ukuran sedang. Cocok untuk tampilan standar pada aplikasi atau website.
  - `Image-URL-L` : URL untuk gambar sampul buku dengan ukuran besar. Berguna untuk tampilan detail. 
  
- **Ratings.csv**, memiliki atribut sebagai berikut :

  <img width="316" alt="2 0" src="assets\2.png">
  
  **Penjelasan :**
  - `RangeIndex` : Dataset memiliki 1,149,780 baris, dari indeks 0 hingga 1,149,779.
  - `Data Columns` : Terdapat 3 kolom dalam dataset.
  - `Non-Null Count` : Menunjukkan jumlah nilai yang tidak kosong dalam setiap kolom. Semua kolom memiliki 1,149,780 nilai non-null, yang berarti tidak ada nilai yang hilang (NaN).
  - `Dtype` : Menunjukan Tipe data dari setiap kolom.
  -`Memory Usage` : Dataset menggunakan sekitar 26.3 MB memori di RAM.

  Penjelasan kolom : 
  - `User-ID` : Menunjukkan ID unik yang diberikan kepada setiap pengguna yang memberikan rating untuk buku tertentu. Atribut ini digunakan untuk mengidentifikasi setiap pengguna dalam dataset.
  - `ISBN` : Merupakan kode ISBN *(International Standard Book Number)* yang digunakan untuk mengidentifikasi setiap buku secara unik. Setiap ISBN merepresentasikan satu buku yang dapat dinilai oleh pengguna. Atribut ini membantu menghubungkan rating dengan buku yang relevan.
  - `Book-Rating` : Merupakan rating yang diberikan oleh pengguna untuk buku tertentu. Nilai rating bervariasi dari 0 hingga 10, di mana nilai 0 kemungkinan menunjukkan buku yang belum dibaca atau tidak dinilai, sementara nilai yang lebih tinggi mencerminkan tingkat kepuasan pengguna terhadap buku tersebut.

  Deskripsi statistik untuk *dataset* `ratings` pada fitur `Book-Rating` dapat dilihat pada gambar di bawah ini.

  <img width="167" alt="12" src="assets\3.png">


  Dari gambar di atas dapat dilihat bahwa terdapat,
  - Total jumlah data (`count`) sebanyak 1.149.780;
  - Rata-rata *rating* (`mean`) 3;
  - Simpangan baku/standar deviasi *rating* (`std`) 4;
  - *Rating* Minimal (`min`), kuartil bawah/Q1 *rating* (`25%`), kuartil tengah/Q2/median *rating* (`50%`) 0;
  - Kuartil atas/Q3 *rating* (`75%`) 7;
  - *Rating* maksimum (`max`) 10

  Berikut adalah visualisasi grafik histogram frekuensi sebaran data *rating* pengguna terhadap buku yang sudah pernah dibaca, mulai dari *rating* terendah yaitu 1 hingga *rating* tertinggi yaitu 10.

  <img width="720" alt="13" src="assets\4.png">


  Berdasarkan hasil visualisasi grafik histogram "Jumlah Rating Buku", dapat disimpulkan bahwa rating yang paling sering diberikan pada buku yang telah dibaca adalah rating 0, dengan jumlah sekitar lebih dari 700.000. Rating 0 ini dapat menimbulkan bias dan mempengaruhi hasil analisis, sehingga data dengan rating 0 sebaiknya dihapus pada tahap persiapan data.
  
- **Users.csv**, memiliki atribut sebagai berikut :
  
  <img width="293" alt="3" src="assets\5.png">
  
  **Penjelasan :**
  - `RangeIndex` : Dataset memiliki 278,858 baris, dari indeks 0 hingga 278,857.
  - `Data Columns` : Terdapat 3 kolom dalam dataset.
  - `Non-Null Count` : Menunjukkan jumlah nilai yang tidak kosong dalam setiap kolom:
  - `Dtype` : Menunjukan Tipe data dari setiap kolom.
  -`Memory Usage` : Dataset menggunakan sekitar 6.4 MB memori di RAM.

  Penjelasan kolom :
  - `User-ID` : Identitas unik pengguna berupa bilangan bulat atau integer
  - `Location` : Lokasi tempat tinggal pengguna
  - `Age` : Umur pengguna

- **Pengecekkan Missing Value**

  Missing value adalah nilai yang hilang atau tidak ada dalam sebuah dataset. Hal ini terjadi ketika data tidak tersedia atau tidak tercatat untuk suatu entri atau atribut tertentu. Missing value sering ditemukan dalam berbagai bentuk, seperti kosong, NaN (Not a Number), atau null, dan bisa muncul karena berbagai alasan, seperti kesalahan pengumpulan data, ketidaksesuaian antara sumber data, atau kelalaian dalam pencatatan. Pengecekan *missing value* pada *dataframe* dapat dilakukan dengan menggunakan fungsi `.isnull().sum()`, yang akan menghasilkan total jumlah data yang kosong atau hilang (*missing*).

  Pada pembuatan Book Recommendation System ini beberapa data ditemuka terdapat missing value yaitu :

   - Books
     
     <img width="92" alt="m1" src="assets\6.png">
     
     Berdasarkan hasil di atas, dapat dilihat bahwa pada *dataframe* `books` terdapat beberapa atribut yang memiliki nilai kosong atau *null*, yaitu pada kolom `book_author` sebanyak 2 data, `publisher` sebanyak 2 data, dan `image_l_url` sebanyak 3 data.

   - Rating
     
     <img width="100" alt="m2" src="assets\7.png">
     
     Berdasarkan hasil di atas, dapat dilihat bahwa pada *dataframe* `ratings`, tidak ditemukan adanya nilai kosong atau *null* pada setiap kolom atau atributnya.

   - User
     
     <img width="89" alt="m3" src="assets\8.png">
     
     Dapat dilihat bahwa pada *dataframe* `users` terdapat atribut yang memiliki nilai kosong atau *null*, yaitu pada atribut `age` sebanyak 110.762 data.

- **Pengecekkan Duplicate Data**

  Pengecekan data duplikat dilakukan untuk memastikan tidak ada baris atau entri data yang muncul lebih dari sekali, yang dapat memengaruhi hasil analisis. Untuk memeriksa adanya data duplikat atau data yang sama dalam sebuah dataframe, kita dapat menggunakan fungsi `.duplicated().sum()`. Berikut ini adalah hasil pengecekan duplicate pada setiap data yang digunakan.
  
  <img width="232" alt="11" src="assets\9.png>

   Berdasarkan gambar di atas, dapat disimpulkan bahwa data telah bersih dari duplikasi. Hal ini menunjukkan bahwa setiap baris data kini bersifat unik, tanpa adanya pengulangan entri. Dengan demikian, data siap digunakan untuk analisis atau pemrosesan lebih lanjut tanpa khawatir akan bias akibat adanya data duplikat.

## Data Preparation
Pada tahap data preparation, data diolah dan ditransformasikan agar menjadi format yang sesuai untuk proses pemodelan. Tahap ini sangat penting untuk memastikan bahwa model dapat bekerja secara optimal dengan data yang bersih, terstruktur, dan relevan. Proses data preparation melibatkan beberapa langkah utama, yaitu:

- **Mengubah Nama Kolom**
  
  Perubahan nama kolom bertujuan untuk memudahkan proses pemanggilan dataframe dengan nama kolom yang lebih mudah diingat.
  - Books

    <img width="805" alt="5" src="assets\10.png">

  - Ratings

    <img width="226" alt="6" src="assets\11.png">

  - Users

    <img width="263" alt="7" src="assets\12.png">

- **Penggabungan Data ISBN**
  
  Penggabungan data ISBN buku dilakukan dengan menggunakan fungsi `.concatenate` yang disediakan oleh library numpy. Data ISBN ini ada pada dua dataframe, yaitu dataframe buku dan dataframe rating, dan penggabungan dilakukan berdasarkan kolom atau atribut isbn.
  
  <img width="296" alt="a" src="assets\13.png">

  
- **Penggabungan Data User**

  Penggabungan data user_id pada buku dilakukan dengan menggunakan fungsi `.concatenate` dari library numpy. Data user_id terdapat dalam dua dataframe, yaitu dataframe rating dan dataframe user, dan penggabungan dilakukan berdasarkan kolom atau atribut user_id.

  <img width="281" alt="b" src="assets\14.png">

- **Penggabungan Data Buku dan Rating**

  Proses penggabungan (merge) dilakukan untuk mengintegrasikan data dari dataframe buku dan dataframe rating menjadi satu dataframe yang komprehensif. Dengan langkah ini, informasi yang sebelumnya terpisah dapat digabungkan, sehingga mempermudah analisis atau pemodelan lebih lanjut.

  <img width="809" alt="18" src="assets\15.png">

- **Handling Missing value**
  
  Handling Missing Value adalah proses yang dilakukan untuk menangani data yang hilang atau tidak lengkap dalam dataset. Data yang hilang dapat menyebabkan masalah dalam analisis atau pelatihan model, sehingga perlu ditangani dengan cara yang sesuai agar hasil analisis tetap valid dan akurat. Berikut ini adalah tahapan yang dilakukan

  - Book
    
    Seperti yang kita tahu bahwa pada data ini terdapat nilai null pada kolom book_author sebanyak 2 data, publisher sebanyak 2 data, dan image_l_url sebanyak 3 data. Oleh karena itu, data yang kosong tersebut dapat dihapus dengan menggunakan fungsi .dropna().
    
    <img width="226" alt="m1 1" src="assets\16.png">
    
    Dapat dilihat dari gambar di atas setelah penghapusan, pengecekan ulang akan menunjukkan bahwa tidak ada lagi data yang kosong atau *null*.

  - Rating
    
    Pada dataframe ini memang tidak ditemukan adanya missing value didalamnya. Namun, penghapusan nilai rating 0 tetap perlu dilakukan. Hal ini karena berdasarkan hasil analisis pada tahap *data understanding*, rating 0 merupakan kategori yang paling banyak muncul, yaitu sebanyak 716.109 data. Kondisi ini berpotensi menyebabkan bias dalam analisis data. Oleh karena itu, kategori rating 0 tidak disertakan. Data tersebut tidak akan diikutsertakan ke dalam *dataframe*, sehingga data yang diambil adalah data *rating* yang lebih dari 0, yaitu *rating* 1 hingga *rating* 10 saja. Hasil visualisasi grafik histogram setelah penghapusan dapat dilihat pada gambar di bawah ini.
    
    <img width="689" alt="8" src="assets\17.png">
    
    Berdasarkan hasil visualisasi grafik histogram di atas, rating 0 telah di hapus dan distribusi frekuensi data terlihat lebih rapih dan jelas. Terutama pada rating 1 hingga rating 4.

  - User
    
    Dataframe `user` memiliki sebanyak 110.762 *missing value* pada fitur `age`, sehingga diperlukan penanganan untuk mengisi data yang hilang tersebut. Dalam kasus ini, data yang kosong akan diisi dengan nilai modus, yaitu nilai yang paling sering muncul dalam data `age`. Proses ini dilakukan menggunakan fungsi `.fillna()` dan `.mode()` untuk menggantikan *missing value* dengan nilai modus secara otomatis.
    
    <img width="299" alt="m3 1" src="assets\18.png">
    
    Berikut ini adalah hasil visualisasi grafik histogram umur.
    
    <img width="371" alt="10" src="assets\19.png">
    
    Dari grafik di atas dapat dilihat bahwa umur pengguna paling banyak berada pada rentang usia 20 hingga 30 tahun.
  
Setelah melewati tahap preparation di atas, selanjutnya adalah tahap preparation pada masing-masing pendekatan : 
1. **Content-Based Filtering**

   Pada metode content-based recommendation, dilakukan analisis terhadap deskripsi atau informasi terkait item untuk menemukan item lain yang serupa berdasarkan konten. Salah satu pendekatan yang digunakan adalah dengan menerapkan TF-IDF (Term Frequency-Inverse Document Frequency). *Term Frequency Inverse Document Frequency Vectorizer* `TF-IDF Vectorizer` *Algorithm* merupakan algoritma yang dapat melakukan kalkulasi dan transformasi dari teks mentah menjadi representasi angka yang memiliki makna tertentu dalam bentuk matriks serta dapat digunakan dan dimengerti oleh model *machine learning*. Berikut adalah tahapan yang dilakukan :

   - Inisialisasi dan Penerapan TF-IDF Vectorizer

     Pada tahap pertama, menginisialisasi objek `TfidfVectorizer()` untuk mengubah data teks, dalam hal ini penulis buku (`book_author`), menjadi representasi numerik berbasis TF-IDF. Proses ini bertujuan untuk menilai pentingnya setiap kata dalam konteks keseluruhan kumpulan buku.
     
     <img width="316" alt="f" src="assets\20.png">

   - Membuat Matriks TF-IDF

     Selanjutnya, menerapkan TF-IDF pada kolom penulis buku (`book_author`) untuk membuat matriks TF-IDF yang merepresentasikan setiap penulis dalam bentuk vektor. Matriks ini akan memiliki dimensi sesuai dengan jumlah kata unik yang ditemukan pada data.
     
     <img width="694" alt="g" src="assets\21.png">

   - Konversi Matriks TF-IDF ke Bentuk Dense

     Matriks TF-IDF yang terbentuk pada tahap sebelumnya disimpan dalam bentuk sparse matrix. Untuk mempermudah interpretasi, selanjutnya mengonversinya menjadi matriks padat (dense matrix) agar dapat lebih mudah dianalisis.
     
     <img width="407" alt="h" src="assets\22.png">

   - Melihat Matriks TF-IDF

     Matriks TF-IDF yang telah dibentuk kemudian dapat divisualisasikan menggunakan `pandas.DataFrame` untuk melihat bagaimana kata-kata pada penulis buku terdistribusi. Kolom-kolom mewakili kata-kata unik (fitur) yang ada dalam data, sementara baris mewakili judul buku. Hasil ini memungkinkan kita untuk memeriksa seberapa signifikan setiap kata untuk setiap buku.
     
     <img width="768" alt="i" src="assets\23.png">

2. **Collaborative Filtering**

   Sistem rekomendasi penyaringan kolaboratif (*Collaborative Filtering Recommendation*) adalah teknik yang memberikan rekomendasi item berdasarkan preferensi pengguna di masa lalu, misalnya dengan menggunakan *rating* yang telah diberikan oleh pengguna, serta menyarankan item yang mirip dengan pola preferensi pengguna lainnya. Berikut adalah tahapan yang dilakukan :
     - Melakukan encoding pada `user_id`
       
       Encoding adalah proses konversi data kategorikal (seperti ID atau label dalam bentuk teks) menjadi format numerik, yang diperlukan oleh algoritma pembelajaran mesin (machine learning). Mengubah ID pengguna yang bersifat unik dan kategorikal menjadi angka. Hal ini membantu model untuk memahami dan mengolah interaksi antara pengguna dan buku. Misalnya, pengguna dengan ID `user_1` bisa diwakili dengan angka 0, `user_2` dengan angka 1, dan seterusnya.
       
       <img width="927" alt="j" src="assets\24.png">

     - Melakukan encoding pada `ISBN`
       
       Sama seperti `user_id`, ID buku (ISBN) yang bersifat kategorikal diubah menjadi angka. Setiap buku diberikan ID unik berbentuk angka, yang memudahkan model dalam mengidentifikasi dan memproses hubungan antara pengguna dan buku.
       
       <img width="924" alt="k" src="assets\25.png">

     - Memetakan `user_id` dan `isbn` ke dalam data frame

       Setelah memiliki kamus encoding untuk pengguna dan buku, langkah selanjutnya adalah mengganti nilai pada kolom user_id dan isbn di dataset dengan ID numerik yang sudah kita buat.
       
       <img width="286" alt="l" src="assets\26.png">

     - Pengecekan jumlah user, jumlah buku, dan rating minimal & maksimal

       Setelah encoding selesai, selanjutnya dapat menampilkan informasi dasar mengenai data, seperti jumlah pengguna dan buku yang ada dalam dataset, serta rating minimum dan maksimum yang terdapat pada dataset.
       
       <img width="561" alt="m" src="assets\27.png">

     - Shuffling the Data (pengacakan)
       
       Data yang diacak (shuffled) digunakan untuk memastikan bahwa data yang digunakan untuk melatih model tidak terurut atau terstruktur dalam cara yang bisa menimbulkan bias. Ini membantu memastikan model belajar dari data yang beragam.
       
       <img width="274" alt="n" src="assets\28.png">

     - Memisahkan variabel (X) dan target (Y)
       
        <img width="442" alt="o" src="assets\28(2).png">

     - Membagi data menjadi training (80) dan validation (20)

       Pembagian data dilakukan untuk melatih model pada training set dan mengevaluasi kinerjanya pada validation set. Data dibagi menjadi 80% untuk training dan 20% untuk validation.
       
       <img width="416" alt="p" src="assets\29.png">

       
## Modeling and Result
Tahap berikutnya adalah membangun model machine learning yang berfungsi sebagai sistem rekomendasi untuk memberikan rekomendasi buku terbaik kepada pengguna, menggunakan beberapa algoritma sistem rekomendasi.  

1. **Content-Based Filtering**
   
   Content-based filtering adalah teknik rekomendasi yang mengandalkan informasi atau atribut dari item untuk merekomendasikan item yang serupa berdasarkan preferensi atau interaksi pengguna. Salah satu metode yang sering digunakan dalam content-based filtering adalah Cosine Similarity, yang mengukur seberapa mirip dua item berdasarkan fitur mereka misalnya, deskripsi atau kategori buku.
   
   Cosine Similarity digunakan untuk mengukur tingkat kesamaan antara dua vektor, dalam hal ini adalah representasi buku yang dihasilkan dari TF-IDF. Nilainya berkisar antara -1 hingga 1, di mana:
     - 1 menunjukkan kesamaan penuh
     - 0 menunjukkan tidak ada kesamaan
     - -1 menunjukkan perbedaan penuh.
       
     Untuk melakukan perhitungan derajat kesamaan (*similarity degree*) antar judul buku dapat dilakukan dengan teknik *cosine similarity* menggunakan fungsi `cosine_similarity` dari library `sklearn`.
   
   <img width="317" alt="q" src="assets\30.png">

     Metode ini sangat berguna dalam Content-Based Filtering karena membantu menghitung kemiripan antara buku yang berbeda berdasarkan atribut seperti deskripsi atau genre. Cosine Similarity akan melakukan perhitungan derajat kesamaan (similarity degree) antar judul buku. Ukuran matriks yang diperoleh adalah sebesar 10.000 data buku dan 10.000 data buku juga.
   
     <img width="186" alt="r" src="assets\31.png">
     
     
     <img width="767" alt="15" src="assets\32.png">

     Selanjutnya untuk membuat fungsi rekomendasi. Top-N Recommendation adalah langkah akhir dalam sistem rekomendasi di mana algoritma memilih sejumlah buku (N) dengan nilai kesamaan tertinggi terhadap buku yang sedang dicari atau yang sudah dinikmati oleh pengguna. Buku-buku ini disusun berdasarkan skor kesamaan, sehingga sistem dapat memberikan rekomendasi yang paling relevan dan menarik bagi pengguna. Hasil pengujian sistem rekomendasi dengan pendekatan content-based recommendation adalah sebagai berikut.

     <img width="934" alt="s" src="assets\33.png">

     Gambar diatas menunjukan data berdasarkan detail buku yang telah dibaca berdasarkan readed_book_title yaitu `Proxies. berikut ini pencarian yang direkomendasikan yang paling mirip dengan buku "Proxies" : 
  
     <img width="290" alt="17" src="assets\34.png">

     Sistem yang telah dibangun berhasil memberikan rekomendasi beberapa judul buku berdasarkan input judul buku "Proxies", dan menghasilkan daftar buku yang relevan berdasarkan perhitungan yang dilakukan oleh sistem.
   
   Kelebihan dan kekurangan Content-based Filtering : 
       
       Kelebihan :
       1. Rekomendasi sangat dipersonalisasi berdasarkan preferensi pengguna sebelumnya, yang memastikan relevansi yang lebih tinggi.
       2. Tidak bergantung pada perilaku pengguna lain, sehingga bisa memberikan rekomendasi meskipun data pengguna terbatas.
       3. Mudah diimplementasi.
      
       Kekurangan :
       1. Hanya bisa merekomendasikan item yang mirip dengan yang sudah disukai pengguna, sehingga tidak bisa memberikan rekomendasi yang lebih beragam atau eksploratif.
       2. Sistem cenderung merekomendasikan item yang serupa dengan yang telah ada, yang bisa membatasi variasi rekomendasi dan tidak membantu pengguna menemukan item yang berbeda atau baru.
       3. Jika item baru tidak memiliki cukup data atau deskripsi, sistem kesulitan memberikan rekomendasi yang relevan.
     
3. **Collaborative Filtering**

   Collaborative Filtering adalah teknik rekomendasi yang memberikan saran item kepada pengguna berdasarkan preferensi pengguna lain yang memiliki kesamaan pola atau perilaku dengan pengguna tersebut. Teknik ini biasanya menggunakan data seperti rating yang diberikan oleh pengguna terhadap item misalnya buku untuk mengidentifikasi pola atau kesamaan dengan pengguna lainnya. Kemudian, item yang disukai oleh pengguna yang memiliki kesamaan preferensi akan direkomendasikan kepada pengguna yang belum memilih atau memberi rating pada item tersebut. Pada tahap pembuatan model akan menggunakan kelas `RecommenderNet` dengan `keras model class`
   
   - RecommenderNet

     RecommenderNet adalah model Neural Collaborative Filtering yang dirancang untuk memberikan rekomendasi kepada pengguna berdasarkan interaksi mereka dengan item (seperti buku). Model ini menggunakan embedding untuk memetakan pengguna dan item ke dalam ruang vektor berdimensi rendah, sehingga memungkinkan model untuk menangkap hubungan yang lebih kompleks antara keduanya. Proses utamanya melibatkan vektor embedding untuk pengguna dan item yang dihitung melalui produk titik (dot product), kemudian ditambahkan dengan bias pengguna dan item untuk mengakomodasi preferensi atau popularitas yang lebih umum. Hasil dari interaksi ini diproses dengan fungsi aktivasi sigmoid, yang menghasilkan nilai antara 0 dan 1, mencerminkan kemungkinan apakah seorang pengguna akan menyukai atau berinteraksi dengan item tertentu. Model ini efektif dalam memahami pola preferensi pengguna dan memberikan rekomendasi yang relevan berdasarkan data historis yang ada.
     
     <img width="461" alt="t" src="assets\35.png">

   - Recommendation Testing
     
     Berdasarkan model yang telah dilatih, berikut ini adalah hasil evaluasi dari sistem rekomendasi buku yang menggunakan pendekatan collaborative filtering recommendation.
     
     <img width="425" alt="u" src="assets\36.png">

     <img width="652" alt="21" src="assets\37.png">

      Berdasarkan hasil di atas, dapat dilihat bahwa sistem akan mengambil pengguna secara acak, yaitu pengguna dengan `user_id` **1178**. Selanjutnya, sistem akan mencari buku dengan *rating* tertinggi dari pengguna tersebut, yaitu:

      *   **Way Out West Lives a Coyote Named Frank (Picture Puffins)** oleh **Jillian Lund**  
      *   **Island of the Blue Dolphins (Laurel Leaf Books)** oleh **Scott O'Dell**  
      *   **How to Fight a Girl** oleh **THOMAS ROCKWELL**  
      *   **Romantic Obsessions and Humiliations of Annie Sehlmeier ** oleh **Louise Plummer**  

      Kemudian, sistem akan membandingkan antara buku dengan *rating* tertinggi dari pengguna tersebut dan semua buku lainnya yang belum pernah dibaca, lalu mengurutkan buku yang akan direkomendasikan berdasarkan nilai prediksi rekomendasi tertinggi. Terdapat 10 daftar buku yang direkomendasikan oleh sistem, yaitu:

      *   **The Secret Life of Bees** oleh **Sue Monk Kidd**  
      *   **The Red Tent (Bestselling Backlist)** oleh **Anita Diamant**  
      *   **Life of Pi** oleh **Yann Martel**  
      *   **Angela's Ashes: A Memoir** oleh **Frank McCourt**  
      *   **Les Fourmis** oleh **Bernard Werber**  
      *   **My Grandfathers Blessings : Stories of Strength, Refuge, and Belonging** oleh **Rachel Naomi Remen M.D.**  
      *   **The Grapes of Wrath: John Steinbeck Centennial Edition (1902-2002)** oleh **John Steinbeck**  
      *   **The Handmaid's Tale** oleh **Margaret Atwood**  
      *   **The Golden Mean: In Which the Extraordinary Correspondence of Griffin &amp; Sabine Concludes** oleh **Nick Bantock**  
      *   **The Watsons Go to Birmingham - 1963 (Yearling Newbery)** oleh **CHRISTOPHER PAUL CURTIS**  

      Dapat dibandingkan antara daftar ***Book with high ratings from user*** dan ***Top 10 Books Recommendation***, terdapat beberapa kesesuaian pola rekomendasi berdasarkan preferensi pengguna. Hal ini menunjukkan bahwa sistem yang dibangun dapat memberikan rekomendasi buku kepada pengguna dengan hasil yang relevan dan sesuai prediksi.

     Kelebihan dan kekurangan Collaborative Filtering : 
       
         Kelebihan :
         1. Tidak membutuhkan pemahaman mendalam mengenai konten item yang dianalisis.
         2. Dapat Menemukan Rekomendasi Baru.
         3. Bisa memberikan rekomendasi yang lebih beragam dan tidak terbatas pada item serupa dengan yang sudah ada.
      
         Kekurangan :
         1. Sistem kesulitan memberikan rekomendasi untuk pengguna baru (user cold start) atau item baru (item cold start) yang tidak memiliki cukup data interaksi untuk dianalisis.
         2. Ketika jumlah item sangat besar, komputasi untuk mencari kesamaan antar pengguna atau item bisa menjadi sangat berat dan memerlukan waktu serta sumber daya yang besar.
         3. Jika item sedikit sistem bisa kesulitan dalam menghasilkan rekomendasi yang akurat.

       
## Evaluation
1. **Content-Based Filtering**

   Sistem rekomendasi Content-Based Filtering yang dibangun berhasil memberikan rekomendasi buku berdasarkan kemiripan konten antara buku yang telah dibaca pengguna dengan buku lainnya. Teknik Evaluasi yang digunakan untuk content-based filtering adalah dengan menggunakan precission, rumus dari teknik ini adalah :
   
   $$P = \frac{\text{Jumlah rekomendasi yang relevan}}{\text{Jumlah total item yang direkomendasikan}}$$

   **Penjelasan:**
   - **Jumlah rekomendasi yang relevan:** Ini adalah jumlah item yang direkomendasikan oleh sistem yang sesuai dengan kebutuhan atau preferensi pengguna. Artinya, item ini dianggap bermanfaat atau relevan oleh pengguna.
   - **Jumlah total** item yang direkomendasikan: Ini adalah jumlah keseluruhan item yang direkomendasikan oleh sistem, termasuk yang relevan maupun yang tidak relevan.
     
   Precision adalah salah satu metrik evaluasi yang digunakan untuk mengukur keakuratan rekomendasi dalam sistem rekomendasi. Precision menunjukkan proporsi item yang direkomendasikan dan relevan dibandingkan dengan jumlah total item yang direkomendasikan.

   Masih menggunakan data yang sama pada tahap Modeling content-based recommendation, pada proses hasil Top-N Recommendation, akan dilakukan proses pencarian judul buku atau book_title yang memiliki kemiripan berdasarkan data buku yang telah dibaca oleh pengguna, yaitu 'Proxies'. Hasil dari Top-N Recommendation mendapatkan beberapa rekomendasi judul buku seperti pada tabel berikut:

   <img width="290" alt="17" src="assets\38.png">

   Dari hasil rekomendasi di atas, diketahui bahwa proses rekomendasi berhasil menghasilkan beberapa judul buku yang memiliki kemiripan tertentu dengan buku *'Proxies'*, berdasarkan analisis kesamaan konten melalui penghitungan **Cosine Similarity**. Proses ini menggunakan representasi vektor dari data buku untuk menentukan tingkat kemiripan antarjudul, menghasilkan rekomendasi dengan skor **precision** mencapai **88%**. Skor ini menunjukkan kemampuan sistem untuk secara akurat mengidentifikasi buku yang relevan sesuai dengan preferensi pengguna, memberikan dasar yang kuat untuk pengembangan sistem rekomendasi yang lebih optimal.
   
3. **Collaborative Filtering**

   Berdasarkan model machine learning yang telah dibangun menggunakan *embedding layer* dengan optimizer Adam dan fungsi loss binary crossentropy, metrik yang digunakan untuk evaluasi adalah Root Mean Squared Error (RMSE). Perhitungan RMSE dapat dilakukan menggunakan rumus berikut :
   
   $$RMSE=\sqrt{\sum^{n}_{i=1} \frac{y_i - y\\_pred_i}{n}}$$
   
   Di mana, nilai $n$ merupakan jumlah *dataset*, nilai $y_i$ adalah nilai sebenarnya, dan $y\\_pred$ yaitu nilai prediksinya terdahap $i$ sebagai urutan data dalam *dataset*.
   
   Nilai RMSE yang rendah menunjukkan bahwa perbedaan antara nilai yang diprediksi oleh model dan nilai observasi yang sebenarnya sangat kecil. Dengan kata lain, semakin kecil nilai RMSE, semakin akurat prediksi model dibandingkan dengan data asli. Berikut ini adalah visualisasi dari hasil training dan validation error menggunakan metrik RMSE, serta grafik yang menunjukkan training dan validation loss selama proses pelatihan.

   <img width="719" alt="22" src="assets\39.png">

   Secara keseluruhan, grafik diatas menunjukkan bahwa model berhasil mempelajari pola dari data dan dapat memberikan hasil yang baik pada data latih maupun data validasi. Penurunan yang stabil pada RMSE dan loss mengindikasikan bahwa model semakin akurat dalam memprediksi hasil.

## Kesimpulan
Kesimpulannya, model yang dibangun untuk merekomendasikan buku menggunakan dua pendekatan, yaitu **Content-based Recommendation** dan **Collaborative Filtering Recommendation**, telah berhasil dikembangkan dan mampu memberikan rekomendasi yang sesuai dengan preferensi pengguna. Pada pendekatan **Collaborative Filtering**, sistem membutuhkan data rating yang diberikan oleh pengguna untuk menentukan kesamaan preferensi antara pengguna yang satu dengan pengguna lainnya, dan berdasarkan informasi ini, rekomendasi dapat diberikan. Sementara itu, pada pendekatan **Content-based Filtering**, data rating tidak diperlukan. Sistem ini mengandalkan analisis terhadap atribut atau konten dari masing-masing buku, seperti genre, deskripsi, dan penulis, untuk memberikan rekomendasi yang relevan berdasarkan buku yang sudah dibaca oleh pengguna.

Kedua pendekatan tersebut memiliki kelebihan dan kekurangannya masing-masing. **Collaborative Filtering** cenderung memberikan rekomendasi yang lebih beragam karena mempertimbangkan pola preferensi pengguna lain, namun dapat menghadapi masalah ketika data rating terbatas atau baru. Di sisi lain, **Content-based Filtering** memberikan rekomendasi yang lebih spesifik berdasarkan atribut buku, namun cenderung membatasi keberagaman karena hanya merekomendasikan buku yang serupa dengan yang sudah dibaca sebelumnya. Meskipun begitu, kedua teknik ini saling melengkapi dan dapat memberikan sistem rekomendasi yang lebih efektif jika digabungkan.

## Referensi
[1]. (https://tirto.id/6-alasan-mengapa-minat-baca-masyarakat-indonesia-masih-rendah-gCNE) Sulthoni. -*6 Alasan Mengapa Minat Baca Masyarakat Indonesia Masih Rendah*. tirto.id. https://tirto.id/6-alasan-mengapa-minat-baca-masyarakat-indonesia-masih-rendah-gCNE

[2]. (https://databoks.katadata.co.id/demografi/statistik/871e4e286982d42/pisa-2022-kemampuan-membaca-pelajar-indonesia-tergolong-rendah-di-asean) katadata. -*PSkor Kemampuan Membaca Pelajar ASEAN menurut PISA (2022)*. katadata. https://databoks.katadata.co.id/demografi/statistik/871e4e286982d42/pisa-2022-kemampuan-membaca-pelajar-indonesia-tergolong-rendah-di-asean

[3]. (https://id.quora.com/Apakah-kamu-lebih-suka-baca-buku-di-situs-Wattpad-atau-toko-buku-offline) Restu I Aji -*Apakah kamu lebih suka baca buku di situs Wattpad atau toko buku offline?*. Quora. https://id.quora.com/Apakah-kamu-lebih-suka-baca-buku-di-situs-Wattpad-atau-toko-buku-offline

[4]. (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) Scikit Learn -*TfidfVectorizer*. Scikit Learn. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

[5]. (https://towardsdatascience.com/tf-idf-simplified-aba19d5f5530) Luthfi Ramadhan -*TF-IDF Simplified*. Medium. https://towardsdatascience.com/tf-idf-simplified-aba19d5f5530

[6]. (https://www.sciencedirect.com/topics/computer-science/cosine-similarity) ScienceDirect -*Cosine Similarity*. ScienceDirect. https://www.sciencedirect.com/topics/computer-science/cosine-similarity

[7]. (https://realpython.com/build-recommendation-engine-collaborative-filtering)  Abhinav Ajitsaria -*Build a Recommendation Engine With Collaborative Filtering*. Real Python. https://realpython.com/build-recommendation-engine-collaborative-filtering