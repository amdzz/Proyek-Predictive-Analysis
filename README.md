# Laporan Proyek Machine Learning - Ahmad Zul Zhafran

## Domain Proyek

Kualitas udara merupakan isu lingkungan yang semakin krusial, terutama di wilayah perkotaan seperti Jakarta, yang menghadapi tantangan polusi udara akibat aktivitas industri, transportasi, dan pertumbuhan populasi. Data Indeks Standar Pencemaran Udara (ISPU) dari tahun 2010 hingga 2021 menunjukkan variasi tingkat polutan seperti PM10, PM2.5, SO2, CO, O3, dan NO2, yang memengaruhi kesehatan masyarakat dan lingkungan. Pemantauan dan analisis kualitas udara menjadi penting untuk memahami pola polusi, mengidentifikasi faktor utama pencemaran, dan mendukung pengambilan kebijakan berbasis data untuk mitigasi dampaknya. Proyek ini bertujuan untuk menganalisis dataset ISPU DKI Jakarta menggunakan teknik pembelajaran mesin guna memprediksi kategori kualitas udara dan mengevaluasi kontribusi polutan terhadap tingkat pencemaran.

Penerapan model pembelajaran mesin seperti Random Forest dan XGBoost dalam proyek ini memungkinkan analisis mendalam terhadap hubungan antar polutan dan prediksi kategori kualitas udara secara akurat. Dengan memanfaatkan dataset historis, proyek ini juga mengeksplorasi pentingnya fitur polutan melalui visualisasi seperti heatmap korelasi dan diagram pentingnya fitur, serta menangani ketidakseimbangan data dengan teknik SMOTE. Hasil dari proyek ini diharapkan dapat memberikan wawasan tentang pola kualitas udara di Jakarta, mendukung upaya pencegahan polusi, dan menjadi dasar untuk pengembangan sistem prediksi kualitas udara yang lebih canggih di masa depan.  Firdaus et al. (2024) telahmengimplementasikan klasifikasi di DKI Jakarta dengan menggunakan algoritma Random Forest. Berdasarkan penelitian tersebut didapatkan hasil yang memuaskan dimana pada data train mendapatkan nilai precision, recall, dan F1-score yang sempurna yaitu 100% disemua kelas dan AUC juga sebesar 100%. Kemudian Nababan A. A. et al. (2023) telah mengimplementasikan teknik XGBoost dengan SMOTE untuk melakukan klasifikasi kualitas udara. Hasil yang didapat menunjukkan performa XGBoost yang sangat baik dalam klasifikasi kualitas udara degngan rata-rata akurasi yang dihasilkan sebesar 98%.

Firdaus, R., Husnul Habibie, & Yoze Rizki. (2024). Implementasi Algoritma Random Forest Untuk Klasifikasi Pencemaran Udara di Wilayah Jakarta Berdasarkan Jakarta Open Data. Jurnal FASILKOM, 14(2), 520-525 1824-1834, 2023. 

Nababan, A. A., Jannah, M., Aulina, M., & Andrian, D. (2023). PREDIKSI KUALITAS UDARA MENGGUNAKAN XGBOOST DENGAN SYNTHETIC MINORITY OVERSAMPLING TECHNIQUE (SMOTE) BERDASARKAN INDEKS STANDAR PENCEMARAN UDARA (ISPU). JTIK (Jurnal Teknik Informatika Kaputama), 7(1), 214-219.

## Business Understanding

### Problem Statements

- Pernyataan Masalah 1
Lembaga pemerintahan atau lingkungan memerlukan sistem otomatis yang mampu memprediksi atau mengklasifikasikan kualitas udara berdasarkan data sensor, untuk mendukung pengambilan keputusan yang cepat dan akurat.
- Pernyataan Masalah 2
Informasi kualitas udara sering kali disajikan dalam bentuk angka konsentrasi polutan (seperti PM10, SO2, CO), yang sulit diinterpretasikan langsung oleh masyarakat umum atau pengambil kebijakan.
- Pernyataan Masalah 3
Kualitas udara dipengaruhi oleh kombinasi kompleks dari berbagai faktor seperti emisi kendaraan, aktivitas industri, kondisi cuaca, dan topografi wilayah. Analisis manual terhadap hubungan antar faktor ini kurang efisien dan rawan kesalahan
- Pernyataan Masalah 4
Dalam banyak kasus, data yang tersedia memiliki distribusi kelas yang tidak seimbang-misalnya lebih banyak data pada kategori "Baik" dibandingkan "Sangat Tidak Sehat"-yang menyebabkan model machine learning cenderung bias dan tidak optimal dalam memprediksi kategori minoritas.

### Goals

- Membangun model machine learning yang mampu mengklasifikasikan kualitas udara (Baik, Sedang, Tidak Sehat, dst.) berdasarkan data polutan seperti PM10, SO2, CO, O3, dan NO2.
- Mengubah data numerik konsentrasi polutan menjadi kategori kualitas udara yang mudah dipahami oleh masyarakat umum dan pembuat kebijakan.
- Menganalisis feature importance dari model Random Forest dan XGBoost untuk mengetahui zat polutan mana yang paling memengaruhi kategori kualitas udara.
- engimplementasikan metode oversampling (seperti SMOTE) untuk mengatasi distribusi kelas yang timpang dan meningkatkan akurasi prediksi pada kelas minoritas.

### Solution statements
- Menerapkan Dua Algoritma Machine Learning: Random Forest dan XGBoost. Model ini dilatih menggunakan data polutan dengan pembagian train-test dan evaluasi menggunakan metrik seperti accuracy, f1-score, dan recall. Model terbaik dipilih berdasarkan kinerja evaluasi.
- Setelah pelatihan, dilakukan analisis terhadap bobot pentingnya masing-masing fitur (polutan) untuk mengetahui pengaruh relatif mereka terhadap kualitas udara.
- Menggunakan SMOTE (Synthetic Minority Over-sampling Technique) untuk memperbanyak data dari kelas minoritas sebelum pelatihan model, sehingga meningkatkan generalisasi model dan performa pada kategori yang jarang.
- Evaluasi dilakukan dengan metrik seperti accuracy, f1_score (weighted), recall, dan visualisasi confusion matrix untuk memastikan model bekerja baik di semua kelas.

## Data Understanding
Proyek ini dimulai dengan pengumpulan data kualitas udara yang 
bersumber dari [Kaggle](https://www.kaggle.com/datasets/senadu34/air-quality-index-in-jakarta-2010-2021/), yang mencakup data pengukuran kualitas udara di Jakarta dengan interval waktu per hari. Data ini mencakup beragam parameter penting yang menjadi indikator utama dalam penilaian kualitas udara, termasuk konsentrasi partikel PM10 dan PM2.5, serta gas-gas polutan seperti SO2, CO, O3, dan NO2. Selain itu, dataset ini menyertakan informasi tambahan seperti kategori kualitas udara yang menunjukkan tingkat polusi udara berdasarkan konsentrasi tertinggi polutan, serta kolom tanggal dan waktu pengukuran, komponen kritis yang paling dominan, dan nilai maksimum dari masing-masing polutan. Pada tahapan ini juga dilakukan proses EDA dengan visualisasi data secara univariat maupun multivariat.

### Variabel-variabel pada Air Quality Index Jakarta adalah sebagai berikut:
- tanggal: Tanggal ketika pengukuran Indeks Kualitas Udara (AQI) dicatat.
- stasiun: Nama atau identitas dari stasiun pemantauan tempat pengukuran dilakukan.
- pm25: Konsentrasi partikel dengan diameter 2.5 mikrometer atau kurang (PM2.5), diukur dalam mikrogram per meter kubik udara (µg/m³).
- pm10: Konsentrasi partikel dengan diameter 10 mikrometer atau kurang (PM10), diukur dalam mikrogram per meter kubik udara (µg/m³).
- so2: Konsentrasi gas sulfur dioksida (SO2), diukur dalam satuan part per million (ppm).
- co: Konsentrasi gas karbon monoksida (CO), diukur dalam part per million (ppm).
- o3: Konsentrasi gas ozon (O3), diukur dalam part per million (ppm).
- no2: Konsentrasi gas nitrogen dioksida (NO2), diukur dalam part per million (ppm).
- max: Nilai tertinggi dari seluruh polutan pada tanggal dan stasiun tersebut. Nilai ini menunjukkan konsentrasi tertinggi dari PM2.5, PM10, SO2, CO, O3, dan NO2.
- critical: Jenis polutan yang memiliki konsentrasi tertinggi pada tanggal dan stasiun tersebut.
- categori: Kategori kualitas udara berdasarkan nilai "max", yang menggambarkan tingkat kebersihan atau pencemaran udara.

![image](https://github.com/user-attachments/assets/780ef9c5-5c94-45b5-ac6c-508e166ea523)

Pada tahapan ini, dataset yang digunakan terdiri dari 5538 baris dan 11 kolom, memberikan gambaran awal mengenai skala data yang dianalisis. Jumlah missing values yang teridentifikasi meliputi pm10 (311), pm25 (4018), so2 (126), co (84), o3 (100), no2 (102), dan categori (0), yang menunjukkan tingkat ketidaklengkapan data pada beberapa variabel, terutama pm25, yang dapat memengaruhi analisis lebih lanjut. Data menunjukkan dua duplikasi ditemukan sehingga ditindak dengan cara dihapus. 

![image](https://github.com/user-attachments/assets/7f0d66eb-efea-4de9-a64d-c53afe498616)

Visualisasi boxplot menunjukkan adanya outlier pada variabel pm10 (242), so2 (46), co (240), o3 (188), dan no2 (198), namun outlier ini masih berada dalam rentang normal berdasarkan distribusi data, sehingga tidak perlu dihapus. Untuk analisis selanjutnya, akan digunakan model Random Forest (RF) dan XGBoost, yang dikenal robust terhadap outlier, sehingga potensi dampak dari nilai ekstrem dalam dataset dapat diminimalkan secara alami oleh model.

![dist](https://github.com/user-attachments/assets/5430adf7-7d1e-44b0-97b1-5a99528fce5b)

Gambar tersebut merupakan grafik visualisasi dari distribusi kolom kategori pada dataset. Pada sumbu horizontal (x-axis), terdapat kategori kualitas udara, yaitu ‘SEDANG’, ‘BAIK’, ‘TIDAK SEHAT’, ‘SANGAT TIDAK SEHAT’, ‘BERBAHAYA’, dan ‘TIDAK ADA DATA’. Sementara itu, sumbu vertikal (y-axis) menunjukkan jumlah kemunculan masing-masing kategori dalam 
dataset. Dari grafik, dapat disimpulkan bahwa: 
1. Kategori sedang memiliki jumlah data tertinggi. 
2. Kategori tidak sehat merupakan kategori kedua terbanyak. 
3. Kategori baik, sangat tidak sehat, dan berbahaya memiliki jumlah data 
yang jauh lebih rendah. 
4. Kategori tidak ada data mencerminkan adanya data yang tidak memiliki 
informasi terkait kategori kualitas udara. 
5. Kategori tidak ada data dan berbahaya memiliki jumlah data yang sangat 
sedikit. Sehingga diputuskan untuk menghapus kategori berbahaya.

![corr](https://github.com/user-attachments/assets/c881bf1c-df3f-46de-a198-02fd30df1f28)

Gambar yang ditampilkan di atas adalah heatmap korelasi antar polutan udara, yang menunjukkan hubungan linear antara setiap pasangan variabel polutan seperti pm2.5, pm10, so2, co, o3, dan no2. Nilai korelasi berkisar dari -1 hingga 1, di mana:

1. Nilai mendekati 1 menunjukkan korelasi positif kuat (jika satu naik, yang lain juga cenderung naik),
2. Nilai mendekati -1 menunjukkan korelasi negatif kuat (jika satu naik, yang lain cenderung turun),
3. Nilai mendekati 0 menunjukkan korelasi lemah atau tidak ada korelasi linear.


## Data Preparation
Tahap data preparation dilakukan untuk memastikan bahwa data yang digunakan dalam pelatihan model sudah bersih, valid, dan relevan. Proses ini dimulai dengan pemilihan fitur-fitur yang dianggap paling berpengaruh terhadap kualitas udara, yaitu pm10, pm25, so2, co, o3, no2, serta kolom categori sebagai label klasifikasi. Pemilihan ini bertujuan untuk menyederhanakan data dan hanya mempertahankan informasi yang diperlukan dalam proses pemodelan. Selanjutnya, dilakukan pembersihan data dengan menghapus baris-baris yang memiliki kategori TIDAK ADA DATA. Hal ini penting karena nilai tersebut tidak merepresentasikan kualitas udara yang sebenarnya, sehingga dapat mengganggu proses pelatihan model dan menghasilkan prediksi yang tidak akurat. Setelah itu, dilakukan pengecekan terhadap nilai-nilai kosong (missing values) pada setiap kolom menggunakan fungsi isnull().sum(). Tujuannya adalah untuk memastikan bahwa tidak ada fitur yang memiliki nilai kosong yang dapat mempengaruhi proses pelatihan model. Kemudian data dengan label kategori BERBAHAYA juga dihapus karena jumlah datanya sangat sedikit (hanya satu baris), sehingga tidak cukup untuk dilakukan proses resampling seperti SMOTE.

### 1. Penanganan Missing Values
Dilakukan pengecekan terhadap missing values menggunakan fungsi isnull().sum() untuk mengetahui jumlah nilai kosong pada setiap kolom. Jumlah missing values yang teridentifikasi adalah sebagai berikut:
- pm10: 311
- pm25: 4018
- so2: 126
- co: 84
- o3: 100
- no2: 102
- categori: 0

Untuk menangani missing values tersebut, dilakukan proses imputasi menggunakan nilai mean dari masing-masing kolom. Keberadaan label yang sangat jarang justru bisa menyebabkan error atau bias pada model. Dikarenakan missing values yang ditemukan pada kolom pm25 mencapai 80% dari jumlah total data, maka diputuskan untuk menghapus kolom pm25 ini.

### 2. Encoding
Kolom kategori yang semula bertipe object dikonversi menjadi nilai numerik menggunakan LabelEncoder, menghasilkan variabel y_encoded sebagai target. Proses ini penting untuk memungkinkan algoritma machine learning memproses label secara numerik. Setelah encoding, kolom kategori dihapus dari fitur X sehingga X hanya terdiri dari kolom numerik pm10, so2, co, o3, dan no2.

### 3. Split Data
Dataset kemudian dibagi menjadi data latih (training) dan data uji (testing) menggunakan fungsi train_test_split dengan rasio 80:20 (test_size=0.2) dan random_state=42 untuk menjaga konsistensi hasil. Hasil dari proses ini adalah X_train, X_test, y_train, dan y_test, yang kemudian digunakan dalam tahap pelatihan dan evaluasi model.

![image](https://github.com/user-attachments/assets/2947e234-3316-412d-811f-cb536fdee01d)

### 4. Resampling Data

Selain itu, untuk mengatasi ketidakseimbangan data pada label target, teknik resampling dilakukan menggunakan SMOTE dengan random_state=42, yang menghasilkan X_train_resampled dan y_train_resampled. Proses ini meningkatkan jumlah sampel dari kelas minoritas, dengan distribusi label y_train_resampled menunjukkan diratakan menjadi 2526, sehingga memastikan model dapat belajar dengan lebih baik dari data yang lebih seimbang, sambil tetap mempertahankan transparansi dan reproduksibilitas proses.

![image](https://github.com/user-attachments/assets/1e646d2f-dbf6-4b98-b35a-feb2ff2cc8b4)

## Modeling
Tahapan modeling merupakan inti dari proyek klasifikasi kualitas udara ini. Dalam proyek ini, digunakan dua algoritma machine learning yaitu Random Forest dan XGBoost. Keduanya dipilih karena dikenal memiliki performa tinggi dalam tugas klasifikasi, serta mampu menangani data dengan fitur numerik secara efisien. Random Forest merupakan algoritma ensemble yang membentuk beberapa pohon keputusan dan menggabungkan hasilnya untuk meningkatkan akurasi prediksi. Salah satu kelebihan utama Random Forest adalah kemampuannya dalam mengurangi overfitting yang sering terjadi pada pohon keputusan tunggal. Selain itu, Random Forest juga mampu menangani data dengan fitur yang saling berkorelasi. Namun, kekurangannya adalah waktu komputasi yang relatif lebih lama dibandingkan model sederhana serta kurang transparan dalam interpretasi hasil. Sementara itu, XGBoost (Extreme Gradient Boosting) adalah algoritma boosting yang menggabungkan pohon-pohon keputusan secara bertahap dengan menekankan pada kesalahan prediksi dari model sebelumnya. XGBoost dikenal sangat kuat dalam berbagai kompetisi data science karena kemampuannya menangani outlier, overfitting, serta efisiensi komputasi yang tinggi. Namun, model ini memerlukan proses tuning parameter yang lebih kompleks agar performanya optimal.

Model RF dikonfigurasi dengan parameter:
- n_estimators=600 (jumlah pohon untuk meningkatkan akurasi dan stabilitas)
- max_depth=15 (membatasi kedalaman pohon untuk mencegah overfitting)
- criterion='gini' (menggunakan indeks Gini untuk pemisahan)
- max_features='sqrt' (memilih akar kuadrat dari jumlah fitur untuk efisiensi)
- random_state=42 (menjaga reproduksibilitas)
- n_jobs=-1 (menggunakan semua core CPU untuk mempercepat pelatihan)

Sementara itu, model XGBoost dikonfigurasi dengan parameter: 
- n_estimators=600 (jumlah iterasi boosting)
- max_depth=15 (kedalaman pohon maksimum)
- learning_rate=0.1 (mengontrol kontribusi setiap pohon untuk mencegah overfitting)
- subsample=0.8 (menggunakan 80% data per iterasi untuk efisiensi)
- colsample_bytree=0.8 (menggunakan 80% fitur per pohon untuk diversifikasi)
- use_label_encoder=False (mencegah encoding otomatis label)
- eval_metric='mlogloss' (metrik evaluasi untuk klasifikasi multikelas)
- random_state=42 (reproduksibilitas)

Kedua model dilatih menggunakan X_train_resampled dan y_train_resampled yang telah diseimbangkan dengan SMOTE.

Hasil kedua model disajikan dalam tabel berikut:

![image](https://github.com/user-attachments/assets/97349cd7-9755-499a-b92d-541ca7a1238e)

## Evaluation
Dalam proyek ini, digunakan empat metrik evaluasi utama untuk menilai performa model klasifikasi kualitas udara, yaitu akurasi, precision, recall, dan F1-score. 

- Akurasi merupakan rasio prediksi benar (positif dan negatif) dengan keseluruhan data.
- Precision merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan hasil yang diprediksi positf. 
- Recall merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan data yang benar positif.
- F1-Score merupakan perbandingan rata-rata precision dan recall yang dibobotkan.

Metrik-metrik ini dipilih karena sesuai dengan konteks permasalahan klasifikasi multi-kelas. Akurasi digunakan untuk melihat secara umum seberapa banyak prediksi model yang benar dibandingkan dengan jumlah data keseluruhan. Namun, karena dalam data terdapat kemungkinan ketidakseimbangan antar kelas, maka precision dan recall digunakan untuk memberikan penilaian yang lebih rinci per kelas. Precision mengukur seberapa tepat model dalam memberikan label suatu kategori, sementara recall menunjukkan seberapa baik model dalam menangkap semua data yang memang termasuk dalam kategori tersebut. F1-score kemudian digunakan sebagai keseimbangan antara precision dan recall, khususnya berguna saat kita ingin menghindari ketimpangan antara keduanya.

Hasil dari evaluasi kedua model adalah sebagai berikut:

### 1. Evaluasi Model Random Forest
- Akurasi: 94.04%
- Macro Avg F1-score: 0.94
- Recall: 0.94
- Weighted Avg Precision: 0.94
Model Random Forest paling baik mengklasifikasikan kelas “SEDANG” dan “TIDAK SEHAT”, namun masih terdapat beberapa kesalahan klasifikasi pada kelas “BAIK” dan “SEDANG”.

### 2. Evaluasi Model XGBoost
- Akurasi: 92.77%
- Macro Avg F1-score: 0.92
- Recall: 0.93
- Weighted Avg Precision: 0.93
Meskipun XGBoost memiliki performa yang baik, terlihat bahwa nilai akurasi dan macro average F1-score sedikit lebih rendah dibanding Random Forest.

### 3. Komparasi

![image](https://github.com/user-attachments/assets/cfce92d4-0b1a-4e0d-86cf-33b53b95dc6f)

Berdasarkan hasil evaluasi, Random Forest dipilih sebagai model terbaik karena memiliki nilai precision, recall, dan F1-score (weighted average) sebesar 0.94, sedikit lebih tinggi dibandingkan XGBoost yang mencapai 0.93 pada semua metrik. Pilihan ini sejalan dengan kebutuhan untuk memaksimalkan akurasi klasifikasi pada dataset yang telah diseimbangkan.

### 4. Analisis Feature Importance

![image](https://github.com/user-attachments/assets/b0613af6-d770-4f96-9b61-3e18e2b924e0)

Berdasarkan grafik perbandingan feature importance antara model Random Forest dan XGBoost, dapat disimpulkan bahwa O3 (ozon) merupakan fitur yang paling berpengaruh terhadap klasifikasi kualitas udara, dengan skor importance tertinggi di kedua model. Disusul oleh PM10 yang juga menunjukkan kontribusi signifikan meskipun lebih rendah dibanding O3. Polutan lainnya seperti SO2, CO, dan NO2 memiliki pengaruh yang jauh lebih kecil, yang terlihat dari skor importance yang rendah di kedua model. Konsistensi hasil antara Random Forest dan XGBoost menunjukkan bahwa model memiliki persepsi yang serupa terhadap variabel-variabel penting dalam menentukan kategori kualitas udara.

### Keterkaitan Evaluasi Model dengan Business Understanding

#### Problem Statement
Model Random Forest yang telah dievaluasi terbukti berhasil mengatasi seluruh pernyataan masalah yang telah dirumuskan:
- Model memberikan prediksi kualitas udara secara cepat dan akurat berdasarkan data polutan sensor. Evaluasi menunjukkan akurasi sebesar 94.04%, sehingga kebutuhan lembaga pemerintahan/lingkungan akan sistem otomatis telah terpenuhi.
- Model mengubah nilai numerik polutan menjadi label kualitas udara (Baik, Sedang, Tidak Sehat, dll.) yang mudah dipahami oleh masyarakat umum maupun pengambil kebijakan.
- Melalui analisis feature importance dari Random Forest dan XGBoost, model dapat mengungkapkan faktor polutan utama seperti O3 dan PM10, menggantikan analisis manual yang kompleks dan rawan kesalahan.
- Penerapan SMOTE berhasil memperbaiki ketidakseimbangan data. Model mampu memprediksi kategori minoritas seperti "Sangat Tidak Sehat" dengan F1-score tinggi (0.91), mengatasi bias yang umum pada model yang dilatih dengan data tidak seimbang.

#### Goals
Semua tujuan yang ditetapkan pada tahap *Business Understanding* telah tercapai melalui proses modeling dan evaluasi:
- Model machine learning (Random Forest) telah dikembangkan dan mampu mengklasifikasikan kualitas udara dengan akurasi tinggi.
- Data numerik berhasil dikonversi menjadi label kategori kualitas udara yang lebih komunikatif dan informatif.
- Feature importance berhasil dianalisis, memberikan informasi polutan dominan terhadap penurunan kualitas udara.
- Ketidakseimbangan kelas berhasil diatasi menggunakan teknik SMOTE, yang meningkatkan performa model di kelas minoritas.


#### Solution Statement
Setiap solusi yang direncanakan pada awal proyek terbukti efektif dan berdampak langsung terhadap kualitas model:
- Model Random Forest dan XGBoost telah diterapkan. Random Forest dipilih sebagai model terbaik berdasarkan evaluasi akurasi, F1-score, dan recall, yang menunjukkan hasil yang sangat memuaskan di seluruh kelas.
- Analisis feature importance dari kedua model telah dilakukan. Hasilnya menunjukkan bahwa O₃ dan PM10 adalah fitur dengan pengaruh terbesar, memberikan insight bagi pengendalian polusi.
- Penerapan teknik SMOTE berhasil memperbaiki distribusi kelas dan meningkatkan kemampuan model dalam mengklasifikasikan kategori yang jarang muncul.
- Evaluasi yang menyeluruh dilakukan menggunakan metrik yang sesuai (accuracy, F1-score, recall, confusion matrix), memastikan kinerja model tidak bias dan dapat diandalkan.

