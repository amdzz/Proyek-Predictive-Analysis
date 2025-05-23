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
- category: Kategori kualitas udara berdasarkan nilai "max", yang menggambarkan tingkat kebersihan atau pencemaran udara.

## Data Preparation
Tahap data preparation dilakukan untuk memastikan bahwa data yang digunakan dalam pelatihan model sudah bersih, valid, dan relevan. Proses ini dimulai dengan pemilihan fitur-fitur yang dianggap paling berpengaruh terhadap kualitas udara, yaitu pm10, pm25, so2, co, o3, no2, serta kolom categori sebagai label klasifikasi. Pemilihan ini bertujuan untuk menyederhanakan data dan hanya mempertahankan informasi yang diperlukan dalam proses pemodelan.

Selanjutnya, dilakukan pembersihan data dengan menghapus baris-baris yang memiliki kategori TIDAK ADA DATA. Hal ini penting karena nilai tersebut tidak merepresentasikan kualitas udara yang sebenarnya, sehingga dapat mengganggu proses pelatihan model dan menghasilkan prediksi yang tidak akurat. Setelah itu, dilakukan pengecekan terhadap nilai-nilai kosong (missing values) pada setiap kolom menggunakan fungsi isnull().sum(). Tujuannya adalah untuk memastikan bahwa tidak ada fitur yang memiliki nilai kosong yang dapat mempengaruhi proses pelatihan model.

Kemudian, kolom pm25 dihapus dari dataset. Keputusan ini dapat diambil karena adanya pertimbangan korelasi tinggi dengan pm10, atau karena model awal menunjukkan performa yang baik meskipun tanpa fitur tersebut. Untuk menyederhanakan model dan menghindari redundansi, penghapusan ini dianggap wajar. Setelah itu, dilakukan verifikasi ulang kolom-kolom yang tersisa untuk memastikan struktur data sudah sesuai.

Terakhir, data dengan label kategori BERBAHAYA juga dihapus karena jumlah datanya sangat sedikit (hanya satu baris), sehingga tidak cukup untuk dilakukan proses resampling seperti SMOTE. Keberadaan label yang sangat jarang justru bisa menyebabkan error atau bias pada model. Setelah penghapusan tersebut, indeks data diatur ulang agar tetap rapi dan berurutan. Seluruh proses ini dilakukan secara berurutan untuk memastikan bahwa data yang masuk ke model sudah dalam kondisi optimal untuk pelatihan dan evaluasi.

## Modeling
Tahapan modeling merupakan inti dari proyek klasifikasi kualitas udara ini. Dalam proyek ini, digunakan dua algoritma machine learning yaitu Random Forest dan XGBoost. Keduanya dipilih karena dikenal memiliki performa tinggi dalam tugas klasifikasi, serta mampu menangani data dengan fitur numerik secara efisien.

Random Forest merupakan algoritma ensemble yang membentuk beberapa pohon keputusan dan menggabungkan hasilnya untuk meningkatkan akurasi prediksi. Salah satu kelebihan utama Random Forest adalah kemampuannya dalam mengurangi overfitting yang sering terjadi pada pohon keputusan tunggal. Selain itu, Random Forest juga mampu menangani data dengan fitur yang saling berkorelasi. Namun, kekurangannya adalah waktu komputasi yang relatif lebih lama dibandingkan model sederhana serta kurang transparan dalam interpretasi hasil.

Sementara itu, XGBoost (Extreme Gradient Boosting) adalah algoritma boosting yang menggabungkan pohon-pohon keputusan secara bertahap dengan menekankan pada kesalahan prediksi dari model sebelumnya. XGBoost dikenal sangat kuat dalam berbagai kompetisi data science karena kemampuannya menangani outlier, overfitting, serta efisiensi komputasi yang tinggi. Namun, model ini memerlukan proses tuning parameter yang lebih kompleks agar performanya optimal.

Berdasarkan hasil evaluasi dari dua algoritma yang digunakan, yaitu Random Forest dan XGBoost, dapat dilakukan analisis perbandingan performa model. Model Random Forest menunjukkan akurasi sebesar 93.94%, dengan nilai precision, recall, dan f1-score yang merata di angka 0.93 hingga 1.00 untuk masing-masing kelas. Confusion matrix menunjukkan bahwa sebagian besar prediksi sesuai dengan label aktual, khususnya untuk kelas SEDANG dan TIDAK SEHAT, walaupun masih terdapat sedikit kesalahan prediksi pada kelas BAIK yang diklasifikasikan sebagai SEDANG.

Sementara itu, model XGBoost memiliki akurasi yang sedikit lebih tinggi, yaitu 94.13%, dengan performa metrik evaluasi yang juga kuat. Nilai precision, recall, dan f1-score konsisten tinggi untuk seluruh kelas, terutama pada kelas SEDANG dan TIDAK SEHAT. Berdasarkan confusion matrix, XGBoost berhasil memprediksi lebih banyak data dengan benar dibanding Random Forest, khususnya dalam mengurangi kesalahan pada kelas SEDANG dan TIDAK SEHAT.

Dengan mempertimbangkan seluruh metrik (accuracy, precision, recall, dan f1-score) serta confusion matrix, XGBoost dipilih sebagai model terbaik untuk menyelesaikan permasalahan klasifikasi kualitas udara ini. Keunggulan XGBoost terletak pada kemampuannya dalam menangani data tidak seimbang serta fleksibilitasnya dalam melakukan boosting untuk meningkatkan performa klasifikasi. Selain itu, XGBoost juga menunjukkan stabilitas prediksi yang lebih baik dibanding Random Forest dalam eksperimen ini.

## Evaluation
Dalam proyek ini, digunakan empat metrik evaluasi utama untuk menilai performa model klasifikasi kualitas udara, yaitu akurasi, precision, recall, dan F1-score. 

- Akurasi merupakan rasio prediksi benar (positif dan negatif) dengan keseluruhan data.
- Precision merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan hasil yang diprediksi positf. 
- Recall merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan data yang benar positif.
- F1-Score merupakan perbandingan rata-rata precision dan recall yang dibobotkan.

Metrik-metrik ini dipilih karena sesuai dengan konteks permasalahan klasifikasi multi-kelas. Akurasi digunakan untuk melihat secara umum seberapa banyak prediksi model yang benar dibandingkan dengan jumlah data keseluruhan. Namun, karena dalam data terdapat kemungkinan ketidakseimbangan antar kelas, maka precision dan recall digunakan untuk memberikan penilaian yang lebih rinci per kelas. Precision mengukur seberapa tepat model dalam memberikan label suatu kategori, sementara recall menunjukkan seberapa baik model dalam menangkap semua data yang memang termasuk dalam kategori tersebut. F1-score kemudian digunakan sebagai keseimbangan antara precision dan recall, khususnya berguna saat kita ingin menghindari ketimpangan antara keduanya.

Berdasarkan hasil evaluasi, model Random Forest memberikan akurasi sebesar 93.95% dengan nilai F1-score, precision, dan recall yang juga tinggi dan konsisten di sekitar 0.94. Model XGBoost memiliki performa serupa, dengan akurasi sebesar 94.13% dan metrik lainnya yang juga kuat. Perbedaan performa kedua model sangat tipis. Namun, dari hasil confusion matrix terlihat bahwa XGBoost sedikit lebih baik dalam mengenali kelas SEDANG dan TIDAK SEHAT, dengan jumlah prediksi yang salah lebih sedikit. Oleh karena itu, meskipun keduanya sama-sama akurat, model XGBoost dipilih sebagai model terbaik karena memberikan hasil prediksi yang sedikit lebih stabil dan lebih baik dalam mengklasifikasikan kategori yang paling dominan. Dengan menggunakan metrik-metrik ini, dapat disimpulkan bahwa model mampu mengklasifikasikan kualitas udara dengan sangat baik dan dapat diandalkan untuk digunakan dalam konteks nyata.
