# Big Data Analytics — Recommendation System Dashboard

Dashboard interaktif **1 aplikasi / 8 halaman** untuk tujuh aktivitas praktik Recommendation System Week 7, berdasarkan notebook dan data yang dibagikan dosen (bukan data simulasi baru dari Kaggle).

## Halaman dan sumber

| Halaman | Metode | Dasar praktik / data |
|---|---|---|
| 1. Apriori | Frequent itemsets dan association rules | `Apriori (1).ipynb`, lima transaksi contoh |
| 2. Market Basket Analysis | Apriori, support, confidence, lift | `Association Rule_ Groceries.R`, `data_groceries (1) (2).csv` |
| 3. Content-Based (contoh) | TF-IDF dan cosine similarity | `Content_Based_Recommendation (1).ipynb`, lima film contoh |
| 4. Movie Recommendation | TF-IDF pada genre MovieLens | `Content_Based_Filtering_ML (1).ipynb`, `Movie (1).zip / movie.csv` |
| 5. User-Based Collaborative | Cosine similarity pengguna | `Collaborative_filter (1).ipynb`, sembilan rating contoh |
| 6. Item-Based Collaborative | kNN dengan jarak cosine | `Item_Based_Collaborative_Filtering_Copy1.ipynb`, arsip `BX-Users (1) (2).zip` |
| 7. Hybrid Recommendation | Gabungan Content-Based dan User-Based | `hybrid (1).ipynb`, rating + genre contoh |

Halaman kedelapan adalah **Overview**, berisi peta metode dan sumber data.

## Struktur folder

```text
recommendation_dashboard/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── data/
    ├── groceries.csv
    ├── movie.csv
    ├── rating.csv
    └── book_knn_interactions.csv.gz
```

- File `movie.csv`, `rating.csv`, dan `groceries.csv` merupakan salinan dari berkas yang disediakan dosen.
- File `book_knn_interactions.csv.gz` adalah turunan terkompresi dari **tiga CSV asli** dalam arsip Book-Crossing. Ini **bukan** data buatan. Berkas ini memungkinkan deployment ringan dan menghindari membaca 1,1 juta rating pada setiap pemuatan aplikasi.
- **music.csv** dan **Online Retail.xlsx** tidak dibawa ke aplikasi tujuh aktivitas karena tidak dipanggil oleh script asli pada paket praktik yang direview.
- Script R dosen menjalankan association rules atas objek `Groceries` bawaan paket `arules` (bukan atas `data_groceries.csv`); dashboard ini secara eksplisit menghitung aturan langsung dari file CSV yang tersedia. Perbedaan data dan parameter dapat memengaruhi jumlah hasil.

### Filter data Book-Crossing yang sudah dilakukan

1. Membaca file `BX-Book-Ratings.csv`, `BX-Books.csv`, dan `BX-Users.csv` dari ZIP dosen.
2. Mempertahankan pengguna yang awalnya memiliki >=200 rating.
3. Mempertahankan nilai rating yang frekuensinya >=100 (seperti notebook asli).
4. Menggabungkan rating dengan judul dan memilih judul yang memiliki >=50 rating.
5. Memfilter lokasi yang memuat `usa` atau `canada`.
6. Menghapus duplikat pasangan pengguna–judul buku.

Hasil pada lingkungan pemrosesan: **54.731 interaksi**, **746 judul buku**, dan **734 pengguna**. Saat membaca CSV buku terdapat beberapa baris tidak valid yang dilewati dengan `on_bad_lines="skip"`, sehingga angka mungkin berbeda dari pembacaan di lingkungan lain. Data ringkas mempertahankan `userID`, `bookTitle`, `bookRating`, `bookAuthor`, dan `yearOfPublication`.

## Cara jalankan di komputer

Gunakan **Python 3.12** (sesuai troubleshooting deployment sebelumnya):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Upload ke GitHub

1. Buat repository **baru** di GitHub. Contoh: `recommendation-system-bda`.
2. Upload **isi folder** ini ke root repository (bukan mengunggah ZIP tanpa diekstrak). Pastikan `app.py`, `requirements.txt`, folder `data`, dan folder `.streamlit` berada sesuai struktur di atas.
3. Commit semua file.
4. Di Streamlit Community Cloud pilih **Create app → From GitHub**.
5. Tentukan repository dan branch `main`, **Main file path = `app.py`**.
6. Di **Advanced settings**, gunakan **Python 3.12**, lalu deploy.

**Penting:** jangan mengganti aplikasi contoh `app_belajar.py` dosen; gunakan repository terpisah untuk dashboard tugas ini.

## Interpretasi dan batasan

- Apriori menemukan pola keterkaitan barang, **bukan** hubungan sebab-akibat.
- Content-based hanya menggunakan deskripsi/genre; hasil teratas dapat memiliki skor sama jika genre identik.
- User-based pada data contoh kecil mungkin tidak menghasilkan rekomendasi untuk pengguna tertentu; itu normal, bukan kesalahan dashboard.
- Item-based menggunakan nilai 0 dari Book-Crossing sesuai praktik; rating 0 di data ini adalah sinyal **implisit**, bukan pernyataan pengguna memberikan skor ketidaksukaan.
- Hybrid menggunakan MinMaxScaler pada *candidate items* dan bobot pengguna. Skornya bersifat relatif terhadap kandidat pada run tersebut.
- Dashboard ini alat eksplorasi materi kuliah; tidak melakukan validasi rekomendasi skala produksi atau uji kualitas model pada holdout data.
