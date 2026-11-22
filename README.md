# Klasifikasi Biner dengan SVM (scratch) — Dataset Electricity (Elec2)

Implementasi **Linear Soft-Margin Support Vector Machine** dari nol (hanya NumPy) untuk memprediksi apakah harga listrik **naik (`UP`)** atau **turun (`DOWN`)** pada pasar listrik New South Wales (NSW), Australia.

Tujuan:

1. **Memahami model matematis SVM** (margin, hinge loss, regularisasi, gradient descent) sekaligus penerapannya dalam kode.
2. **Meningkatkan kualitas performa model secara bertahap**, dengan evaluasi yang memperhatikan bahwa data ini adalah deret waktu (perbandingan *random split* vs *chronological split*).

## DATASET

**Electricity (Elec2)** dikumpulkan dari pasar listrik NSW, Australia, dan umum dipakai untuk mempelajari perubahan harga listrik (klasifikasi biner).

- **45.312** observasi, **8 fitur** + 1 target
- Tidak ada nilai kosong dan tidak ada baris duplikat
- Distribusi kelas: **57.55% `DOWN`** (26.075) dan **42.45% `UP`** (19.237)

| Kolom | Deskripsi |
| --- | --- |
| `date` | Tanggal pengamatan, dinormalisasi ke 0–1 (7 Mei 1996 – 5 Des 1998 menurut dokumentasi dataset) |
| `day` | Hari dalam seminggu (1–7) |
| `period` | Slot pengamatan 30 menit dalam sehari (48 nilai unik), dinormalisasi ke 0–1 |
| `nswprice` | Harga listrik NSW |
| `nswdemand` | Permintaan listrik NSW |
| `vicprice` | Harga listrik Victoria |
| `vicdemand` | Permintaan listrik Victoria |
| `transfer` | Transfer listrik antarwilayah NSW–Victoria |
| `class` | Target: harga naik (`UP`) atau turun (`DOWN`) |

**Sumber**: 

- [Max Halford](https://maxhalford.github.io/files/datasets/electricity.zip) (dipakai notebook, diunduh otomatis saat dijalankan)
- [Kaggle](https://www.kaggle.com/datasets/yashsharan/the-elec2-dataset)

## MODEL MATEMATIS

Label kelas: $y_i \in \{-1, +1\}$ (`DOWN` = −1, `UP` = +1).

**Fungsi keputusan dan hyperplane**

$$
f(x) = w^T x + b
$$
$$
\hat{y} = sign(f(x))
$$

Batas pemisah (*decision boundary*) adalah himpunan titik dengan $w^T x + b = 0$. Pada implementasi, titik dengan $f(x) = 0$ dipetakan ke kelas $+1$.

**Margin.** Misalkan $x_1$ dan $x_2$ berada pada batas margin masing-masing kelas, sehingga $w^T x_1 + b = +1$ dan $w^T x_2 + b = -1$. Jika dikurangkan, diperoleh $w^T(x_1 - x_2) = 2$. Ruas kiri persamaan berikut adalah proyeksi $(x_1 - x_2)$ pada arah normal hyperplane, yaitu lebar margin:

$$
\frac{w^T}{\lVert w \rVert}(x_1 - x_2) = \frac{2}{\lVert w \rVert}
\quad\Rightarrow\quad
M = \frac{2}{\lVert w \rVert}
$$

**Hard-margin.** Memaksimalkan margin ekuivalen dengan meminimalkan $\lVert w \rVert^2$:

$$
\min_{w,b}\ \tfrac{1}{2}\lVert w \rVert^2
\quad \text{dengan} \quad
y_i(w^T x_i + b) \ge 1
$$

**Soft-margin.** Diperkenalkan *slack variable* $\xi_i \ge 0$ agar sebagian data boleh melanggar margin:

$$
\min_{w,b,\xi}\ \tfrac{1}{2}\lVert w \rVert^2 + C\sum_{i=1}^{n}\xi_i
\quad \text{dengan} \quad
y_i(w^T x_i + b) \ge 1 - \xi_i
$$

**Hinge loss.** Dari kendala di atas, nilai terkecil $\xi_i$ adalah

$$
\xi_i = L_i = \max\big(0,\ 1 - y_i f(x_i)\big)
$$

Loss bernilai 0 bila data berada di luar atau tepat pada margin, dan naik secara linear bila data masuk ke dalam margin atau salah diklasifikasikan.

**Fungsi objektif** yang diminimalkan oleh kode:

$$
J(w,b) = \frac{\lambda}{2}\lVert w \rVert^2 + \frac{1}{n}\sum_{i=1}^{n}\max\big(0,\ 1 - y_i(w^T x_i + b)\big)
$$

Hubungannya dengan parameter $C$ pada formulasi soft-margin adalah $C = \dfrac{1}{\lambda n}$.

**Gradien.** Hinge loss tidak terdiferensialkan di $y_i f(x_i) = 1$, sehingga yang dipakai adalah *subgradien*:

- Jika $y_i f(x_i) \ge 1$: $\ \nabla_w J_i = \lambda w,\ \ \nabla_b J_i = 0$
- Jika $y_i f(x_i) < 1$: $\ \nabla_w J_i = \lambda w - y_i x_i,\ \ \nabla_b J_i = -y_i$

Gradien total adalah rata-rata seluruh sampel. Pembaruan parameter dengan *learning rate* $\eta$:

$$
w \leftarrow w - \eta\,\nabla_w J, \qquad b \leftarrow b - \eta\,\nabla_b J
$$

## IMPLEMENTASI

Model ada di [`svm_model2.py`](svm_model2.py) (kelas `SVM`, hanya bergantung pada NumPy).

| Komponen | Keterangan |
| --- | --- |
| Optimasi | *Batch gradient descent* (seluruh data per iterasi) |
| Inisialisasi | $w = 0,\ b = 0$ |
| Label | Wajib `-1` dan `+1` |
| Prediksi | `+1` jika $f(x) \ge 0$, `-1` jika $f(x) < 0$ |
| Riwayat train | `objective_history`, `hinge_loss_history`, `accuracy_history` (dicatat tiap iterasi) |

Parameter konstruktor:

| Parameter | Default | Arti |
| --- | :---: | --- |
| `iters` | `1000` | Jumlah iterasi gradient descent |
| `eta` | `0.01` | Learning rate $\eta$ |
| `lambdaP` | `0.01` | Parameter regularisasi $\lambda$ |

Metode: `fit(X, Y)`, `decision_function(X)`, `predict(X)`, `mean_hinge_loss(X, Y)`, `objective_function(X, 

## ALUR EKSPERIMEN

Seluruh alur ada di [`svm_binary_class2.ipynb`](svm_binary_class2.ipynb):

1. **Konsep**: klasifikasi biner dan model matematis SVM.
2. **Load dataset** dari URL dan **EDA** (info dataset, statistik, target, missing/duplikat, univariat, bivariat, korelasi, analisis temporal).
3. **Pre-processing**
   - Fitur: `nswprice`, `nswdemand`, `vicprice`, `vicdemand`, `transfer`. Fitur waktu (`date`, `day`, `period`) tidak dipakai pada versi ini.
   - Target: `DOWN → -1`, `UP → +1`.
4. **Train-test split (80/20)** dengan dua skema pembanding:
   - *Random split* (`shuffle=True`, `stratify=y`, `random_state=42`)
   - *Chronological split* (36.249 baris pertama untuk latih, 9.063 baris terakhir untuk uji, tanpa pengacakan)
5. **Standardisasi** (`StandardScaler`) yang di-*fit* hanya pada data latih masing-masing skema.
6. **Training** SVM untuk kedua skema, lalu **evaluasi** (accuracy, precision, recall, F1, mean hinge loss, confusion matrix).
7. **Penyimpanan** laporan, grafik, model, dan scaler ke folder `Results/`.

## LIMITASI DAN RENCANA PENGEMBANGAN

**Keterbatasan saat ini**

- Hyperparameter (`iters`, `eta`, `lambdaP`) belum disetel. Data validasi belum dipakai.
- Fitur waktu (`period`, `day`) belum dimanfaatkan padahal `period` memperlihatkan pola harian yang kuat.
- Hanya model linear tanpa kernel, dan belum dibandingkan dengan implementasi acuan (misalnya `scikit-learn`).
- Evaluasi hanya satu split dan satu kali eksekusi.

**Rencana**

- [ ] Setel `lambdaP` dan `eta` dengan data validasi kronologis (blok setelah data latih), dan uji data uji sekali di akhir.
- [ ] Uji performa berdasarkan pengaruh *time-series*.
- [ ] Bandingkan dengan `LinearSVC`/`SGDClassifier(loss="hinge")` sebagai *sanity check* implementasi.
- [ ] Tambahkan `class_weight`, metrik ROC-AUC, serta evaluasi per periode waktu.
- [ ] Variasi optimasi: *mini-batch SGD* dengan learning rate meluruh dan kriteria berhenti otomatis.
- [ ] Pendalaman matematis: bentuk dual, Lagrangian, KKT, dan kernel (misalnya RBF).

## STRUKTUR REPO

```
.
├── Lampiran/                    # gambar dokumentasi (ilustrasi SVM dan hasil EDA)
├── Results/
│   ├── figures/                 
│   ├── models/                         
│   └── reports/                 
├── svm_binary_class2.ipynb      # notebook utama (konsep, EDA, pre-processing, training, evaluasi)
├── svm_model2.py                # kelas SVM (NumPy)
├── svm_binary_class1.ipynb      # versi sebelumnya (arsip)
├── svm_model1.py                # versi sebelumnya (arsip)
├── requirements.txt
└── README.md
```

Awalan angka pada nama berkas di `Results/` adalah nomor versi eksperimen (`ver` pada notebook). Berkas `.pkl` dihasilkan saat notebook dijalankan.

## REFERENSI

- [Kaggle | SVM from Scratch](https://www.kaggle.com/code/fareselmenshawii/svm-from-scratch)
- [GitHub | MWalega/ml_algos_python/SVM.py](https://github.com/MWalega/ml_algos_python/blob/master/SVM.py)
- [GitHub | akash18tripathi/SVM-from-scratch-Constraints-analysis](https://github.com/akash18tripathi/SVM-from-scratch-Constraints-analysis/blob/main/SVM.ipynb)
- [YouTube | Siddhardhan: Support Vector Machine Classifier from Scratch](https://youtube.com/playlist?list=PLfFghEzKVmjvzS4DILijsdQk27Ew7xIPu&feature=shar)

---

**Author**: [Nna](https://github.com/andalive)