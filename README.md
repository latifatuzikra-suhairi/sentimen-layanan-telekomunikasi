# ANALISIS SENTIMEN TINGKAT KEPUASAN PENGGUNA PENYEDIA LAYANAN TELEKOMUNIKASI SELULER INDONESIA PADA PLATFORM TWTITTER
Banyaknya perusahaan penyedia layanan telekomunikasi selular di Indonesia menjadikan tiap provider harus memahami kelebihan dan kekurangan layanan perusahaannya sendiri dan juga perusahaan pesaing. Kelebihan dan kekurangan tersebut dapat dilihat salah satunya dari cuitan media sosial, seperti Twitter. Dengan demikian, diperlukan analisis sentimen pada Twitter pengguna menyangkut penyedia layanan telekomunikasi selular tersebut. Dalam proyek ini, analisis sentimen dikelompokkan dalam analisis sentimen positif dan negatif dengan menggunakan perbandingan dua model algoritma, yaitu Naive Bayes dan Support Vector Machine.

## DAFTAR ISI
- [DATASET](#dataset)
- [DATA PREPROCESSING](#data-processing)
- [FEATURE ENGINEERING](#feature-engineering)
- [MODELLING](#modelling)
- [MODEL EVALUATION](#model-evaluation)
- [CONCLUSION](#conclusion)
  
## DATASET
Dataset bersumber dari [Github](https://github.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indoensia) yang berisikan 300 data tentang cuitan pengguna terhadap penyedia layanan telekomunikasi di Twitter. Data terdiri dari 3 kolom:
1. `Id`, id komentar
2. `Sentiment`, sentimen dari komentar berupa nilai positive atau negative
3. `Text Tweet`, teks komentar

## DATA PREPROCESSING
4 tahapan memproses data yang dilakukan:
1. Casefolding
   Melibatkan pengubahan teks menjadi lower case, menghapus angka menggunakan regex yang sudah ditetapkan, menghapus karakter tanda baca menggunakan regex yang sudah ditetapkan, dan menghapus whitespace.
   ```python
   import re
   def casefolding(text):
    text = text.lower()                               # Mengubah teks menjadi lower case
    text = re.sub(r'[-+]?[0-9]+', '', text)           # Menghapus angka
    text = re.sub(r'[^\w\s]','', text)                # Menghapus karakter tanda baca
    text = text.strip()                               # Menghapus whitespace
    return text
   ```
2. Text Normalize
    Dilakukan untuk mengubah kata slang menjadi bentuk baku sesuai dengan kaidah bahasa Indonesia. Menggunakan bantuan dari kamus yang telah dibuat oleh [KSNugroho](https://raw.githubusercontent.com/ksnugroho/klasifikasi-spam-sms/master/data/key_norm/csv)
    ```python
    def text_normalize(text):
      text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0] if (key_norm['singkat'] == word).any() else word for word in text.split()])
      text = str.lower(text)
      return text
    ``` 
3. Filtering
   Pemilihan kata-kata penting atau kata-kata apa saja yang di gunakan untuk mewakili dokumen. Dalam prosesnya, digunakan penghapusan stopwords berdasarkan corpus Indonesia milik library nltk.
   ```python
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    
    stopwords_ind = stopwords.words('indonesian')
    more_stopword = ['url', 'provider_name', 'user_mention', 'product_name', 'boikot_provider_name', 'boikotprovider_name']                    # Tambahkan kata lain dalam daftar stopword

    stopwords_ind = stopwords_ind + more_stopword
    
    # Buat fungsi untuk langkah filtering stopword
    def remove_stopwords(text):
      clean_words = []
      text = text.split()
      for word in text:
          if word not in stopwords_ind:
              clean_words.append(word)
      return ' '.join(clean_words)
   ```
4. Stemming
   Proses pemetaan dan penguraian bentuk dari suatu kata menjadi bentuk kata dasarnya. Menggunakan library Sastrawi
   ```python
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

   # Buat fungsi untuk langkah stemming bahasa Indonesia
    def stemming(text):
      text = stemmer.stem(text)
      return text
   ```
5. Label encoding
   Nilai `0` mewakili data berlabel `negative` dan nilai `1` mewakili data berlabel `positive`.
   ```python
   data['Sentiment'] = data['Sentiment'].replace({'positive':'1', 'negative': '0'}).astype(int)
   ```

## FEATURE ENGINEERING
1. Feature Extraction
   Mengubah data text menjadi vektor agar mudah dipahami oleh komputer dengan `TF-IDF`. Hasilnya, didapatkan 2959 fitur dalam corpus
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   tf_idf = TfidfVectorizer(ngram_range=(1,3))
   tf_idf.fit(X)
   ```
2. Feature selection
   Mengambil beberapa fitur terbaik untuk pemodelan machine learning. Diambil 1200 fitur terbaik dari 2595 fitur yang ada berdasarkan `chi-square`.
   ```python
    from sklearn.feature_selection import SelectKBest 
    from sklearn.feature_selection import chi2
   
    chi2_features = SelectKBest(chi2, k=1200) 
    X_kbest_features = chi2_features.fit_transform(X, y) 
    ```
## MODELLING
1. Naive Bayes
   Pemodelan dilakukan dengan Gaussian Naive Bayes menggunakan hyperparameter tuning pada parameter `var_smoothing`. Hasilnya, diperoleh parameter terbaik dengan nilai `var_smoothing: 0.0533669923120631`
  
3. Support Vector Machine
   Pemodelan dilakukan dengan SVM menggunakan hyperparameter tuning pada parameter `C`, `gamma`, dan `kernel`, dan `max_iter`. Hasilnya, diperoleh parameter terbaik dengan nilai `C:1.0, gamma:0.1, kernel:linear, max_iter:50`

## MODEL EVALUATION
1. Naive Bayes
   Memiliki confusion matrix sebagai berikut:
   ![Confusion Matrix Naive Bayes](https://raw.githubusercontent.com/latifatuzikra-suhairi/sentimen-layanan-telekomunikasi/main/static/CF_GNB_Sentimen_Telekomunikasi.png)

   Hasilnya, model ini memiliki **akurasi 98%** dengan jumlah prediksi yang benar 88 data dan prediksi yang salah 2 data.

2. Support Vector Machine
   Memiliki confusion matrix sebagai berikut:
   ![Confusion Matrix SVM](https://raw.githubusercontent.com/latifatuzikra-suhairi/sentimen-layanan-telekomunikasi/main/static/CF_SVM_Sentimen_Telekomunikasi.png)

   Hasilnya, model ini memiliki **akurasi 78.8%** dengan jumlah prediksi yang benar 71 data dan prediksi yang salah 19 data.

## CONCLUSION
Model analisis sentiment Tingkat Kepuasan Pengguna Penyedia Layanan Telekomunikasi Seluler Indonesia Pada Platform Twitter yang terbaik adalah **model Gaussian Naive Bayes** dengan nilai **akurasi model 98%** dibandingkan dengan **model Support Vector Machine Classifier** yang hanya memiliki **nilai akurasi 79%**
