# Titanic Survival Analysis & ML Pipeline — Project Report

## 1) Proje Özeti (Executive Summary)
- Amaç: Titanic veri setinde yolcuların hayatta kalıp kalmadığını açıklamak ve tahmin etmek için uçtan uca, yeniden üretilebilir bir veri analizi ve makine öğrenmesi hattı (pipeline) kurmak.
- Yaklaşım: 
  - Keşifsel Veri Analizi (EDA) → 6 temel içgörü görselleştirmesi,
  - Özellik mühendisliği (Title, FamilySize, IsAlone, FarePerPerson, TicketGroupSize, CabinInitial),
  - `ColumnTransformer` + `Pipeline` ile tutarlı ön işleme,
  - Çoklu model denemeleri ve Stratified CV,
  - RandomForest hiperparametre ayarı (GridSearchCV),
  - Açıklanabilirlik (özellik önemleri, opsiyonel SHAP),
  - Kaggle’a gönderime hazır `submission.csv`.
- Sonuç: Public leaderboard skoru 0.75119; tek akışlı, tekrarlanabilir pipeline ile üretildi.

---

## 2) Veri Seti ve Kapsam
- Kaynak: Kaggle — “Titanic: Machine Learning from Disaster”.
- Dosyalar:
  - `train.csv` (etiketli, `Survived` hedefi),
  - `test.csv` (etiketsiz, Kaggle değerlendirmesi için),
  - `gender_submission.csv` (örnek/baseline).
- Önemli sütunlar: `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.

---

## 3) Keşifsel Veri Analizi (EDA) — Bulgular ve Görseller
Aşağıdaki 6 görsel, EDA’nın temel içgörülerini özetler. Görsel yerlerine dosya adlarını yerleştirildi; sunumda bu alanlara görselleri ekleyiniz.

1) Cinsiyete Göre Hayatta Kalma
- İçgörü: Kadın yolcuların hayatta kalma oranı anlamlı biçimde daha yüksek.
- Görsel:
![Açıklama](images/output1.png)

2) Yolcu Sınıfı (Pclass) ve Ücret (Fare)
- İçgörü: 1. sınıf yolcular daha yüksek ücret ve daha yüksek hayatta kalma eğilimine sahip.
- Görsel: output2

3) Yaş Dağılımı (KDE) x Hayatta Kalma
- İçgörü: Çocuklar/gençlerde görece daha yüksek hayatta kalma eğilimi görünüyor.
- Görsel: output3

4) Ünvan (Title) ve Hayatta Kalma
- İçgörü: Ünvan bilgisi güçlü ayırt edicilik taşıyor; nadir ünvanlar `Rare` altında toplanarak genellenebilirlik artırılıyor.
- Görsel: output4

5) Aile Büyüklüğü (FamilySize) ve IsAlone
- İçgörü: Tek başına seyahat (IsAlone=1) hayatta kalma ile negatif ilişkili; çok büyük ailelerde de risk artabiliyor.
- Görsel: output5

6) Korelasyon Isı Haritası (Sayısal Değişkenler)
- İçgörü: `Sex`, `Pclass`, `Fare` gibi değişkenler `Survived` ile anlamlı ilişkiler gösteriyor.
- Görsel: output6

---

## 4) Özellik Mühendisliği (Feature Engineering)
- FamilySize: `SibSp + Parch + 1`
- IsAlone: `1` (yalnız) / `0` (değil)
- Title: `Name` içinden ünvan çıkarımı; sık olmayanlar `Rare` olarak birleştirildi
- FarePerPerson: `Fare / FamilySize`
- TicketGroupSize: Aynı `Ticket`’ı paylaşan kişi sayısı
- CabinInitial: `Cabin` ilk karakteri; eksikler `U`
- Uygulama: Tüm adımlar `FunctionTransformer(add_features)` içinde; böylece eğitim ve tahmin aşamalarında aynı kalır (data leakage önlenir).

---

## 5) Ön İşleme ve Mimari
- Sayısal boru: `SimpleImputer(median)` → `StandardScaler`
- Kategorik boru: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')`
- Birleşim: `ColumnTransformer(num + cat)` → `Pipeline(feat → pre → clf)`
- Determinizm: `np.random.seed(42)` ve uygun modellerde `random_state=42`

---

## 6) Modelleme, Doğrulama ve Seçim
- Aday Modeller:
  - Logistic Regression (standart ve `class_weight='balanced'`),
  - RandomForest (standart ve `class_weight='balanced'`),
  - Opsiyonel booster’lar: XGBoost / LightGBM / CatBoost (kuruluysa).
- Çapraz Doğrulama:
  - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
  - Skorlar: `accuracy`, `f1`, `roc_auc`
- Seçim:
  - CV sonuçlarına göre en iyi aday (`best_pipe`),
  - RandomForest için `GridSearchCV` ile hiperparametre ayarı (`rf_best`),
  - Test değerlendirmesi sonrası `final_pipe = rf_best` veya `best_pipe`.

---

## 7) Hiperparametre Optimizasyonu (RF)
- Arama ızgarası örnekleri:
  - `n_estimators: [200, 400, 800]`
  - `max_depth: [None, 5, 10, 20]`
  - `min_samples_split: [2, 5, 10]`
- Kıstas: Çoğunlukla F1 ile arama yapıldı; Kaggle metriği accuracy olduğu için istenirse accuracy temelli seçimle güncellenebilir.

---

## 8) Değerlendirme (Test) ve Eğriler
- Test Metrikleri:
  - Sınıflandırma raporu (precision/recall/F1),
  - Confusion Matrix,
  - ROC eğrisi (AUC),
  - Precision–Recall eğrisi (AP).
- Öğrenme Eğrisi (Learning Curve):
  - Eğitim verisi arttıkça bias/variance davranışı; veri eklemenin kazancı.
- Doğrulama Eğrisi (Validation Curve, örn. RF `n_estimators`):
  - Hiperparametre–performans ilişkisi; `plt.xscale('log', base=2)` ile okunabilirlik.

---

## 9) Açıklanabilirlik (Explainability)
- Özellik Önemleri:
  - Ağaç tabanlı modellerde `feature_importances_` ile ilk 20 özellik grafiği.
- SHAP (opsiyonel):
  - Uygun ortamda `TreeExplainer` ile global etki dağılımı;
  - Kurulum yoksa güvenli biçimde atlanır (try/except).

---

## 10) Sonuçlar ve Etki
- Bulgular:
  - Cinsiyet, sınıf ve ücret gibi demografik/sosyal göstergeler güçlü sinyal içeriyor.
  - Ünvan ve aile temelli özellikler (Title, FamilySize, IsAlone) tahmin gücünü belirgin artırıyor.
- Kaggle Sonucu:
  - Public Leaderboard Accuracy: 0.75119
  - Not: Public dilim skorudur; private dilim farklı olabilir.
- Etki:
  - Uçtan uca, tek akıştan tekrar üretilebilir bir ML hattı oluşturuldu ve harici platformda ölçüldü.

---

## 11) Yeniden Üretilebilirlik ve Teslimatlar
- Tek akış: `titanic.ipynb` içindeki BÖLÜM 1–13 “Run All” ile veri → FE → ön işleme → CV → tuning → final → kayıt → submission.
- Model kaydı: `joblib.dump(final_pipe, "model.pkl")`; `load` sonrası test doğrulaması.
- Submission üretimi:
  - `BÖLÜM 13`: `BASE_FEATURES` kontrolü, güvenli model seçimi (final/rf_best/best_pipe/fallback), `submission.csv` yazımı.
  - Şekil: `(418, 2)`; kolonlar `PassengerId`, `Survived`; değerler 0/1.

---

## 12) Nasıl Çalıştırılır (Windows)
1) Kurulum
```powershell
cd "C:\Users\LENOVO\Desktop\Aktif Projeler\TITANIC_ANALYSYS"
pip install -r [requirements.txt](http://_vscodecontentref_/0)
