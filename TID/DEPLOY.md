# 🚀 TID Projesi — Canlıya Alma Rehberi

Bu rehber, TID İşaret Dili Algılama projesini **GitHub Pages + Hugging Face Spaces** ile ücretsiz olarak canlıya almanızı sağlar.

---

## 📋 Mimari

```
GitHub Pages (Frontend)          Hugging Face Spaces (Backend)
┌──────────────────────┐         ┌──────────────────────────┐
│  • HTML/CSS/JS       │  HTTP   │  • FastAPI               │
│  • Kamera            │ ──────► │  • Random Forest (pkl)   │
│  • MediaPipe         │ ◄────── │  • 1D CNN (h5)           │
│  • Sonuç gösterimi   │         │  • Ücretsiz Python       │
└──────────────────────┘         └──────────────────────────┘
kullanici.github.io/repo         kullanici-tid-api.hf.space
```

---

## Adım 1: Hugging Face Space Oluşturma (Backend)

### 1.1 Hesap Aç
1. [huggingface.co](https://huggingface.co) adresine git
2. **Sign Up** → Ücretsiz hesap oluştur

### 1.2 Yeni Space Oluştur
1. Sağ üstteki profil → **New Space**
2. Ayarlar:
   - **Space name**: `tid-api`
   - **SDK**: `Docker`
   - **Visibility**: `Public`
3. **Create Space** tıkla

### 1.3 Dosyaları Yükle

Space'i klonla ve dosyaları kopyala:

```bash
# Space'i klonla
git lfs install
git clone https://huggingface.co/spaces/hidrqx/tid-api
cd tid-api

# Proje dosyalarını kopyala

cp ../huggingface-api/app.py .
cp ../huggingface-api/requirements.txt .
cp ../huggingface-api/Dockerfile .
cp ../huggingface-api/README.md .

# Models klasörü oluştur
mkdir models

# Model dosyalarını kopyala (BÜYÜK DOSYALAR — Git LFS ile)
cp ../random_forest/models/model_tek_el.pkl models/
cp ../random_forest/models/model_iki_el.pkl models/
cp ../random_forest/models/le_tek_el.pkl models/
cp ../random_forest/models/le_iki_el.pkl models/
cp "../1d cnn/tid_70_kelime_cnn1d_best.h5" models/

# Git LFS ile büyük dosyaları işaretle
git lfs track "*.pkl"
git lfs track "*.h5"

# Commit ve push
git add .
git commit -m "TID API - RF + CNN1D modelleri"
git push
```

> ⚠️ **Not**: model_iki_el.pkl ~613 MB olduğu için yükleme birkaç dakika sürebilir.

### 1.4 Space URL'ini Not Al
Push sonrası Hugging Face Space otomatik build eder. URL şöyle olacak:
```
https://KULLANICI-ADIN-tid-api.hf.space
```

Build tamamlandığında tarayıcıda açıp `{"status":"ok",...}` görmelisin.

---

## Adım 2: Frontend'e API URL'ini Yaz

`TID/docs/js/app.js` dosyasını aç, **5. satırdaki** URL'yi değiştir:

```javascript
// ÖNCE:
const API_BASE = "https://YOUR-USERNAME-tid-api.hf.space";

// SONRA (kendi kullanıcı adınla):
const API_BASE = "https://semihesim-tid-api.hf.space";
```

---

## Adım 3: GitHub Pages'ı Aktifleştir

### 3.1 Değişiklikleri Commit Et
```bash
cd TID
git add docs/
git commit -m "GitHub Pages web sitesi"
git push origin main
```

### 3.2 GitHub Pages Ayarı
1. GitHub'da repo sayfasına git
2. **Settings** → sol menüden **Pages**
3. **Source** bölümünde:
   - **Branch**: `main`
   - **Folder**: `/docs`
4. **Save** tıkla

### 3.3 Site Yayında! 🎉
Birkaç dakika sonra siteniz şu adreste yayında olacak:
```
https://KULLANICI-ADIN.github.io/Turk-Isaret-Dili-Algilama-/
```

---

## ✅ Test Etme

1. Siteyi tarayıcıda aç
2. Sağ üstte **"API bağlı ✓"** yazısını kontrol et
3. **"Kamerayı Aç"** butonuna tıkla
4. Elinizi gösterip işaret yapın
5. Random Forest → anlık harf tahmini
6. 1D CNN → 3 saniyelik kayıt sonrası kelime tahmini

---

## 🔧 Sorun Giderme

| Sorun | Çözüm |
|---|---|
| "API bağlanamadı" | HF Space URL'ini kontrol et, Space'in build olduğundan emin ol |
| İlk istek yavaş (~30sn) | HF Space uyku modundaydı, uyanıyor. Normal. |
| Kamera açılmıyor | HTTPS gerekli. GitHub Pages zaten HTTPS kullanır. |
| MediaPipe yüklenmiyor | İnternet bağlantısını kontrol et |

---

## 💡 İpuçları

- HF Space **48 saat** kullanılmazsa uyur. İlk istekte ~30 saniye uyanma süresi olur.
- Ücretsiz tier'da **16 GB RAM** ve **2 vCPU** var — modeller rahat çalışır.
- Daha hızlı yanıt için HF Space'i **Upgraded** (ücretli) yapabilirsiniz.
