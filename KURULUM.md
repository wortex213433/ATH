# 🚀 Kurulum ve Kullanım Kılavuzu

## 1. Kütüphaneleri Yükleyin

```bash
pip install -r requirements.txt
```

## 2. Veri Klasörü Hazırlayın

Görsellerinizi bir klasöre yerleştirin. Örnek:

```
C:\Users\Wortex\image_generation_ai\data\my_images\
├── image1.jpg
├── image2.jpg
├── image3.jpg
└── ...
```

**Önemli:** 
- En az 1000 görsel önerilir
- Görseller .jpg, .png, .bmp formatında olabilir
- Tüm görselleri tek bir klasöre koyun

## 3. Modeli Eğitin

```bash
python train.py --data_dir ./data/my_images --num_epochs 100 --batch_size 32
```

### Parametreler:

| Parametre | Açıklama | Varsayılan |
|-----------|----------|-----------|
| `--data_dir` | Görsellerin bulunduğu klasör (zorunlu) | - |
| `--num_epochs` | Epoch sayısı | 100 |
| `--batch_size` | Batch boyutu | 64 |
| `--lr` | Learning rate | 0.0002 |
| `--image_size` | Görsel boyutu (64 veya 128) | 64 |
| `--output_dir` | Çıktı klasörü | ./output |

### Örnek Komutlar:

**Başlangıç için (64x64 görsel):**
```bash
python train.py --data_dir ./data/my_images --num_epochs 50 --batch_size 64
```

**Daha kaliteli için (128x128):**
```bash
python train.py --data_dir ./data/my_images --num_epochs 200 --batch_size 16 --image_size 128
```

**GPU bellek sorunu için:**
```bash
python train.py --data_dir ./data/my_images --batch_size 8
```

## 4. Görsel Üretin

Eğitim tamamlandıktan sonra:

```bash
python generate.py --model_path ./output/models/generator_epoch_100.pth --num_images 10
```

### Parametreler:

| Parametre | Açıklama | Varsayılan |
|-----------|----------|-----------|
| `--model_path` | Model dosyası (zorunlu) | - |
| `--num_images` | Üretilecek görsel sayısı | 10 |
| `--output_dir` | Çıktı klasörü | ./generated |

### Örnek:

```bash
python generate.py --model_path ./output/models/generator_epoch_100.pth --num_images 20 --output_dir ./my_generated_images
```

## 5. Sonuçları Kontrol Edin

Eğitim sırasında:
- `output/samples/` - Her epoch'ta üretilen örnekler
- `output/models/` - Kaydedilen model dosyaları

Üretim sonrası:
- `generated/` - Üretilen görseller

## ⚡ Hızlı Başlangıç

1. Klasör yapısı oluşturun:
```bash
mkdir data\my_images
```

2. Görsellerinizi `data\my_images` klasörüne koyun

3. Eğitimi başlatın:
```bash
python train.py --data_dir ./data/my_images --num_epochs 10 --batch_size 16
```

4. Görsel üretin:
```bash
python generate.py --model_path ./output/models/generator_epoch_10.pth
```

## 💡 İpuçları

### Kaliteyi Artırmak İçin:
- Daha fazla veri kullanın (10,000+ görsel)
- Epoch sayısını artırın (200-500)
- Daha büyük görsel boyutu kullanın (128x128)
- Batch size'ı azaltın

### Sorun Giderme:

**"CUDA out of memory" hatası:**
```bash
# Batch size'ı azaltın
python train.py --data_dir ./data/my_images --batch_size 8
```

**Yavaş eğitim:**
```bash
# Worker sayısını azaltın
python train.py --data_dir ./data/my_images --num_workers 2
```

**Kötü kalite:**
- Daha fazla veri toplayın
- Daha uzun eğitin (500+ epoch)
- Farklı learning rate deneyin (0.0001 veya 0.0003)

## 📊 Eğitim İlerlemesi

Eğitim sırasında şunları göreceksiniz:
- `Loss_D`: Discriminator kaybı (düşük olmalı)
- `Loss_G`: Generator kaybı
- `D(x)`: Gerçek görsellere skor
- `D(G(z))`: Sahte görsellere skor

İyi eğitim için `D(x)` yüksek, `D(G(z))` düşük başlamalı ve sonra dengelenmeli.

## 🎨 Model Parametreleri

Modeli özelleştirmek için:

```bash
# Daha büyük latent space
python train.py --nz 128

# Daha güçlü modeller
python train.py --ngf 128 --ndf 128
```

