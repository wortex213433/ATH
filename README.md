ATH - DCGAN Implementation

Bu proje, Deep Convolutional Generative Adversarial Network (DCGAN) kullanarak görsel üreten bir yapay zeka modelini içerir.

## 🎯 Özellikler

- DCGAN mimarisi ile görsel üretimi
- PyTorch ile implementasyon
- Kolay kullanılabilir training ve generation scriptleri
- Özelleştirilebilir hiperparametreler

## 📦 Kurulum

### 1. Gerekli Kütüphaneleri Yükleyin

```bash
pip install -r requirements.txt
```

### 2. Veri Hazırlığı

Görsellerinizi bir klasöre yerleştirin. Örnek klasör yapısı:

```
data/
  ├── your_images/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
```

## 🚀 Kullanım

### Training (Model Eğitimi)

```bash
python train.py --data_dir path/to/your/images --num_epochs 100 --batch_size 64
```

#### Önemli Parametreler:

- `--data_dir`: Eğitim görsellerinizin bulunduğu klasör
- `--num_epochs`: Eğitim epoch sayısı (default: 100)
- `--batch_size`: Batch boyutu (default: 64)
- `--lr`: Learning rate (default: 0.0002)
- `--nz`: Latent vector boyutu (default: 100)
- `--image_size`: Görsel boyutu (default: 64x64)

#### Örnek Training Komutu:

```bash
python train.py --data_dir ./data/my_images --num_epochs 200 --batch_size 32 --lr 0.0002 --output_dir ./output
```

### Generation (Görsel Üretme)

Eğitilmiş model ile görsel üretmek için:

```bash
python generate.py --model_path ./output/models/generator_epoch_100.pth --num_images 10
```

#### Parametreler:

- `--model_path`: Eğitilmiş model dosyasının yolu
- `--num_images`: Üretilecek görsel sayısı
- `--output_dir`: Çıktı klasörü (default: ./generated)

## 🎨 Model Mimarisi

### Generator
- Input: 100 boyutlu noise vector
- Output: 64x64x3 RGB görsel
- Activation: ReLU + BatchNorm
- Son katman: Tanh (-1, 1 arası normalizasyon)

### Discriminator
- Input: 64x64x3 RGB görsel
- Output: Gerçek/Sahte skoru (0-1)
- Activation: LeakyReLU + BatchNorm
- Loss: Binary Cross Entropy

## 📊 Training İpuçları

1. **Veri Miktarı**: En az 1000 görsel önerilir
2. **Görsel Boyutu**: 64x64 veya 128x128 en iyi sonuç verir
3. **Epoch Sayısı**: Genellikle 50-200 epoch yeterlidir
4. **GPU Kullanımı**: CUDA destekli GPU önerilir
5. **Learning Rate**: 0.0002 genellikle iyi çalışır

## 🔧 Hiperparametre Ayarı

### Daha İyi Sonuçlar İçin:

```python
# Daha büyük model
python train.py --ngf 128 --ndf 128 --batch_size 32

# Daha uzun eğitim
python train.py --num_epochs 500 --save_every 10

# Farklı latent boyutu
python train.py --nz 128

# Daha büyük görseller (128x128)
python train.py --image_size 128
```

## 📁 Proje Yapısı

```
.
├── models/
│   └── dcgan.py          # Generator ve Discriminator modelleri
├── data/
│   └── dataset.py        # Veri yükleme modülü
├── train.py              # Eğitim scripti
├── generate.py           # Görsel üretme scripti
├── requirements.txt      # Gerekli kütüphaneler
└── README.md             # Bu dosya
```

## 🐛 Sorun Giderme

### CUDA Memory Hatası
```bash
# Batch size'ı düşürün
python train.py --batch_size 16
```

### Yavaş Training
```bash
# Worker sayısını artırın
python train.py --num_workers 8
```

### Kötü Kalite
- Daha fazla veri toplayın
- Epoch sayısını artırın
- Farklı hiperparametreler deneyin

## 📚 Referanslar

- [DCGAN Paper](https://arxiv.org/abs/1511.06434)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 📝 Lisans

Bu proje açık kaynak kodlu bir implementasyondur.

**wortex213433**

