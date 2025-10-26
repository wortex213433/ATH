"""
Quick Start Example for DCGAN
Demonstrates basic usage of the image generation AI
"""

import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

def print_info():
    print("=" * 60)
    print("DCGAN Image Generation AI - Quick Start Guide")
    print("=" * 60)
    print()
    print("Bu AI modeli ile kendi görsellerinizi eğitebilirsiniz!")
    print()
    print("📝 NASIL BAŞLARSANIZ:")
    print()
    print("1. Görsellerinizi bir klasöre koyun")
    print("   Örnek: C:/Users/Wortex/image_generation_ai/my_images/")
    print()
    print("2. Kütüphaneleri yükleyin:")
    print("   pip install -r requirements.txt")
    print()
    print("3. Modeli eğitin:")
    print("   python train.py --data_dir ./my_images --num_epochs 100")
    print()
    print("4. Görsel üretin:")
    print("   python generate.py --model_path ./output/models/generator_epoch_100.pth --num_images 10")
    print()
    print("=" * 60)
    print()

def test_model():
    """Test the model architecture"""
    print("Testing model architecture...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Import models
    from models.dcgan import Generator, Discriminator
    
    # Test Generator
    netG = Generator(nz=100, ngf=64, nc=3).to(device)
    test_noise = torch.randn(1, 100, 1, 1, device=device)
    fake_image = netG(test_noise)
    print(f"✅ Generator works! Generated image shape: {fake_image.shape}")
    
    # Test Discriminator
    netD = Discriminator(nc=3, ndf=64).to(device)
    output = netD(fake_image)
    print(f"✅ Discriminator works! Output shape: {output.shape}")
    
    print()
    print("🎉 Model mimarisi başarıyla test edildi!")
    print()

if __name__ == '__main__':
    print_info()
    
    # Test if models work
    try:
        test_model()
    except Exception as e:
        print(f"⚠️ Model testi başarısız oldu: {e}")
        print("Kütüphaneleri yüklediğinizden emin olun: pip install -r requirements.txt")

