"""
Setup Check - Ensure all dependencies are installed
"""

import sys

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} - YÜKLENMELİ: pip install {package_name}")
        return False

def main():
    print("=" * 60)
    print("DCGAN Image Generation AI - Kurulum Kontrolü")
    print("=" * 60)
    print()
    
    required_packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("Pillow", "PIL"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
        ("matplotlib", "matplotlib"),
    ]
    
    print("Gerekli kütüphaneler kontrol ediliyor...")
    print()
    
    all_ok = True
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            all_ok = False
    
    print()
    
    if all_ok:
        print("=" * 60)
        print("🎉 Tüm kütüphaneler yüklü! Kurulum tamamlandı.")
        print("=" * 60)
        print()
        print("📝 SONRAKI ADIMLAR:")
        print()
        print("1. Görsellerinizi bir klasöre koyun (örn: ./data/my_images/)")
        print("2. Modeli eğitin:")
        print("   python train.py --data_dir ./data/my_images --num_epochs 100")
        print("3. Görsel üretin:")
        print("   python generate.py --model_path ./output/models/generator_epoch_100.pth")
        print()
    else:
        print("=" * 60)
        print("⚠️ Bazı kütüphaneler eksik!")
        print("=" * 60)
        print()
        print("Yüklemek için:")
        print("pip install -r requirements.txt")
        print()

if __name__ == '__main__':
    main()

