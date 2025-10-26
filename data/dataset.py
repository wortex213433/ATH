"""
Custom dataset class for image loading
"""

import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ImageDataset(Dataset):
    """Custom dataset for loading images from a directory"""
    
    def __init__(self, root_dir, image_size=64, transform=None):
        self.root_dir = root_dir
        self.image_size = image_size
        
        # Get all image file paths
        self.image_files = []
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        if os.path.isdir(root_dir):
            for file in os.listdir(root_dir):
                if file.lower().endswith(supported_formats):
                    self.image_files.append(os.path.join(root_dir, file))
        
        # Define default transform if not provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transform
        
        print(f"Loaded {len(self.image_files)} images from {root_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

