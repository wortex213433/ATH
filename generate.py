"""
Generate images from trained DCGAN model
"""

import torch
from torchvision.utils import save_image
import argparse
import os

from models.dcgan import Generator


def generate_images(args):
    """Generate images using trained model"""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize generator
    netG = Generator(nz=args.nz, ngf=args.ngf, nc=args.nc).to(device)
    
    # Load trained model
    checkpoint = torch.load(args.model_path, map_location=device)
    netG.load_state_dict(checkpoint['model_state_dict'])
    netG.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Generate images
    with torch.no_grad():
        for i in range(args.num_images):
            # Generate noise
            noise = torch.randn(1, args.nz, 1, 1, device=device)
            
            # Generate image
            fake_image = netG(noise).cpu()
            
            # Save image
            save_image(
                fake_image,
                os.path.join(args.output_dir, f'generated_image_{i+1}.png'),
                normalize=True
            )
    
    print(f"Generated {args.num_images} images in {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate images with DCGAN')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to trained generator model')
    parser.add_argument('--nz', type=int, default=100,
                        help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64,
                        help='generator features')
    parser.add_argument('--nc', type=int, default=3,
                        help='number of channels (RGB=3)')
    
    # Generation arguments
    parser.add_argument('--num_images', type=int, default=10,
                        help='number of images to generate')
    parser.add_argument('--output_dir', type=str, default='./generated',
                        help='path to output directory')
    
    args = parser.parse_args()
    
    generate_images(args)


if __name__ == '__main__':
    main()

