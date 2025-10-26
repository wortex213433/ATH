"""
Training script for DCGAN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import argparse
import os
from tqdm import tqdm

from models.dcgan import Generator, Discriminator, weights_init
from data.dataset import ImageDataset


def train_dcgan(args):
    """Main training loop for DCGAN"""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
    
    # Create dataset and dataloader
    dataset = ImageDataset(args.data_dir, image_size=args.image_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Initialize networks
    netG = Generator(nz=args.nz, ngf=args.ngf, nc=args.nc).to(device)
    netD = Discriminator(nc=args.nc, ndf=args.ndf).to(device)
    
    # Initialize weights
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)
    
    # Training labels
    real_label = 1.0
    fake_label = 0.0
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.num_epochs):
        for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")):
            
            # Update Discriminator
            netD.zero_grad()
            
            # Train with real images
            real_data = data.to(device)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            
            output = netD(real_data)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            # Train with fake images
            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # Update Generator
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            # Print statistics every 100 batches
            if (i + 1) % 100 == 0:
                print(f'[{epoch}/{args.num_epochs}][{i+1}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
        
        # Save generated samples
        if (epoch + 1) % args.save_every == 0:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                save_image(
                    fake,
                    os.path.join(args.output_dir, 'samples', f'fake_samples_epoch_{epoch+1}.png'),
                    nrow=8,
                    normalize=True
                )
        
        # Save models
        if (epoch + 1) % args.save_every == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': netG.state_dict(),
                    'optimizer_state_dict': optimizerG.state_dict(),
                },
                os.path.join(args.output_dir, 'models', f'generator_epoch_{epoch+1}.pth')
            )
    
    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train DCGAN')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='path to dataset')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='path to output directory')
    
    # Model arguments
    parser.add_argument('--nz', type=int, default=100,
                        help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64,
                        help='generator features')
    parser.add_argument('--ndf', type=int, default=64,
                        help='discriminator features')
    parser.add_argument('--nc', type=int, default=3,
                        help='number of channels (RGB=3)')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='training batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for Adam optimizer')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--image_size', type=int, default=64,
                        help='image size')
    parser.add_argument('--save_every', type=int, default=5,
                        help='save model every N epochs')
    
    args = parser.parse_args()
    
    train_dcgan(args)


if __name__ == '__main__':
    main()

