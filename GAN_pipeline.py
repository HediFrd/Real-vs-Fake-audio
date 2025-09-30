import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image


class SpectrogramDataset(Dataset):
    """Dataset for loading real and fake spectrograms"""
    def __init__(self, real_spectrograms, fake_spectrograms):
        self.real = torch.FloatTensor(np.array(real_spectrograms))
        self.fake = torch.FloatTensor(np.array(fake_spectrograms))
        
        if len(self.real.shape) == 3:
            self.real = self.real.unsqueeze(1)
        if len(self.fake.shape) == 3:
            self.fake = self.fake.unsqueeze(1)
        
        self.n_real = len(self.real)
        self.n_fake = len(self.fake)
        
    def __len__(self):
        return self.n_real + self.n_fake
    
    def __getitem__(self, idx):
        if idx < self.n_real:
            return self.real[idx], torch.tensor(1.0)  # Real = 1
        else:
            return self.fake[idx - self.n_real], torch.tensor(0.0)  # Fake = 0


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_channels=1, features_d=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, features_d, 4, 2, 1),
            nn.LeakyReLU(0.2), #first layer

            nn.Conv2d(features_d, features_d*2, 4, 2, 1),
            nn.BatchNorm2d(features_d*2),
            nn.LeakyReLU(0.2),#second layer

            nn.Conv2d(features_d*2, features_d*4, 4, 2, 1),
            nn.BatchNorm2d(features_d*4),
            nn.LeakyReLU(0.2),#third layer

            nn.Conv2d(features_d*4, features_d*8, 4, 2, 1),
            nn.BatchNorm2d(features_d*8),
            nn.LeakyReLU(0.2),#final layer

            nn.Conv2d(features_d*8, 1, 4, 1, 0),
            nn.Sigmoid() #output
        )
    
    def forward(self, x):
        out = self.disc(x)
        out = nn.functional.adaptive_avg_pool2d(out, 1)
        return out.view(-1)


# GAN Trainer
class SpectrogramGAN:
    def __init__(self, img_channels=1, features_d=64, lr=2e-4, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.discriminator = Discriminator(img_channels, features_d).to(self.device)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()
        
    def train_epoch(self, dataloader):
        self.discriminator.train()
        epoch_loss, correct, total = 0.0, 0, 0
        
        for spectrograms, labels in dataloader:
            spectrograms, labels = spectrograms.to(self.device), labels.to(self.device)
            self.optimizer_d.zero_grad()
            
            outputs = self.discriminator(spectrograms)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer_d.step()
            
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            epoch_loss += loss.item()
            
        return epoch_loss / len(dataloader), 100 * correct / total
    
    def save_model(self, path):
        torch.save(self.discriminator.state_dict(), path)


# Load Spectrogram Images
def load_spectrograms_from_folder(folder_path, size=(128, 128)):
    files = list(Path(folder_path).glob("*.*"))  # load all image types
    specs = []
    for f in files:
        img = Image.open(f).convert("L").resize(size)
        specs.append(np.array(img, dtype=np.float32) / 255.0)
    return specs


if __name__ == "__main__":
    base_dir = Path(r"C:\Users\hedif\OneDrive\Documents\GitHub\Real-vs-Fake-audio\spectrograms_output")
    real_specs = load_spectrograms_from_folder(base_dir / "real")
    fake_specs = load_spectrograms_from_folder(base_dir / "fake")
    
    print(f"Loaded {len(real_specs)} real and {len(fake_specs)} fake spectrograms.")
    
    dataset = SpectrogramDataset(real_specs, fake_specs)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    gan = SpectrogramGAN(img_channels=1)
    num_epochs = 50
    
    for epoch in range(num_epochs):
        loss, acc = gan.train_epoch(train_loader)
        # Clear previous output for a cleaner display
        print(f"\rEpoch [{epoch+1}/{num_epochs}] Loss: {loss:.4f} Acc: {acc:.2f}%", end="")
    print("\nTraining done.")
    
    gan.save_model("discriminator.pth")
    print("Model saved as discriminator.pth.")
