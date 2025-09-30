import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Dataset class
# ----------------------------
class SpectrogramDataset(Dataset):
    """Dataset for loading real and fake spectrograms"""
    def __init__(self, real_spectrograms, fake_spectrograms):
        self.real = torch.FloatTensor(np.array(real_spectrograms))
        self.fake = torch.FloatTensor(np.array(fake_spectrograms))
        
        # Add channel dimension if needed
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

# ----------------------------
# Discriminator class
# ----------------------------
class Discriminator(nn.Module):
    """Discriminator network for classifying spectrograms as real or fake."""
    def __init__(self, img_channels=1, features_d=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, features_d, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(features_d, features_d*2, 4, 2, 1),
            nn.BatchNorm2d(features_d*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(features_d*2, features_d*4, 4, 2, 1),
            nn.BatchNorm2d(features_d*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(features_d*4, features_d*8, 4, 2, 1),
            nn.BatchNorm2d(features_d*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(features_d*8, 1, 4, 1, 0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.disc(x)
        out = nn.functional.adaptive_avg_pool2d(out, 1)
        return out.view(-1)

# ----------------------------
# GAN Trainer class
# ----------------------------
class SpectrogramGAN:
    """Trainer for the discriminator only."""
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

# ----------------------------
# Utility functions to load images
# ----------------------------
def load_spectrograms_from_folder(folder_path, size=(128, 128)):
    """Load images from a folder and return a list of normalized numpy arrays"""
    files = list(Path(folder_path).glob("*.*"))
    specs = []
    for f in files:
        img = Image.open(f).convert("L").resize(size)
        specs.append(np.array(img, dtype=np.float32)/255.0)
    return specs

def load_test_spectrograms(folder_path, size=(128, 128)):
    """Load test images from subfolders 'real' and 'fake'"""
    folder_path = Path(folder_path)
    real_folder = folder_path / "real"
    fake_folder = folder_path / "fake"

    real_specs = [np.array(Image.open(f).convert("L").resize(size), dtype=np.float32)/255.0
                  for f in real_folder.glob("*.*") if f.is_file()]
    fake_specs = [np.array(Image.open(f).convert("L").resize(size), dtype=np.float32)/255.0
                  for f in fake_folder.glob("*.*") if f.is_file()]

    if len(real_specs) == 0 and len(fake_specs) == 0:
        raise ValueError(f"No images found in {folder_path}")

    X = np.array(real_specs + fake_specs)
    y = np.array([1]*len(real_specs) + [0]*len(fake_specs))
    filenames = [f.name for f in list(real_folder.glob("*.*")) + list(fake_folder.glob("*.*"))]

    X_tensor = torch.FloatTensor(X)
    if len(X_tensor.shape) == 3:
        X_tensor = X_tensor.unsqueeze(1)
    y_tensor = torch.FloatTensor(y)
    return X_tensor, y_tensor, filenames

# ----------------------------
# Test function
# ----------------------------
def test_discriminator(model, X_test, y_test, filenames, device='cuda'):
    model.eval()
    device = device if torch.cuda.is_available() else 'cpu'
    model.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    with torch.no_grad():
        outputs = model(X_test)
        predictions = (outputs > 0.5).float()
        correct = (predictions == y_test).sum().item()
        accuracy = 100 * correct / len(y_test)
    
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Save results
    results_df = pd.DataFrame({
        "Filename": filenames,
        "Prediction": predictions.cpu().numpy(),
        "Label": y_test.cpu().numpy(),
        "Output_Prob": outputs.cpu().numpy()
    })
    results_df.to_csv("test_results.csv", index=False)
    print("Test results saved to test_results.csv")

    # Plot histogram
    plt.figure(figsize=(6,4))
    plt.hist(outputs.cpu().numpy()[y_test.cpu().numpy()==1], bins=20, alpha=0.6, label="Real")
    plt.hist(outputs.cpu().numpy()[y_test.cpu().numpy()==0], bins=20, alpha=0.6, label="Fake")
    plt.title("Discriminator Output Probabilities")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig("test_results_plot.png")
    plt.show()
    print("Plot saved to test_results_plot.png")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # --- Load training data ---
    base_dir = Path(r"C:\Users\hedif\OneDrive\Documents\GitHub\Real-vs-Fake-audio\spectrograms_output")
    real_specs = load_spectrograms_from_folder(base_dir / "real")
    fake_specs = load_spectrograms_from_folder(base_dir / "fake")
    print(f"Loaded {len(real_specs)} real and {len(fake_specs)} fake spectrograms.")

    # --- Create dataset and dataloader ---
    dataset = SpectrogramDataset(real_specs, fake_specs)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # --- Initialize GAN / discriminator ---
    gan = SpectrogramGAN(img_channels=1)
    num_epochs = 3

    # --- Training loop ---
    for epoch in range(num_epochs):
        loss, acc = gan.train_epoch(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss:.4f} Acc: {acc:.2f}%")

    # --- Save model ---
    gan.save_model("discriminator.pth")
    print("Model saved as discriminator.pth.")

    # --- Load and test folder with subfolders 'real' and 'fake' ---
    test_folder = r"C:\Users\hedif\OneDrive\Documents\GitHub\Real-vs-Fake-audio\spectrograms_output_test"
    X_test, y_test, filenames = load_test_spectrograms(test_folder)
    test_discriminator(gan.discriminator, X_test, y_test, filenames)
