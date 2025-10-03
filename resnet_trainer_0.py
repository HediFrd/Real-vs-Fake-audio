import csv
from misc_utils import Spectrogram_Dataset
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torchvision.transforms as tvt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Dataset settings
train_real_root = r"D:\Deep_Detect\dataset\training\real"
train_fake_root = r"D:\Deep_Detect\dataset\training\fake"
test_real_root = r"D:\Deep_Detect\dataset\testing\real"
test_fake_root = r"D:\Deep_Detect\dataset\testing\fake"
holdout_root = r"D:\Deep_Detect\dataset\holdout"
holdout_csv = r"C:\Users\jeremy\Python_Projects\Real-vs-Fake-audio\holdout_csv"
train_dur_range = [5, 5]  # Range of audio target durations for training
test_dur_range = [5, 5]  # Range of audio target durations for testing
aud_rand_start = True  # Whether to start the audio sampling at a random point
spec_size = (224, 224)  # Number of frequencies and number of windows in the spectrogram
max_freq_list = [8000, 8000, 2000]  # Maximum frequency on the spectrogram scale - list of 1 or 3 elements
dB_min_list = [None, -100, None]  # Decibel minimum - list of 1 or 3 elements, None means no limit applied
dB_max_list = [None, -60, None]  # Decibel maximum - list of 1 or 3 elements, None means no limit applied

# Image transforms applied to spectrograms during training/testing
train_transform = tvt.Compose([
        tvt.RandomApply([tvt.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.25),
        tvt.RandomApply([tvt.GaussianBlur(kernel_size=9, sigma=(0.2, 1.0))], p=0.25),
        tvt.RandomHorizontalFlip(p=0.1),
        tvt.ToTensor()
    ])
test_transform = tvt.Compose([tvt.ToTensor()])
holdout_transform = test_transform

# Training settings
model_state_dict = None  # If you want to load a pretrained model file, write the file path here
n_epochs = 10  # Number of training epochs
batch_size = 16  # Training and evaluation batch size
init_lr = 0.5  # Initial learning rate before cosine decay
weight_decay = 1e-4  # Weight decay
momentum = 0.9  # Momentum

def main():

    # Define train/test datasets and dataloaders
    train_dataset = Spectrogram_Dataset(train_real_root, train_fake_root, train_dur_range, aud_rand_start, spec_size, max_freq_list, dB_min_list, dB_max_list, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset = Spectrogram_Dataset(test_real_root, test_fake_root, test_dur_range, aud_rand_start, spec_size, max_freq_list, dB_min_list, dB_max_list, test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # holdout_dataset = Spectrogram_Dataset(holdout_root, None, test_dur_range, aud_rand_start, spec_size, max_freq_list, dB_min_list, dB_max_list, test_transform)
    # holdout_dataloader = DataLoader(holdout_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Define model
    model = resnet18(weights=None, num_classes=2).to(device)
    if model_state_dict is not None:
        model.load_state_dict(torch.load(model_state_dict))
    model.train()

    # Define optimizer and LR scheduler
    optimizer = optim.SGD(model.parameters(), lr=init_lr, weight_decay=weight_decay, momentum=momentum)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * n_epochs)

    # Initial eval
    eval_loop(model, test_dataloader)

    # Training loop
    best_test_acc = 0.5
    for epoch in range(n_epochs):
        train_one_epoch(epoch, model, train_dataloader, optimizer, lr_scheduler)

        # Eval after each epoch
        _, _, _, _, curr_test_acc = eval_loop(model, test_dataloader)
        if curr_test_acc > best_test_acc:
            print("Saving new best model\n")
            torch.save(model.state_dict(), f"saved_models/best_resnet_statedict_0.pth")
            best_test_acc = curr_test_acc

    # holdout_loop(model, holdout_dataloader, holdout_dataset, holdout_csv)

def train_one_epoch(epoch, model, train_dataloader, optimizer, lr_scheduler):
    print("\nTraining 1 epoch")

    model.train()
    for ii, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = torch.nn.functional.cross_entropy(output, labels)
        print(f"Epoch {epoch + 1}/{n_epochs} | Batch {ii}/{len(train_dataloader)} | Loss {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

def eval_loop(model, test_dataloader):
    print("\nRunning evaluation")

    n_real_correct = 0
    n_real = 0
    n_fake_correct = 0
    n_fake = 0
    model.eval()
    with torch.no_grad():
        for ii, (images, labels) in enumerate(test_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            where_real = torch.where(labels == 1)[0]
            n_real_correct += torch.sum(torch.argmax(output, dim=1)[where_real] == labels[where_real]).item()
            n_real += len(where_real)
            where_fake = torch.where(labels == 0)[0]
            n_fake_correct += torch.sum(torch.argmax(output, dim=1)[where_fake] == labels[where_fake]).item()
            n_fake += len(where_fake)

    n_real_incorrect = n_real - n_real_correct
    n_fake_incorrect = n_fake - n_fake_correct
    print(f"Number of real samples correctly classified: {n_real_correct} ({n_real_correct / n_real * 100}%)")
    print(f"Number of real samples incorrectly classified: {n_real_incorrect} ({n_real_incorrect / n_real * 100}%)")
    print(f"Number of fake samples correctly classified: {n_fake_correct} ({n_fake_correct / n_fake * 100}%)")
    print(f"Number of fake samples incorrectly classified: {n_fake_incorrect} ({n_fake_incorrect / n_fake * 100}%)")
    eval_acc = (n_real_correct + n_fake_correct) / (n_real + n_fake)
    print(f"Test Accuracy: {eval_acc}\n")
    return n_real_correct, n_real_incorrect, n_fake_correct, n_fake_incorrect, eval_acc

def holdout_loop(model, holdout_dataloader, holdout_dataset, holdout_csv):
    print("\nEvaluating holdout")

    model_preds = []
    model.eval()
    with torch.no_grad():
        for ii, (images, labels) in enumerate(holdout_dataloader):
            images = images.to(device)
            output = model(images)
            model_preds.extend(torch.argmax(output, dim=1).tolist())

    files_list = [os.path.basename(aud_file) for aud_file in holdout_dataset.real_paths]
    with open(holdout_csv, "w", newline="") as f:
        out_writer = csv.writer(f, delimiter=",")
        for file_pred in zip(files_list, model_preds):
            out_writer.writerow([file_pred[0], file_pred[1]])

if __name__ == "__main__":
    main()
