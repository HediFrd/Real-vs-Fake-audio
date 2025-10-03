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
train_real_root = r"../Datasets/Deep_Detect/dataset/training/real"
train_fake_root = r"../Datasets/Deep_Detect/dataset/training/fake"
test_real_root = r"../Datasets/Deep_Detect/dataset/testing/real"
test_fake_root = r"../Datasets/Deep_Detect/dataset/testing/fake"
holdout_root = r"../Datasets/Deep_Detect/dataset/holdout"
holdout_csv = r"outputs/holdout_4.csv"  # Write holdout results to here
train_dur_range = [1, 5]  # Range of audio target durations for training
test_dur_range = [3, 3]  # Range of audio target durations for testing
aud_rand_start = True  # Whether to start the audio sampling at a random point
spec_size = (224, 224)  # Number of frequencies and number of windows in the spectrogram
max_freq_list = [8000, 4000, 2000]  # Maximum frequency on the spectrogram scale - list of 1 or 3 elements
dB_min_list = [None, None, None]  # Decibel minimum - list of 1 or 3 elements, None means no limit applied
dB_max_list = [None, None, None]  # Decibel maximum - list of 1 or 3 elements, None means no limit applied

# Image transforms applied to spectrograms during training/testing
train_transform = tvt.Compose([
        tvt.RandomApply([tvt.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.25),
        tvt.RandomApply([tvt.GaussianBlur(kernel_size=9, sigma=(0.2, 1.0))], p=0.25),
        tvt.RandomHorizontalFlip(p=0.1),
        tvt.ToTensor()
    ])
test_transform = tvt.Compose([tvt.ToTensor()])
holdout_transform = test_transform

# Model settings
model_load_path = None  # Resume training/evaluation from the state_dict here
model_save_path = r"saved_models/best_resnet_statedict_4.pth"  # After training, save state_dict to here

# Training settings
n_epochs = 10  # Number of training epochs
batch_size = 16  # Training and evaluation batch size
init_lr = 0.5  # Initial learning rate before cosine decay
weight_decay = 1e-4  # Weight decay
momentum = 0.9  # Momentum

def main():

    # Define train/test datasets and dataloaders
    train_dataset = Spectrogram_Dataset(train_real_root, train_fake_root, train_dur_range, aud_rand_start, spec_size, max_freq_list, dB_min_list, dB_max_list, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = Spectrogram_Dataset(test_real_root, test_fake_root, test_dur_range, aud_rand_start, spec_size, max_freq_list, dB_min_list, dB_max_list, test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # holdout_dataset = Spectrogram_Dataset(holdout_root, None, test_dur_range, aud_rand_start, spec_size, max_freq_list, dB_min_list, dB_max_list, test_transform)
    # holdout_dataloader = DataLoader(holdout_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define model
    model = resnet18(weights=None, num_classes=2).to(device)
    if model_load_path is not None:
        model.load_state_dict(torch.load(model_load_path))
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
            torch.save(model.state_dict(), model_save_path)
            best_test_acc = curr_test_acc

    # Load best model and test it
    model.load_state_dict(torch.load(model_save_path))
    eval_loop(model, test_dataloader, n_eval_epochs=5)

    # holdout_loop(model, holdout_dataloader, n_eval_epochs=5, holdout_csv=holdout_csv)


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


def eval_loop(model, test_dataloader, n_eval_epochs=1):
    print(f"\nRunning evaluation for {n_eval_epochs} epoch(s)")

    model.eval()
    with torch.no_grad():

        epoch_preds_tens = torch.zeros(size=(n_eval_epochs, len(test_dataloader.dataset), 2))
        epoch_label_tens = torch.zeros(size=(len(test_dataloader.dataset),))
        for eval_epoch in range(n_eval_epochs):

            for ii, (images, labels) in enumerate(test_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)

                # Aggregate model predictions and labels
                epoch_preds_tens[eval_epoch, ii * batch_size:ii * batch_size + images.size(0), :] = output.detach().cpu()
                if eval_epoch == 0:
                    epoch_label_tens[ii * batch_size:ii * batch_size + images.size(0)] = labels.detach().cpu()

        # Sum the predictions across multiple epochs and get the max index
        final_preds = torch.argmax(torch.sum(epoch_preds_tens, dim=0, keepdim=False), dim=1, keepdim=False)

        where_real = torch.where(epoch_label_tens == 1)[0]
        where_fake = torch.where(epoch_label_tens == 0)[0]
        n_real_correct = torch.sum(final_preds[where_real] == epoch_label_tens[where_real]).item()
        n_real_incorrect = torch.sum(final_preds[where_real] != epoch_label_tens[where_real]).item()
        n_fake_correct = torch.sum(final_preds[where_fake] == epoch_label_tens[where_fake]).item()
        n_fake_incorrect = torch.sum(final_preds[where_fake] != epoch_label_tens[where_fake]).item()

    n_real = len(where_real)
    n_fake = len(where_fake)
    print(f"Number of real samples correctly classified: {n_real_correct} ({n_real_correct / n_real})")
    print(f"Number of real samples incorrectly classified: {n_real_incorrect} ({n_real_incorrect / n_real})")
    print(f"Number of fake samples correctly classified: {n_fake_correct} ({n_fake_correct / n_fake})")
    print(f"Number of fake samples incorrectly classified: {n_fake_incorrect} ({n_fake_incorrect / n_fake})")
    eval_acc = (n_real_correct + n_fake_correct) / (n_real + n_fake)
    print(f"Test Accuracy: {eval_acc}\n")
    return n_real_correct, n_real_incorrect, n_fake_correct, n_fake_incorrect, eval_acc


def holdout_loop(model, holdout_dataloader, n_eval_epochs=1, holdout_csv=None):
    print("\nEvaluating holdout")

    model.eval()
    with torch.no_grad():

        epoch_preds_tens = torch.zeros(size=(n_eval_epochs, len(holdout_dataloader.dataset), 2))
        for eval_epoch in range(n_eval_epochs):

            for ii, (images, labels) in enumerate(holdout_dataloader):
                images = images.to(device)
                output = model(images)

                # Aggregate model predictions
                epoch_preds_tens[eval_epoch, ii * batch_size:ii * batch_size + images.size(0), :] = output.detach().cpu()

        # Sum the predictions across multiple epochs and get the max index
        final_preds = torch.argmax(torch.sum(epoch_preds_tens, dim=0, keepdim=False), dim=1, keepdim=False)

    if holdout_csv is not None:
        files_list = [os.path.basename(aud_file) for aud_file in holdout_dataloader.dataset.real_paths]
        with open(holdout_csv, "w", newline="") as f:
            out_writer = csv.writer(f, delimiter=",")
            for file_pred in zip(files_list, final_preds.tolist()):
                out_writer.writerow([file_pred[0], file_pred[1]])


if __name__ == "__main__":
    main()
