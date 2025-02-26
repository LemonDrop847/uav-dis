import os
import time
import torch
import pickle
import argparse
from torchvision import datasets, transforms
from models.arch import MNISTModel, CifarModel
from models.train import train_model_with_dp
from models.test import test_model
from torch.utils.data import DataLoader
from utils.plotter import plot_metrics

parser = argparse.ArgumentParser(description="Script for baseline learning.")
parser.add_argument(
    "global_epochs", type=int, help="Number of global communication rounds"
)
parser.add_argument("dataset_type", type=str, help="Dataset type")
args = parser.parse_args()
global_epochs = args.global_epochs
dataset_type = args.dataset_type
cifar_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
cifar_dataset = datasets.CIFAR10(
    root="./data/cifar", train=True, download=True, transform=cifar_transform
)
mnist_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
mnist_dataset = datasets.MNIST(
    root="./data/mnist", train=True, download=True, transform=mnist_transform
)

full_dataset = mnist_dataset if (dataset_type == "MNIST") else cifar_dataset
train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)

model = MNISTModel() if (dataset_type == "MNIST") else CifarModel()

dp_dict = {"mechanism": "no_dp", "clip": None, "epsilon": None, "delta": None}

def main():
    epoch = 1
    start_time = time.time()
    print("Starting training...")
    while epoch <= global_epochs:
        print(f"\n--- Global Epoch {epoch}/{global_epochs} ---\n")

        model.load_state_dict(train_model_with_dp(model, 1, train_loader, dp_dict))
        
        avg_loss, accuracy, precision, recall, f1 = test_model(
            model=model, dataset=dataset_type
        )
        time_taken = time.time() - start_time
        print(
            f"After Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy * 100:.2f}%, Precision = {precision:.2f}%, Recall = {recall:.2f}% , F1 = {f1:.2f}%, Time= {time_taken}"
        )
        os.makedirs(f"./logs/server/{dataset_type}", exist_ok=True)
        with open(
            f"./logs/server/{dataset_type}/server_log_{global_epochs}.txt", "a"
        ) as f:
            f.write(
                f"Epoch {epoch}: Loss = {avg_loss}, Accuracy = {accuracy}, Precison = {precision}, Recall = {recall}, F1 = {f1}, Time = {time_taken}\n"
            )
        epoch+=1

main()