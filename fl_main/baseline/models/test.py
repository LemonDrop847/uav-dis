import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score


def get_testset(dataset):
    if dataset == "MNIST":
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        test_dataset = datasets.MNIST(
            root="./data/mnist", train=False, download=True, transform=trans_mnist
        )
        return DataLoader(test_dataset, batch_size=32, shuffle=False)
    elif dataset == "CIFAR":
        trans_cifar = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        test_dataset = datasets.CIFAR10(
            "./data/cifar", train=False, download=True, transform=trans_cifar
        )
        return DataLoader(test_dataset, batch_size=32, shuffle=False)


def test_model(model, dataset):
    test_loader = get_testset(dataset)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss, total, correct = 0.0, 0.0, 0.0
    all_labels = []
    all_preds = []
    # correct = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        all_preds.extend(pred_labels.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return loss / len(test_loader), accuracy, precision, recall, f1
