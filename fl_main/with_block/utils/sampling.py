import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def mnist_iid(dataset, num_clients):
    num_items = int(len(dataset) / num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_clients):
    num_shards, num_imgs = num_clients * 2, int(len(dataset) / (num_clients * 2))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype="int64") for i in range(num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
            )
    return dict_users


def mnist_noniid_unequal(dataset, num_clients):
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype=int) for i in range(num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    min_shard, max_shard = 1, 30
    random_shard_size = np.random.randint(min_shard, max_shard + 1, size=num_clients)
    random_shard_size = np.around(
        random_shard_size / sum(random_shard_size) * num_shards
    ).astype(int)

    if sum(random_shard_size) > num_shards:

        for i in range(num_clients):
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )

        random_shard_size = random_shard_size - 1

        for i in range(num_clients):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )
    else:
        for i in range(num_clients):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )

        if len(idx_shard) > 0:
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            for rand in idx_shard:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )
    return dict_users


def cifar_iid(dataset, num_clients):
    num_items = int(len(dataset) / num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_clients):
    num_shards = num_clients * 2
    num_imgs = int(len(dataset) / (num_shards))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype="int64") for i in range(num_clients)}
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
            )
    return dict_users


def cifar_noniid_unequal(dataset, num_clients):
    num_shards = 1200
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype=int) for i in range(num_clients)}
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    # Sort indices based on labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    min_shard, max_shard = 1, 30
    random_shard_size = np.random.randint(min_shard, max_shard + 1, size=num_clients)
    random_shard_size = np.around(
        random_shard_size / sum(random_shard_size) * num_shards
    ).astype(int)

    if sum(random_shard_size) > num_shards:
        for i in range(num_clients):
            if not idx_shard:
                break
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )
        random_shard_size -= 1
        for i in range(num_clients):
            if not idx_shard:
                break
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )
    else:
        for i in range(num_clients):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )
        if idx_shard:
            k = min(dict_users, key=lambda x: len(dict_users[x]))
            for rand in idx_shard:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )
    return dict_users

def disaster_iid(dataset, num_clients):
    num_items = int(len(dataset) / num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def disaster_noniid(dataset, num_clients):
    num_shards = num_clients * 2
    num_imgs = int(len(dataset) / (num_shards))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype="int64") for i in range(num_clients)}
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
            )
    return dict_users


def disaster_noniid_unequal(dataset, num_clients):
    num_shards = 200
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype=int) for i in range(num_clients)}
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    min_shard, max_shard = 1, 10
    random_shard_size = np.random.randint(min_shard, max_shard + 1, size=num_clients)
    random_shard_size = np.around(
        random_shard_size / sum(random_shard_size) * num_shards
    ).astype(int)

    if sum(random_shard_size) > num_shards:
        for i in range(num_clients):
            if not idx_shard:
                break
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )
        random_shard_size -= 1
        for i in range(num_clients):
            if not idx_shard:
                break
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )
    else:
        for i in range(num_clients):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )
        if idx_shard:
            k = min(dict_users, key=lambda x: len(dict_users[x]))
            for rand in idx_shard:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )
    return dict_users


mnist_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
mnist_dataset = datasets.MNIST(
    root="./data/mnist", train=True, download=True, transform=mnist_transform
)

cifar_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
cifar_dataset = datasets.CIFAR10(
    root="./data/cifar", train=True, download=True, transform=cifar_transform
)

disaster_transform = transforms.Compose(
    [
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

disaster_dataset = datasets.ImageFolder(
    os.path.join("./data/disaster", "train"), transform=disaster_transform
)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        if dataset == "DIS":
            self.dataset = disaster_dataset
        elif dataset == "MNIST":
            self.dataset = mnist_dataset
        elif dataset == "CIFAR":
            self.dataset = cifar_dataset
        else:
            raise NotImplementedError()
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        data, label = self.dataset[self.idxs[index]]
        return torch.tensor(data), torch.tensor(label)


def get_indices(dataset, split_type, num_clients):
    if dataset == "MNIST":
        if split_type == "iid":
            return mnist_iid(mnist_dataset, num_clients)
        elif split_type == "noniid":
            return mnist_noniid(mnist_dataset, num_clients)
        elif split_type == "noniid_unequal":
            return mnist_noniid_unequal(mnist_dataset, num_clients)
        else:
            raise NotImplementedError("Invalid Split type")
    if dataset == "DIS":
        if split_type == "iid":
            return disaster_iid(disaster_dataset, num_clients)
        elif split_type == "noniid":
            return disaster_noniid(disaster_dataset, num_clients)
        elif split_type == "noniid_unequal":
            return disaster_noniid_unequal(disaster_dataset, num_clients)
        else:
            raise NotImplementedError("Invalid Split type")
    if dataset == "CIFAR":
        if split_type == "iid":
            return cifar_iid(cifar_dataset, num_clients)
        elif split_type == "noniid":
            return cifar_noniid(cifar_dataset, num_clients)
        elif split_type == "noniid_unequal":
            return cifar_noniid_unequal(cifar_dataset, num_clients)
        else:
            raise NotImplementedError("Invalid Split type")
    else:
        raise ValueError("Invalid Dataset chosen.")
