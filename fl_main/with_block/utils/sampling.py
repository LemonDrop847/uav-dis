import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

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
        else:
            raise NotImplementedError()
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        data, label = self.dataset[self.idxs[index]]
        return torch.tensor(data), torch.tensor(label)


def get_indices(dataset, split_type, num_clients):
    if dataset == "DIS":
        if split_type == "iid":
            return disaster_iid(disaster_dataset, num_clients)
        elif split_type == "noniid":
            return disaster_noniid(disaster_dataset, num_clients)
        elif split_type == "noniid_unequal":
            return disaster_noniid_unequal(disaster_dataset, num_clients)
        else:
            raise NotImplementedError("Invalid Split type")
    else:
        raise ValueError("Invalid Dataset chosen.")
