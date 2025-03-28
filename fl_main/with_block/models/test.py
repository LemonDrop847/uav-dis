import torch
import torch.nn as nn
import os
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score


def get_testset(dataset):
    if dataset == "DIS":
        trans_disaster = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        test_dataset = datasets.ImageFolder(
            os.path.join("./data/disaster", "test"), transform=trans_disaster
        )
        return DataLoader(test_dataset, batch_size=64, shuffle=False)


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


def test_dqn_model(
    agent, env, testing_log, global_epoch, n_episodes=15, max_steps=1000
):
    total_rewards = []
    success_percents = []
    optimal_distance = 2 * (env.grid_size - 1)
    with open(testing_log, "a") as f:
        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            for t in range(max_steps):
                action = agent.select_action(state, epsilon=0.0)
                next_state, reward, done = env.step(action)
                episode_reward += reward
                state = next_state
                if done:
                    break
            total_rewards.append(episode_reward)
            agent_pos = env.agent_pos
            current_distance = abs(agent_pos[0]) + abs(agent_pos[1])
            success_percent = min(100.0, (current_distance / optimal_distance) * 100.0)
            success_percents.append(success_percent)
            f.write(f"{global_epoch*100+episode},{episode_reward},{success_percent}\n")
    avg_reward = np.mean(total_rewards)
    avg_success = np.mean(success_percents)
    print(f"Testing: Avg Reward = {avg_reward:.2f}, Avg Success = {avg_success:.1f}%")
    return total_rewards, success_percents
