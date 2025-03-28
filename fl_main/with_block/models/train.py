import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


def train_model_with_dp(model, local_epochs, train_loader, dp_params):
    """
    dp_params (dict): Dictionary containing DP parameters:
        - 'mechanism' (str): Type of DP mechanism ('no_dp', 'Laplace', 'Gaussian').
        - 'clip' (float): Clipping threshold for gradients.
        - 'epsilon' (float): Privacy budget (epsilon) for DP.
        - 'delta' (float, optional): Privacy budget (delta) for Gaussian DP.
    """
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mechanism = dp_params.get("mechanism", "no_dp")
    clip_threshold = dp_params.get("clip", 1.0)
    epsilon = dp_params.get("epsilon", 1.0)
    delta = dp_params.get("delta", 1e-5) if mechanism == "Gaussian" else None

    model.to(device)
    model.train()
    epoch_loss = []

    for epoch in range(local_epochs):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            log_probs = model(images)
            loss = criterion(log_probs, labels)
            loss.backward()
            # Gradient clipping for DP
            if mechanism != "no_dp":
                per_param_norms = [
                    param.grad.detach().reshape(-1).norm(2)
                    for param in model.parameters()
                ]
                global_norm = torch.sqrt(sum(norm**2 for norm in per_param_norms))
                clip_factor = clip_threshold / (global_norm + 1e-6)
                clip_factor = min(1.0, clip_factor)
                for param in model.parameters():
                    param.grad.data.mul_(clip_factor)

            optimizer.step()

            # Add noise for DP
            if mechanism == "Laplace":
                sensitivity = clip_threshold / len(train_loader.dataset)
                for param in model.parameters():
                    noise = torch.from_numpy(
                        np.random.laplace(0, sensitivity / epsilon, param.shape)
                    ).to(device)
                    param.data.add_(noise)
            elif mechanism == "Gaussian":
                sensitivity = clip_threshold / len(train_loader.dataset)
                std_dev = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
                for param in model.parameters():
                    noise = torch.normal(0, std_dev, size=param.shape).to(device)
                    param.data.add_(noise)

            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        print(
            f"Local epoch {epoch + 1}/{local_epochs}, Epoch Loss: {sum(epoch_loss) / len(epoch_loss)}"
        )

    return model.state_dict()


def train_dqn_model(
    # self,
    client_id,
    agent,
    env,
    training_log,
    global_epoch,
    n_episodes=500,
    max_steps=1000,
    target_update=10,
    epsilon_start=2.0,
    epsilon_end=0.1,
    epsilon_decay=0.995,
):
    epsilon = epsilon_start
    with open(training_log, "a") as f:
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            for t in range(max_steps):
                action = agent.select_action(state, epsilon)
                next_state, reward, done = env.step(action)
                total_reward += reward
                agent.memory.push(
                    (state, action, reward, next_state if not done else None, done)
                )
                state = next_state
                agent.optimize_model()
                if done:
                    break
            if episode % target_update == 0:
                agent.update_target_network()
                f.write(f"{(global_epoch*1000)+episode},{total_reward},{epsilon}\n")
            epsilon = max(epsilon * epsilon_decay, epsilon_end)
    print(f"Client {client_id} finished local training.")
    return total_reward
