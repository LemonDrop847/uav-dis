import os
import re
import glob
import matplotlib
import matplotlib.pyplot as plt


def parse_log_file(file_path):
    data = {
        "Epoch": [],
        "Loss": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
    }

    with open(file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        match = re.match(
            r"Epoch (\d+): Loss = ([\d.]+), Accuracy = ([\d.]+), Precison = ([\d.]+), Recall = ([\d.]+), F1 = ([\d.]+)",
            line,
        )
        if match:
            epoch, loss, accuracy, precision, recall, f1 = match.groups()
            data["Epoch"].append(int(epoch))
            data["Loss"].append(float(loss))
            data["Accuracy"].append(float(accuracy))
            data["Precision"].append(float(precision))
            data["Recall"].append(float(recall))
            data["F1"].append(float(f1))

    return data


def plot_contribution_scores(contribution_scores, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(contribution_scores)
    epochs = list(range(1, len(contribution_scores) + 1))
    clients = contribution_scores[0].keys()

    for i, client in enumerate(clients):
        scores = [epoch_scores[client] for epoch_scores in contribution_scores]
        plt.plot(epochs, scores, label=f"Client {chr(ord('A') + i)}")

    plt.xlabel("Global Communication Rounds")
    plt.ylabel("Contribution Score")
    plt.title("Weighted Contribution Score per Client")
    plt.legend()
    plt.grid(True)
    filename = f"{output_dir}/Client Contribution Scores.png"
    plt.savefig(filename)
    plt.close()


def generate_client_table(client_datasets, client_epochs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    client_ids = list(client_datasets.keys())
    dataset_sizes = [len(client_datasets[client]) for client in client_ids]
    local_epochs = [client_epochs[client] for client in client_ids]

    table_data = [["Client", "Client Address", "Dataset Size", "Local Epochs"]]
    for i, client in enumerate(client_ids):
        table_data.append(
            [chr(ord("A") + i), client, dataset_sizes[i], local_epochs[i]]
        )

    n_rows = len(client_ids) + 1
    n_cols = len(table_data[0])
    fig_width = max(4, n_cols * 4)
    fig_height = max(1.5, n_rows * 1)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=table_data, loc="center", cellLoc="center", colLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width([0, 1, 2])

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#40466e")
            cell.set_text_props(color="w", weight="bold")
        else:
            cell.set_facecolor("#f1f3f8")

    plt.title("Client Distribution", fontsize=14, weight="bold")
    filename = f"{output_dir}/Client Information Table.png"
    plt.savefig(filename)
    plt.close()
    print(f"Client Info saved")


def plot_dataset_size_distribution(client_datasets, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    clients = list(client_datasets.keys())
    client_labels = []
    for i, client in enumerate(clients):
        client_labels.append(chr(ord("A") + i))
    dataset_sizes = [len(client_datasets[client]) for client in clients]

    plt.bar(client_labels, dataset_sizes, color="skyblue")
    plt.xlabel("Clients")
    plt.ylabel("Dataset Size")
    plt.title("Dataset Size Distribution per Client")
    filename = f"{output_dir}/Client Dataset Chart.png"
    plt.savefig(filename)
    plt.close()
    print(f"Client Info saved")

    plt.pie(dataset_sizes, labels=client_labels, autopct="%1.1f%%", startangle=140)
    plt.title("Dataset Size Distribution per Client")
    plt.axis("equal")
    filename = f"{output_dir}/Client Dataset Distribution.png"
    plt.savefig(filename)
    plt.close()
    print(f"Client Info saved")


def plot_server_metric(x_label, y_label, x_data, y_data, output_dir):
    matplotlib.use("Agg")
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, color="r")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{y_label} vs {x_label}")
    filename = f"{output_dir}/{x_label}v{y_label}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Graph saved as {filename}")


def plot_client_metric(xlabel, ylabel, client_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))

    for client_id, data in client_data.items():
        plt.plot(data["Epoch"], data[ylabel], label=f"Client {client_id}")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {xlabel} for Multiple Clients")
    plt.legend()
    filename = f"{output_dir}/{xlabel}v{ylabel}_multiclient.png"
    plt.savefig(filename)
    plt.close()
    print(f"Graph saved as {filename}")


def plot_graphs(split_type, output_dir, dataset_dict, local_epochs):
    log_files = glob.glob(f"logs/server/DIS/{split_type}_server_log*.txt")
    if not log_files:
        print("No server_log files found in the current directory.")
        return

    server_file_path = log_files[0]
    print(f"Processing server file: {server_file_path}")

    data = parse_log_file(server_file_path)
    for metric in ["Loss", "Accuracy", "Precision", "Recall", "F1"]:
        plot_server_metric(
            "Global Communication Rounds",
            metric,
            data["Epoch"],
            data[metric],
            output_dir,
        )

    client_log_files = glob.glob(f"logs/client/DIS/{split_type}_client_log_*.txt")
    if not client_log_files:
        print("No client log files found in the current directory.")
        return

    client_data = {}

    for file_path in client_log_files:
        client_id_match = re.search(r"client_log_(0x[a-fA-F0-9]+)\.txt$", file_path)
        if client_id_match:
            client_id = client_id_match.group(1)
            title = (
                str(local_epochs[client_id]) + "-" + str(len(dataset_dict[client_id]))
            )
            print(f"Processing file: {file_path} for Client {title}")
            client_data[title] = parse_log_file(file_path)

    for metric in ["Loss", "Accuracy", "Precision", "Recall", "F1"]:
        plot_client_metric(
            "Global Communication Rounds", metric, client_data, output_dir
        )


import numpy as np


def parse_log(file_path):
    """
    Reads a CSV log file with format:
       epoch,reward,metric
    Returns:
       raw_epochs: the raw epoch values (numpy array)
       rewards: the reward values (numpy array)
       metric: the third column (e.g., epsilon in training logs or success in test logs)
    """
    try:
        data = np.loadtxt(file_path, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        raw_epochs = data[:, 0]
        rewards = data[:, 1]
        metric = data[:, 2]
        return raw_epochs, rewards, metric
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None


def get_tick_positions_and_labels(raw_epochs, epoch_scale):
    """
    Returns tick positions (indices) and tick labels only for entries where raw_epoch % epoch_scale == 0.
    Labels are formatted as "globalEpoch-0".
    """
    x_positions = np.arange(len(raw_epochs))
    tick_positions = [
        i for i, epoch in enumerate(raw_epochs) if (epoch % epoch_scale) == 0
    ]
    tick_labels = [
        f"{int(epoch // epoch_scale)}-0"
        for epoch in raw_epochs
        if (epoch % epoch_scale) == 0
    ]
    return x_positions, tick_positions, tick_labels


def plot_server_test_results(log_file="logs/server/DQN/server_testing_log.txt", dqn="DQN"):
    """
    Reads the server test log and plots:
      1. Success vs Global Epochs
      2. Reward vs Global Epochs
    X-axis ticks are placed only at the start of each local epoch (epoch % 100 == 0).
    """
    if not os.path.exists(log_file):
        print(f"Server log file {log_file} not found!")
        return

    raw_epochs, rewards, success = parse_log(log_file)
    if raw_epochs is None:
        return

    x_positions, tick_positions, tick_labels = get_tick_positions_and_labels(
        raw_epochs, epoch_scale=100
    )

    # Plot Success vs Global Epochs
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(raw_epochs)), success, color="green", linestyle="-")
    plt.xlabel("Global Epoch - Local Episode")
    plt.ylabel("Success (%)")
    plt.title(f"{dqn} Server Test: Success vs Global Epochs")
    plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"server_test_success {dqn}.png")
    print("Saved server_test_success.png")
    plt.show()

    # Plot Reward vs Global Epochs
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(raw_epochs)), rewards, color="blue", linestyle="-")
    plt.xlabel("Global Epoch - Local Episode")
    plt.ylabel("Reward")
    plt.title(f"{dqn} Server Test: Reward vs Global Epochs")
    plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"server_test_reward {dqn}.png")
    print("Saved server_test_reward.png")
    plt.show()


def plot_client_test_results(log_pattern="logs/client/dqn/test/*.txt", dqn="DQN"):
    """
    Reads multiple client test logs and plots:
      1. Success vs Global Epochs
      2. Reward vs Global Epochs
    Each client's log is plotted as a separate line.
    X-axis ticks are placed only at the start of each local epoch (epoch % 100 == 0).
    """
    test_files = glob.glob(log_pattern)
    if not test_files:
        print(f"No client test log files found with pattern {log_pattern}")
        return

    plt.figure(figsize=(8, 5))
    for file in test_files:
        raw_epochs, rewards, success = parse_log(file)
        if raw_epochs is None:
            continue
        x_positions, tick_positions, tick_labels = get_tick_positions_and_labels(
            raw_epochs, epoch_scale=100
        )
        client_id = os.path.splitext(os.path.basename(file))[0].split("_")[-1]
        plt.plot(
            np.arange(len(raw_epochs)),
            success,
            linestyle="-",
            label=f"Client {client_id}",
        )
    plt.xlabel("Global Epoch - Local Episode")
    plt.ylabel("Success (%)")
    plt.title(f"{dqn} Client Test: Success vs Global Epochs")
    plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"client_test_success {dqn}.png")
    print("Saved client_test_success.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    for file in test_files:
        raw_epochs, rewards, success = parse_log(file)
        if raw_epochs is None:
            continue
        x_positions, tick_positions, tick_labels = get_tick_positions_and_labels(
            raw_epochs, epoch_scale=100
        )
        client_id = os.path.splitext(os.path.basename(file))[0].split("_")[-1]
        plt.plot(
            np.arange(len(raw_epochs)),
            rewards,
            linestyle="-",
            label=f"Client {client_id}",
        )
    plt.xlabel("Global Epoch - Local Episode")
    plt.ylabel("Reward")
    plt.title(f"{dqn} Client Test: Reward vs Global Epochs")
    plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"client_test_reward {dqn}.png")
    print("Saved client_test_reward.png")
    plt.show()


def plot_client_train_logs(log_pattern="logs/clients/dqn/train/*.txt", dqn="DQN"):
    """
    Reads multiple client training logs and plots:
      1. Epsilon decay vs Global Epochs
      2. Training Reward vs Global Epochs
    Each client's log is plotted as a separate line.
    X-axis ticks are placed only at the start of each local epoch (epoch % 1000 == 0).
    """
    train_files = glob.glob(log_pattern)
    if not train_files:
        print(f"No client training log files found with pattern {log_pattern}")
        return

    plt.figure(figsize=(8, 5))
    for file in train_files:
        raw_epochs, rewards, epsilon = parse_log(file)
        if raw_epochs is None:
            continue
        x_positions, tick_positions, tick_labels = get_tick_positions_and_labels(
            raw_epochs, epoch_scale=1000
        )
        client_id = os.path.splitext(os.path.basename(file))[0].split("_")[-1]
        plt.plot(
            np.arange(len(raw_epochs)),
            epsilon,
            linestyle="-",
            label=f"Client {client_id}",
        )
    plt.xlabel("Global Epoch - Local Episode")
    plt.ylabel("Epsilon")
    plt.title(f"{dqn} Client Training: Epsilon Decay vs Global Epochs")
    plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"client_train_epsilon_decay {dqn}.png")
    print("Saved client_train_epsilon_decay.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    for file in train_files:
        raw_epochs, rewards, epsilon = parse_log(file)
        if raw_epochs is None:
            continue
        x_positions, tick_positions, tick_labels = get_tick_positions_and_labels(
            raw_epochs, epoch_scale=1000
        )
        client_id = os.path.splitext(os.path.basename(file))[0].split("_")[-1]
        plt.plot(
            np.arange(len(raw_epochs)),
            rewards,
            linestyle="-",
            label=f"Client {client_id}",
        )
    plt.xlabel("Global Epoch - Local Episode")
    plt.ylabel("Reward")
    plt.title(f"{dqn} Client Training: Reward vs Global Epochs")
    plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"client_train_reward {dqn}.png")
    print("Saved client_train_reward.png")
    plt.show()


def plot_metrics(dataset_dict, local_epochs, contrib_list, split_type):
    output_dir = f"./metrics/DIS/{split_type}"
    generate_client_table(dataset_dict, local_epochs, output_dir)
    plot_dataset_size_distribution(dataset_dict, output_dir)
    plot_contribution_scores(contrib_list, output_dir)
    plot_graphs(split_type, output_dir, dataset_dict, local_epochs)
    plot_server_test_results(f"logs/server/DQN/10.txt", dqn="DQN")
    plot_server_test_results(f"logs/server/DDQN/10.txt", dqn="DDQN")
    plot_client_test_results("logs/client/DQN/test/*.txt", dqn="DQN")
    plot_client_test_results("logs/client/DDQN/test/*.txt", dqn="DDQN")
    plot_client_train_logs("logs/client/DQN/train/*.txt", dqn="DQN")
    plot_client_train_logs("logs/client/DDQN/train/*.txt", dqn="DDQN")
