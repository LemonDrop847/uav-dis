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
    log_files = glob.glob(f"logs/server/CIFAR/{split_type}_server_log*.txt")
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

    client_log_files = glob.glob(f"logs/client/CIFAR/{split_type}_client_log_*.txt")
    if not client_log_files:
        print("No client log files found in the current directory.")
        return

    client_data = {}

    for file_path in client_log_files:
        client_id_match = re.search(r"client_log_(0x[a-fA-F0-9]+)\.txt$", file_path)
        if client_id_match:
            client_id = client_id_match.group(1)
            title = str(local_epochs[client_id])+"-"+str(len(dataset_dict[client_id])) 
            print(f"Processing file: {file_path} for Client {title}")
            client_data[title] = parse_log_file(file_path)

    for metric in ["Loss", "Accuracy", "Precision", "Recall", "F1"]:
        plot_client_metric(
            "Global Communication Rounds", metric, client_data, output_dir
        )


def plot_metrics(dataset_dict, local_epochs, contrib_list, split_type):
    output_dir = f"./metrics/CIFAR/{split_type}"
    generate_client_table(dataset_dict, local_epochs, output_dir)
    plot_dataset_size_distribution(dataset_dict, output_dir)
    plot_contribution_scores(contrib_list, output_dir)
    plot_graphs(split_type, output_dir, dataset_dict, local_epochs)
