import os
import glob
import re
import matplotlib.pyplot as plt

dataset = "CIFAR"


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


def compare_server_metrics(split_types, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    split_data = {}

    for split_type in split_types:
        log_files = glob.glob(f"logs/server/{dataset}/{split_type}_server_log*.txt")
        if not log_files:
            print(f"No server log files found for split type '{split_type}'.")
            continue

        server_file_path = log_files[0]
        print(
            f"Processing server file: {server_file_path} for split type '{split_type}'"
        )
        split_data[split_type] = parse_log_file(server_file_path)

    for metric in ["Loss", "Accuracy", "Precision", "Recall", "F1"]:
        plt.figure(figsize=(10, 6))

        for split_type, data in split_data.items():
            if metric in data:
                plt.plot(data["Epoch"], data[metric], label=f"{split_type} - {metric}")

        plt.xlabel("Global Communication Rounds")
        plt.ylabel(metric)
        plt.title(f"Comparison of {metric} Across Split Types")
        plt.legend()
        filename = f"{output_dir}/Comparison_{metric}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Comparison graph saved as {filename}")


compare_server_metrics(
    ["noniid", "iid", "noniid_unequal"], f"./metrics/{dataset}/comparison/"
)
