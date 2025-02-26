import os
import re
import matplotlib.pyplot as plt


# Function to parse the log file
def parse_log_file(file_path):
    data = {
        "Epoch": [],
        "Loss": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "Time": [],
    }

    with open(file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        match = re.match(
            r"Epoch (\d+): Loss = ([\d.]+), Accuracy = ([\d.]+), Precison = ([\d.]+), Recall = ([\d.]+), F1 = ([\d.]+), Time = ([\d.]+)",
            line,
        )
        if match:
            epoch, loss, accuracy, precision, recall, f1, time_taken = match.groups()
            data["Epoch"].append(int(epoch))
            data["Loss"].append(float(loss))
            data["Accuracy"].append(float(accuracy))
            data["Precision"].append(float(precision))
            data["Recall"].append(float(recall))
            data["F1"].append(float(f1))
            data["Time"].append(float(time_taken))

    return data


def plot_comparison(non_block_data, block_dict, metric, output_dir=f"comparison_plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))

    plt.plot(
        non_block_data["Epoch"],
        non_block_data[metric],
        label="non-FL",
        # marker="o",
    )
    for split_type, block_data in block_dict.items():
        plt.plot(
            block_data["Epoch"],
            block_data[metric],
            label=f"FL with ChainFed {split_type} Split",
            # marker="s",
        )

    plt.title(f"{metric} Comparison")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{metric}_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved {metric} comparison plot at: {plot_path}")


def compare_logs(
    non_block_log_path, block_log_iid, block_log_noniid, block_log_noniid_unequal
):
    non_block_data = parse_log_file(non_block_log_path)
    block_iid = parse_log_file(block_log_iid)
    # block_noniid = parse_log_file(block_log_noniid)
    block_noniid_unequal = parse_log_file(block_log_noniid_unequal)
    block_dict = {
        "iid": block_iid,
        # "noniid": block_noniid,
        "noniid_unequal": block_noniid_unequal,
    }
    metrics = ["Loss", "Accuracy", "Precision", "Recall", "F1", "Time"]
    output_dir = f"comparison_plots/{dataset_type}/combined"
    for metric in metrics:
        plot_comparison(non_block_data, block_dict, metric, output_dir)


if __name__ == "__main__":
    # split_type = "iid"
    global_epochs = 50
    dataset_type = "CIFAR"
    non_block_log_path = (
        f"./baseline/logs/server/{dataset_type}/server_log_{global_epochs}.txt"
    )
    # block_log_path = f"./with_block/logs/server/{dataset_type}/{split_type}_server_log_{global_epochs}.txt"
    block_log_iid = (
        f"./with_block/logs/server/{dataset_type}/iid_server_log_{global_epochs}.txt"
    )
    block_log_noniid = (
        f"./with_block/logs/server/{dataset_type}/noniid_server_log_{global_epochs}.txt"
    )
    block_log_noniid_unequal = f"./with_block/logs/server/{dataset_type}/noniid_unequal_server_log_{global_epochs}.txt"
    compare_logs(
        non_block_log_path, block_log_iid, block_log_noniid, block_log_noniid_unequal
    )
