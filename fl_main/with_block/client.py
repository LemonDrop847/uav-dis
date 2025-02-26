import socket
import pickle
import torch
import os
import argparse
from models.arch import MNISTModel, CifarModel, DisasterModel
from models.train import train_model_with_dp
from models.test import test_model
from utils.utils import send_data, receive_data, get_model_hash
from utils.blockchain import sendhash
from utils.sampling import DatasetSplit
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="Client script for federated learning.")
parser.add_argument(
    "local_epochs", type=int, help="Number of local epochs for training"
)
parser.add_argument(
    "client_id", type=str, help="Client ID for identification with the server"
)
parser.add_argument("private_key", type=str, help="Private key of client wallet")
parser.add_argument("dataset", type=str, help="Dataset Type")
parser.add_argument("split_type", type=str, help="Split type")
parser.add_argument("dp_type", type=str, help="Differential Privacy algorithm")
args = parser.parse_args()

# Change/ Take input later
server_host = "localhost"
server_port = 8080
local_epochs = args.local_epochs  # input("Enter number of Local Epochs:")  # 10
client_id = args.client_id  # input("Enter your client ID: ")
split_type = args.split_type
private_key = args.private_key
dataset_type = args.dataset
dp_type = args.dp_type

model = MNISTModel()

if dataset_type == "MNIST":
    model = MNISTModel()
elif dataset_type == "CIFAR":
    model = CifarModel()
elif dataset_type == "DIS":
    model = DisasterModel()
else:
    raise ModuleNotFoundError()

dp_dict = {}
if dp_type == "no_dp":
    dp_dict = {"mechanism": "no_dp", "clip": None, "epsilon": None, "delta": None}
elif dp_type == "gaussian":
    dp_dict = {"mechanism": "Gaussian", "clip": 1.0, "epsilon": 1.0, "delta": 1e-5}
elif dp_type == "laplace":
    dp_dict = {"mechanism": "Laplace", "clip": 1.0, "epsilon": 1.0, "delta": None}
else:
    raise AttributeError("DP Type not specified")


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

prev_epoch_stats = ""
prev_epoch = 0


def client():
    global train_loader
    global prev_epoch
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn:
        print(f"Connecting to server at {server_host}:{server_port}")
        conn.connect((server_host, server_port))

        start_msg = conn.recv(1024).decode()
        print("Server:", start_msg)
        if start_msg == "Please send your ID":
            send_data(conn, client_id, True)

        data = receive_data(conn, True)
        client_indices = pickle.loads(data)
        train_loader = DataLoader(
            DatasetSplit(dataset=dataset_type, idxs=client_indices),
            batch_size=64 if dataset_type == "DIS" else 32,
            shuffle=True,
        )

        start_msg = conn.recv(1024).decode()
        global_epoch = 0
        while True:
            global_epoch += 1
            print(f"Waiting to receive model for global epoch {global_epoch}")
            global_weights = receive_data(conn, True)
            if global_weights is None:
                print("No weights received. Ending training.")
                break
            global_epoch = global_weights["global_epoch"]
            model.load_state_dict(global_weights["weights"])

            print(f"Starting local training for global epoch {global_epoch}")
            model.load_state_dict(
                train_model_with_dp(model, local_epochs, train_loader, dp_dict)
            )

            avg_loss, accuracy, precision, recall, f1 = test_model(
                model=model, dataset=dataset_type
            )
            print(
                f"Local model after Global Epoch {global_epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy * 100:.2f}%"
            )
            if prev_epoch == global_epoch - 1:
                os.makedirs(f"./logs/client/{dataset_type}", exist_ok=True)
                with open(
                    f"./logs/client/{dataset_type}/{split_type}_client_log_{client_id}.txt",
                    "a",
                ) as f:
                    f.write(
                        f"Epoch {global_epoch}: Loss = {avg_loss}, Accuracy = {accuracy}, Precison = {precision}, Recall = {recall}, F1 = {f1}\n"
                    )
            prev_epoch = global_epoch
            modelhash = get_model_hash(model.state_dict())
            sendhash(client_id, private_key, modelhash, global_epoch)
            print(f"Sending updated weights to server for global epoch {global_epoch}")
            localdata = {"weights": model.state_dict(), "local_epochs": local_epochs}
            send_data(conn, localdata, True)

            ack = conn.recv(1024)
            if ack == b"Training Complete":
                print("Training complete signal received from server.")
                break
            elif ack != b"Update received":
                print("Unexpected response from server:", ack)
                break


if __name__ == "__main__":
    client()
