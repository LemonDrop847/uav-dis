import socket
import os
import threading
import time
import torch
import pickle
import argparse
from torchvision import datasets, transforms
from models.arch import DisasterModel, DQN
from models.test import test_model, test_dqn_model
from utils.utils import (
    average_weights,
    average_nav,
    send_data,
    receive_data,
    get_model_hash,
    compare_hash,
)
from utils.nav_utils import GridWorld, ReplayBuffer, DQNAgent, DoubleDQNAgent
from utils.contrib import calculate_contribution
from utils.blockchain import (
    gethash,
    sendcontribution,
    distributeReward,
    reward_pool,
    getReward,
)
from utils.sampling import get_indices
from utils.plotter import plot_metrics

parser = argparse.ArgumentParser(description="Server script for federated learning.")
parser.add_argument("num_clients", type=int, help="Number of clients")
parser.add_argument(
    "global_epochs", type=int, help="Number of global communication rounds"
)
parser.add_argument("dataset_type", type=str, help="Dataset type")
parser.add_argument("split_type", type=str, help="Dataset Split type")
args = parser.parse_args()

# Server settings
host = "localhost"
port = 8080
num_clients = args.num_clients  # int(input("Enter the number of clients: "))
global_epochs = args.global_epochs  # int(input("Enter the number of global epochs: "))
split_type = (
    args.split_type
)  # input("Enter 'iid', 'noniid', or 'noniid_unequal' for dataset split: ")
dataset_type = args.dataset_type

env = GridWorld(grid_size=10, n_static=15, n_dynamic=5)
replay_buffer = ReplayBuffer(capacity=10000)

device = "cuda" if torch.cuda.is_available() else "cpu"

dis_model = DisasterModel()
dqn_agent = DQNAgent(input_channels=4, n_actions=env.n_actions, device=device)
ddqn_agent = DoubleDQNAgent(input_channels=4, n_actions=env.n_actions, device=device)

model = DisasterModel()

if dataset_type == "DIS":
    model = DisasterModel()
else:
    raise ModuleNotFoundError()

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
full_dataset = datasets.ImageFolder(
    os.path.join("./data/disaster", "train"), transform=disaster_transform
)

if dataset_type == "DIS":
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
    full_dataset = datasets.ImageFolder(
        os.path.join("./data/disaster", "train"), transform=disaster_transform
    )
else:
    raise ModuleNotFoundError()


clients = []
client_dict = {}
client_add_dict = {}
client_data_indices = get_indices(dataset_type, split_type, num_clients)
accuracy_dict = {}
loss_dict = {}
dataset_dict = {}
local_epochs = {}
contrib_list = []
complete_dataset_size = 0


model.to(device)

global_epoch = 0
epoch_event = threading.Event()
lock = threading.Lock()
init_complete = False


def client_handler(conn, addr):
    global init_complete
    global complete_dataset_size
    print(f"Client {addr} attempting to connect...")

    conn.sendall(b"Please send your ID")
    client_id = receive_data(conn, False)
    time.sleep(1)
    if client_id:
        client_dict[conn] = client_id
        client_add_dict[addr] = client_id
        clients.append(conn)
        print(f"Client {client_id} ({addr}) joined successfully.")
    else:
        print(f"Client at {addr} did not send an valid ID and will be disconnected.")
        conn.close()
        return

    while len(clients) < num_clients:
        time.sleep(1)

    curr = 0
    for clientid in client_dict:
        dataset_dict[client_dict[clientid]] = client_data_indices[curr]
        complete_dataset_size += len(dataset_dict[client_dict[clientid]])
        curr += 1
    time.sleep(2)
    print(f"Sending dataset indices to {client_id}")
    send_data(conn, pickle.dumps(dataset_dict[client_dict[conn]]), False)

    init_complete = True
    conn.sendall(b"All clients connected. Starting training.")

    for epoch in range(global_epochs):
        epoch_event.wait()

        print(
            f"Sending model to client {client_dict[conn]} for global epoch {global_epoch}"
        )
        globalweights = {
            "global_epoch": global_epoch,
            "dis_weights": model.state_dict(),
            "dqn_weights": dqn_agent.policy_net.state_dict(),
            "ddqn_weights": ddqn_agent.policy_net.state_dict(),
        }
        send_data(conn, globalweights, False)

        try:
            print(f"Attempting to receive weights from client {client_dict[conn]}...")
            localweights = receive_data(conn, False)
            if localweights:
                with lock:
                    local_epochs[client_add_dict[addr]] = localweights["local_epochs"]
                    client_data[addr] = {
                        "dis_weights": localweights["dis_weights"],
                        "dqn_weights": localweights["dqn_weights"],
                        "ddqn_weights": localweights["ddqn_weights"],
                    }
                conn.sendall(b"Update received")
            else:
                print(f"No weights received from client {client_dict[conn]}")
        except Exception as e:
            print(f"Error receiving data from client {client_dict[conn]}: {e}")


def federated_training():
    reward_pool()
    global global_epoch
    epoch = 0
    start_time = time.time()
    print("Starting federated training...")
    while epoch <= global_epochs:
        with lock:
            global_epoch = epoch
            epoch_event.clear()
            print(f"\n--- Global Epoch {epoch}/{global_epochs} ---\n")

        if global_epoch != 0:
            while len(client_data) < num_clients:
                time.sleep(0.1)

        local_weights_dict = {}
        local_weights = []
        dqn_weights_dict = {}
        dqn_weights = []
        ddqn_weights_dict = {}
        ddqn_weights = []
        with lock:
            for client, weights in client_data.items():
                # print("data from ", client_add_dict[client])
                local_weights_dict[client_add_dict[client]] = weights["dis_weights"]
                local_weights.append(
                    (weights["dis_weights"], len(dataset_dict[client_add_dict[client]]))
                )
                dqn_weights_dict[client_add_dict[client]] = weights["dqn_weights"]
                dqn_weights.append((weights["dqn_weights"]))
                ddqn_weights_dict[client_add_dict[client]] = weights["ddqn_weights"]
                ddqn_weights.append((weights["ddqn_weights"]))
            client_data.clear()

        verif = True
        if epoch != 0:
            for client, weights in local_weights_dict.items():
                # if gethash(client) != get_model_hash(weights):
                # print(client)
                if compare_hash(gethash(client), get_model_hash(weights)) == False:
                    verif = False

        if verif:
            print(f"Data verified for epoch {epoch}")

            if local_weights:
                averaged_weights = average_weights(local_weights)
                avg_dqn_weights = average_nav(dqn_weights)
                avg_ddqn_weights = average_nav(ddqn_weights)
                model.load_state_dict(averaged_weights)
                dqn_agent.policy_net.load_state_dict(avg_dqn_weights)
                dqn_agent.target_net.load_state_dict(avg_dqn_weights)
                ddqn_agent.policy_net.load_state_dict(avg_ddqn_weights)
                ddqn_agent.target_net.load_state_dict(avg_ddqn_weights)
                print(f"Global model updated for epoch {epoch}.")
                torch.save(model.state_dict(), f"runs/dis/model_{epoch}.pt")
                torch.save(
                    dqn_agent.policy_net.state_dict(), f"runs/nav/model_{epoch}.pt"
                )
                torch.save(
                    ddqn_agent.policy_net.state_dict(), f"runs/nav/model_{epoch}.pt"
                )
            else:
                print(f"No local weights received for epoch {epoch}.")

            avg_loss, accuracy, precision, recall, f1 = test_model(
                model=model, dataset=dataset_type
            )
            os.makedirs(f"logs/server/DQN/", exist_ok=True)
            os.makedirs(f"logs/server/DDQN/", exist_ok=True)
            dqn_reward, dqn_success = test_dqn_model(
                agent=dqn_agent,
                env=env,
                testing_log=f"logs/server/DQN/{global_epochs}.txt",
                global_epoch=global_epoch,
            )
            ddqn_reward, ddqn_success = test_dqn_model(
                agent=ddqn_agent,
                env=env,
                testing_log=f"logs/server/DDQN/{global_epochs}.txt",
                global_epoch=global_epoch,
            )
            time_taken = time.time() - start_time
            # if global_epoch != 0:
            accuracy_dict[global_epoch] = accuracy
            loss_dict[global_epoch] = avg_loss
            print(
                f"After Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy * 100:.2f}%, Precision = {precision:.2f}%, Recall = {recall:.2f}% , F1 = {f1:.2f}%, Time= {time_taken} "
            )
            if global_epoch != 0:
                os.makedirs(f"./logs/server/{dataset_type}", exist_ok=True)
                with open(
                    f"./logs/server/{dataset_type}/{split_type}_server_log_{global_epochs}.txt",
                    "a",
                ) as f:
                    f.write(
                        f"Epoch {epoch}: Loss = {avg_loss}, Accuracy = {accuracy}, Precison = {precision}, Recall = {recall}, F1 = {f1}, Time = {time_taken}\n"
                    )
            econtrib = {}
            for client, weight in local_weights_dict.items():
                # print(dataset_dict)
                clientdata = {
                    "weight": weight,
                    "dataset_size": len(dataset_dict[client]) / complete_dataset_size,
                    "local_epochs": local_epochs[client],
                }
                contribscore = calculate_contribution(
                    clientdata,
                    model.state_dict(),
                    accuracy_dict[global_epoch - 1],
                    accuracy,
                    loss_dict[global_epoch - 1],
                    avg_loss,
                )
                econtrib[client] = contribscore
                sendcontribution(client, contribscore)
            if global_epoch != 0:
                contrib_list.append(econtrib)
            epoch += 1
        else:
            print("Verification Failed, discarding Epoch, rerunning in 3 seconds......")
            time.sleep(3)
        epoch_event.set()

    distributeReward()

    plot_metrics(dataset_dict, local_epochs, contrib_list, split_type)

    for client in clients:
        getReward(client_dict[client])
        client.sendall(b"Training Complete")
        client.close()
    print("Training complete. Server shutting down.")


def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()
    print(f"Server started. Waiting for {num_clients} clients to connect...")

    while len(clients) < num_clients:
        # while init_complete==False:
        conn, addr = server_socket.accept()
        threading.Thread(target=client_handler, args=(conn, addr)).start()
        time.sleep(2)
        print(len(clients))
    while init_complete == False:
        time.sleep(1)
    federated_training()


if __name__ == "__main__":
    client_data = {}
    start_server()
