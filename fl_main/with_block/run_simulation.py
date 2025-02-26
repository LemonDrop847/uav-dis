import json
import subprocess
import os

with open("settings.json", "r") as file:
    config = json.load(file)

num_clients = config["num_clients"]
global_epochs = config["global_epochs"]
dataset = config["dataset"]
dataset_split_type = config["dataset_split_type"]

if os.name == "nt":
    start_command = ["start", "cmd", "/k"]
else:
    start_command = ["gnome-terminal", "--"]

server_command = f"conda activate flenv && python server.py {num_clients} {global_epochs} {dataset} {dataset_split_type}"
subprocess.Popen(start_command + [server_command], shell=(os.name == "nt"))
print(f"Started server with command: {server_command}")

for i in range(num_clients):
    client = config["clients"][i]
    client_id = client["client_id"]
    local_epochs = client["local_epochs"]
    private_key = client["private_key"]
    dp_type = client["dp_type"]

    client_command = f"conda activate flenv && python client.py {local_epochs} {client_id} {private_key} {dataset} {dataset_split_type} {dp_type} && exit"
    subprocess.Popen(start_command + [client_command], shell=(os.name == "nt"))
    print(f"Started client {client_id} with command: {client_command}")
