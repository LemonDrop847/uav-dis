import torch
import pickle
import copy
import hashlib
from web3 import Web3


def average_weights(weights):
    total_data_points = sum(dataset_size for _, dataset_size in weights)
    w_avg = copy.deepcopy(weights[0][0])
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key])

        for client_weights, dataset_size in weights:
            w_avg[key] += client_weights[key] * (dataset_size / total_data_points)

    return w_avg


def send_data(conn, data, isclient):
    serialized_data = pickle.dumps(data)
    data_len = len(serialized_data)
    conn.sendall(data_len.to_bytes(4, byteorder="big"))
    conn.sendall(serialized_data)
    if isclient:
        print("Data sent to server")
    else:
        print(f"Sent Global weights to client {conn.getpeername()}")


def receive_data(conn, isclient):
    try:
        data_len_bytes = conn.recv(4)
        if len(data_len_bytes) < 4:
            print("Error: Insufficient length bytes received.")
            return None

        data_len = int.from_bytes(data_len_bytes, byteorder="big")
        # if isclient:
        #     print(f"Expecting {data_len} bytes from server")
        # else:
        #     print(f"Expecting {data_len} bytes from client {conn.getpeername()}")

        if data_len <= 0 or data_len > 10 * 1024 * 1024:
            print(f"Received invalid data length: {data_len}")
            return None

        data = b""
        while len(data) < data_len:
            packet = conn.recv(min(data_len - len(data), 4096))
            if not packet:
                # print("Error: Connection lost or incomplete data")
                return None
            data += packet
            # print(f"Received packet of size {len(packet)} bytes")
        # if isclient:
        #     print("Complete data received from server")
        # else:
        #     print(f"Data of size {len(data)} received from client {conn.getpeername()}")
        return pickle.loads(data)

    except Exception as e:
        print(f"Error receiving data: {e}")
        return None


def get_model_hash(state_dict):
    byte_stream = torch.save(state_dict, "_buffer.pt")
    with open("_buffer.pt", "rb") as f:
        byte_stream = f.read()
    sha256 = hashlib.sha256()
    sha256.update(byte_stream)
    return sha256.hexdigest()


def compare_hash(hash_a, hash_b):
    t = Web3.to_bytes(hexstr=hash_b)
    return hash_a == t
