from dotenv import load_dotenv
import os
from web3 import Web3
from utils.abi import contract_abi, distribute_abi

load_dotenv()

node_url = "http://127.0.0.1:8545"


web3 = Web3(Web3.HTTPProvider(node_url))
contract_address = os.getenv("CONTRACT_ADDRESS")
distribute_contract_address = os.getenv("DISTRIBUTE_ADDRESS")
Chain_id = web3.eth.chain_id
contract = ""
server_address = os.getenv("SERVER_ADDRESS")
server_pkey = os.getenv("SERVER_PKEY")

if web3.is_connected():
    print("-" * 50)
    print("Connection Successful")
    print("-" * 50)
    contract = web3.eth.contract(address=contract_address, abi=contract_abi)
    distribute_contract = web3.eth.contract(
        address=distribute_contract_address, abi=distribute_abi
    )


def reward_pool():
    deposit_amount = web3.to_wei(0.001, "ether")
    nonce = web3.eth.get_transaction_count(server_address)
    reward_pool = distribute_contract.functions.depositReward().build_transaction(
        {
            "chainId": Chain_id,
            "from": server_address,
            "value": deposit_amount,
            "nonce": nonce,
        }
    )
    signed_tx = web3.eth.account.sign_transaction(reward_pool, private_key=server_pkey)
    send_tx = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
    tx_receipt = web3.eth.wait_for_transaction_receipt(send_tx)
    print(f"{deposit_amount} added to pool")


def sendhash(clientid, pkey, hash_str, round):
    # clientid: Wallet address, hash:Weights hash round: global round
    try:

        hash_bytes32 = Web3.to_bytes(hexstr=hash_str)

        if len(hash_bytes32) != 32:
            raise ValueError("Hash is not 32 bytes long.")

        nonce = web3.eth.get_transaction_count(clientid)
        send_hash = contract.functions.addHashContribution(
            clientid, hash_bytes32
        ).build_transaction({"chainId": Chain_id, "from": clientid, "nonce": nonce})
        signed_tx = web3.eth.account.sign_transaction(send_hash, private_key=pkey)
        send_tx = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_receipt = web3.eth.wait_for_transaction_receipt(send_tx)
        print(f"Hash transfer successful for {clientid} for round {round}")
        # print("Here is the tx receipt:")
        # print(tx_receipt)

    except Exception as e:
        print(f"Error sending hash: {e}")
        return None


def gethash(clientid):
    try:
        get_hash = contract.functions.getDeviceHashes(clientid).call()
        return get_hash[-1]
    except Exception as e:
        print(f"Error receiving hash: {e}")
        return None


def sendcontribution(clientid, contributionscore):
    try:
        contributionscore = int(contributionscore * 10**18)
        print(f"Sent contribution of client {clientid}: {contributionscore}")

        nonce = web3.eth.get_transaction_count(server_address)
        send_contribution = contract.functions.addContribution(
            contributionscore, clientid
        ).build_transaction(
            {"chainId": Chain_id, "from": server_address, "nonce": nonce}
        )
        signed_tx = web3.eth.account.sign_transaction(
            send_contribution, private_key=server_pkey
        )
        send_tx = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_receipt = web3.eth.wait_for_transaction_receipt(send_tx)
        print(f"Contribution added successfully!")
        # print("Here is the tx receipt:")
        # print(tx_receipt)
    except Exception as e:
        print(f"Error sending data: {e}")
        return None


def distributeReward():
    try:
        nonce = web3.eth.get_transaction_count(server_address)
        distribute_reward = (
            distribute_contract.functions.calculateReward().build_transaction(
                {"chainId": Chain_id, "from": server_address, "nonce": nonce}
            )
        )
        signed_tx = web3.eth.account.sign_transaction(
            distribute_reward, private_key=server_pkey
        )
        send_tx = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_receipt = web3.eth.wait_for_transaction_receipt(send_tx)
    except Exception as e:
        print(f"Error distributing reward: {e}")
        return None


def getReward(clientid):
    try:
        balance1 = web3.eth.get_balance(clientid)
        nonce = web3.eth.get_transaction_count(server_address)
        distribute_reward = distribute_contract.functions.distributeReward(
            clientid
        ).build_transaction(
            {"chainId": Chain_id, "from": server_address, "nonce": nonce}
        )
        signed_tx = web3.eth.account.sign_transaction(
            distribute_reward, private_key=server_pkey
        )
        send_tx = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_receipt = web3.eth.wait_for_transaction_receipt(send_tx)
        print("reward distributed successfully")
        balance2 = web3.eth.get_balance(clientid)
        reward_claimed = web3.from_wei(balance2 - balance1, "ether")
        print(f"Reward for client {clientid} : {reward_claimed}")
    except Exception as e:
        print(f"Error distributing reward: {e}")
        return None
